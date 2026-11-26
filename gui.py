"""
fem_gui.py - Interaktive FEM mit Grafikfenster (Qt + PyVista)
=============================================================

Start:   python fem_gui.py                 (leeres Fenster, Bauteil im Fenster waehlen)
         python fem_gui.py bauteil.step    (Bauteil direkt laden)
         python fem_gui.py setup.json      (gespeichertes Setup laden)

Ablauf im Fenster
-----------------
1. Datei > Bauteil oeffnen ...   (STEP, IGES, STL, OBJ, PLY, VTK, MSH)  oder  Demo-Balken
2. Rechtsklick > Netzparameter ...  /  Vernetzen        (auch im Menue "Netz")
3. Knoten waehlen (Menue "Auswahl" oder Rechtsklick > Auswahlmodus):
     Knoten einzeln : Linksklick setzt/entfernt einen Knoten (Ctrl+Klick entfernt)
     Flaeche        : Linksklick waehlt die zusammenhaengende Flaeche (Winkel einstellbar)
     Rechteck       : Aufziehen eines Rechtecks (Ctrl = abwaehlen)
     Drehen         : nur Kamera (Linke Maustaste drehen, Mitte verschieben, Rad zoomen)
4. Rechtsklick > Lager aus Auswahl ... / Kraft aus Auswahl ... / Netz lokal verfeinern ...
5. Berechnen > Statisch  oder  Topologieoptimierung
6. Ergebnis: Ansicht > Schnittmodus (C), Feld (F); Datei > Ergebnis als VTK exportieren
7. Datei > Setup speichern (fuer Server) ...   ->  Berechnung ohne Grafik:
       python fem_core.py run setup.json
"""
import sys
from pathlib import Path

import numpy as np
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
import pyvista as pv
from pyvistaqt import QtInteractor

import fem_core as core
from fem_core import FemError


# ==========================================================================
# Kleine Dialog-Bausteine
# ==========================================================================
class Busy:
    """Sanduhr-Cursor waehrend langer Rechnungen."""

    def __enter__(self):
        QW.QApplication.setOverrideCursor(QG.QCursor(QC.Qt.WaitCursor))
        QW.QApplication.processEvents()

    def __exit__(self, *a):
        QW.QApplication.restoreOverrideCursor()


class OptFloat(QW.QWidget):
    """Zahlenfeld mit Haken 'automatisch' (Wert None)."""

    def __init__(self, value=None, default=1.0, auto_text="automatisch", decimals=4,
                 vmin=1e-9, vmax=1e12):
        super().__init__()
        lay = QW.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.chk = QW.QCheckBox(auto_text)
        self.spin = QW.QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(vmin, vmax)
        self.spin.setValue(default if value is None else value)
        self.chk.setChecked(value is None)
        self.spin.setEnabled(value is not None)
        self.chk.toggled.connect(lambda on: self.spin.setEnabled(not on))
        lay.addWidget(self.chk)
        lay.addWidget(self.spin)

    def value(self):
        return None if self.chk.isChecked() else self.spin.value()

    def setValue(self, v):
        self.chk.setChecked(v is None)
        if v is not None:
            self.spin.setValue(v)


class FormDialog(QW.QDialog):
    """Allgemeiner Eingabedialog. specs: (key, Beschriftung, art, standard[, optionen]);
    art: float | int | bool | choice | optfloat | text."""

    def __init__(self, parent, title, specs, info=None, presets=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QW.QVBoxLayout(self)
        if info:
            lab = QW.QLabel(info)
            lab.setWordWrap(True)
            lay.addWidget(lab)
        form = QW.QFormLayout()
        lay.addLayout(form)
        self.w = {}
        for spec in specs:
            key, label, kind, default = spec[:4]
            o = spec[4] if len(spec) > 4 else {}
            if kind == "float":
                w = QW.QDoubleSpinBox()
                w.setDecimals(o.get("decimals", 4))
                w.setRange(o.get("min", -1e12), o.get("max", 1e12))
                w.setValue(default)
            elif kind == "int":
                w = QW.QSpinBox()
                w.setRange(o.get("min", 0), o.get("max", 100000))
                w.setValue(int(default))
            elif kind == "bool":
                w = QW.QCheckBox()
                w.setChecked(bool(default))
            elif kind == "choice":
                w = QW.QComboBox()
                w.addItems(o["items"])
                w.setCurrentIndex(int(default))
            elif kind == "optfloat":
                w = OptFloat(default, o.get("default", 1.0), o.get("auto", "automatisch"),
                             o.get("decimals", 4), o.get("min", 1e-9), o.get("max", 1e12))
            else:
                w = QW.QLineEdit(str(default or ""))
            self.w[key] = (kind, w)
            form.addRow(label, w)
        if presets:
            row = QW.QHBoxLayout()
            for name, vals in presets.items():
                b = QW.QPushButton(name)
                b.clicked.connect(lambda checked=False, v=vals: self.set_values(v))
                row.addWidget(b)
            lay.addLayout(row)
        bb = QW.QDialogButtonBox(QW.QDialogButtonBox.Ok | QW.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def set_values(self, vals):
        for k, v in vals.items():
            kind, w = self.w[k]
            if kind == "bool":
                w.setChecked(bool(v))
            elif kind in ("float", "int"):
                w.setValue(v)

    def values(self):
        out = {}
        for k, (kind, w) in self.w.items():
            out[k] = (w.value() if kind in ("float", "int", "optfloat") else
                      w.isChecked() if kind == "bool" else
                      w.currentIndex() if kind == "choice" else w.text())
        return out

    @staticmethod
    def ask(parent, title, specs, info=None, presets=None):
        dlg = FormDialog(parent, title, specs, info, presets)
        return dlg.values() if dlg.exec() else None


class MaterialDialog(QW.QDialog):
    """Materialdatenbank bearbeiten und auswaehlen (gespeichert in materialien.json)."""

    def __init__(self, parent, db, current=None):
        super().__init__(parent)
        self.db, self.chosen = db, None
        self.setWindowTitle("Materialdatenbank")
        lay = QW.QHBoxLayout(self)
        self.lst = QW.QListWidget()
        self.lst.setMinimumWidth(200)
        lay.addWidget(self.lst)
        right = QW.QVBoxLayout()
        lay.addLayout(right)
        form = QW.QFormLayout()
        right.addLayout(form)
        self.name = QW.QLineEdit()
        self.E = QW.QDoubleSpinBox()
        self.E.setRange(1, 1e8)
        self.E.setDecimals(1)
        self.nu = QW.QDoubleSpinBox()
        self.nu.setRange(0, 0.4999)
        self.nu.setDecimals(3)
        self.rp = OptFloat(None, 250.0, "nicht angegeben", 1)
        self.rho = OptFloat(None, 7.85, "nicht angegeben", 3)
        for lab, w in (("Name", self.name), ("E-Modul [MPa]", self.E), ("Querkontraktion ν", self.nu),
                       ("Streckgrenze/Festigkeit Rp [MPa]", self.rp), ("Dichte [g/cm³]", self.rho)):
            form.addRow(lab, w)
        self.msg = QW.QLabel("")
        right.addWidget(self.msg)
        row = QW.QHBoxLayout()
        right.addLayout(row)
        for txt, fn in (("Neu", self.new), ("Speichern", self.save), ("Löschen", self.delete),
                        ("Übernehmen", self.use), ("Schließen", self.reject)):
            b = QW.QPushButton(txt)
            b.clicked.connect(lambda checked=False, f=fn: f())
            row.addWidget(b)
        self.lst.currentRowChanged.connect(self.on_select)
        self.refresh(current)

    def refresh(self, sel=None):
        self.lst.blockSignals(True)
        self.lst.clear()
        self.lst.addItems(list(self.db))
        self.lst.blockSignals(False)
        if sel in self.db:
            self.lst.setCurrentRow(list(self.db).index(sel))
            self.fill(sel)

    def fill(self, name):
        v = self.db[name]
        self.name.setText(name)
        self.E.setValue(v["E"])
        self.nu.setValue(v["nu"])
        self.rp.setValue(v.get("rp"))
        self.rho.setValue(v.get("rho"))

    def on_select(self, row):
        if 0 <= row < len(self.db):
            self.fill(list(self.db)[row])
            self.msg.setText("")

    def new(self):
        self.name.clear()
        self.lst.clearSelection()
        self.msg.setText("Neues Material: Werte eintragen und speichern.")

    def save(self):
        name = self.name.text().strip()
        if not name:
            self.msg.setText("Bitte einen Namen eingeben.")
            return None
        self.db[name] = dict(E=self.E.value(), nu=self.nu.value(),
                             rp=self.rp.value(), rho=self.rho.value())
        core.save_materials(self.db)
        self.refresh(name)
        self.msg.setText(f"'{name}' gespeichert.")
        return name

    def delete(self):
        row = self.lst.currentRow()
        if row < 0:
            return
        name = list(self.db)[row]
        if QW.QMessageBox.question(self, "Löschen", f"'{name}' löschen?") == QW.QMessageBox.Yes:
            del self.db[name]
            core.save_materials(self.db)
            self.refresh()

    def use(self):
        name = self.save()
        if name:
            self.chosen = name
            self.accept()


# ==========================================================================
# Hauptfenster
# ==========================================================================
class MainWindow(QW.QMainWindow):
    CUTS = ["keine", "Clip-Ebene", "Schnitt-Ebene"]
    BAR_TITLES = ["von Mises [MPa]", "u [mm]", "Dichte"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FEM - interaktiv")
        self.resize(1400, 860)
        self.db = core.load_materials()
        self.setup = core.default_setup()
        self.model = None
        self.fem = None
        self.have_geometry = False
        self.mesh_info = dict(h=None, hmin=None)
        self.res = None
        self.sel = None
        self.mode = "orbit"
        self.only_visible = True
        self.angle = 30.0
        self.view = "setup"
        self.cut_i = self.field_i = 0
        self.thr = 0.5
        self.scale = 1.0
        self._press = None
        self._rb_origin = None
        self._marker_names = []
        self._grid4 = None
        self._vol = None

        if self.db:
            self.apply_material(next(iter(self.db)), quiet=True)

        self.frame = QW.QFrame()
        lay = QW.QVBoxLayout(self.frame)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self.frame)
        lay.addWidget(self.plotter.interactor)
        self.setCentralWidget(self.frame)
        self.plotter.interactor.installEventFilter(self)
        self._rb = QW.QRubberBand(QW.QRubberBand.Rectangle, self.plotter.interactor)

        self._build_docks()
        self._build_menus()
        core.set_logger(self.log)
        self.show_hint()

    # ------------------------------------------------------------------ UI
    def _act(self, menu, text, slot, shortcut=None, checkable=False, checked=False):
        a = QG.QAction(text, self)
        if shortcut:
            a.setShortcut(QG.QKeySequence(shortcut))
        a.setCheckable(checkable)
        a.setChecked(checked)
        a.triggered.connect(lambda checked=False, s=slot: s())
        menu.addAction(a)
        return a

    def _build_docks(self):
        w = QW.QWidget()
        lay = QW.QVBoxLayout(w)
        lay.addWidget(QW.QLabel("<b>Lager &amp; Lasten</b>"))
        self.bc_list = QW.QListWidget()
        self.bc_list.currentRowChanged.connect(self.highlight_bc)
        lay.addWidget(self.bc_list)
        b = QW.QPushButton("Auswahl entfernen")
        b.clicked.connect(lambda checked=False: self.remove_bc())
        lay.addWidget(b)
        lay.addWidget(QW.QLabel("<b>Lokale Verfeinerungen</b>"))
        self.ref_list = QW.QListWidget()
        lay.addWidget(self.ref_list)
        b = QW.QPushButton("Auswahl entfernen (Neu vernetzen nötig)")
        b.clicked.connect(lambda checked=False: self.remove_refinement())
        lay.addWidget(b)
        dock = QW.QDockWidget("Setup", self)
        dock.setWidget(w)
        self.addDockWidget(QC.Qt.LeftDockWidgetArea, dock)
        self.logw = QW.QPlainTextEdit()
        self.logw.setReadOnly(True)
        self.logw.setMaximumBlockCount(2000)
        dock2 = QW.QDockWidget("Protokoll", self)
        dock2.setWidget(self.logw)
        self.addDockWidget(QC.Qt.BottomDockWidgetArea, dock2)
        dock.setMinimumWidth(280)
        dock2.setMaximumHeight(160)

    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&Datei")
        self._act(m, "Bauteil öffnen …", self.open_geometry, "Ctrl+O")
        self._act(m, "Demo-Balken laden", self.open_demo)
        m.addSeparator()
        self._act(m, "Setup öffnen …", self.open_setup, "Ctrl+Shift+O")
        self._act(m, "Setup speichern (für Server) …", self.save_setup, "Ctrl+S")
        m.addSeparator()
        self._act(m, "Ergebnis als VTK exportieren …", self.export_result, "Ctrl+E")
        m.addSeparator()
        self._act(m, "Beenden", self.close, "Ctrl+Q")

        m = mb.addMenu("&Netz")
        self._act(m, "Netzparameter …", self.mesh_params)
        self._act(m, "Vernetzen", self.do_mesh, "Ctrl+M")
        self._act(m, "Lokal verfeinern (aus Auswahl) …", self.refine_from_selection, "Ctrl+R")
        self._act(m, "Alle Verfeinerungen entfernen", self.clear_refinements)

        m = mb.addMenu("&Auswahl")
        grp = QG.QActionGroup(self)
        self.mode_actions = {}
        for key, txt in (("orbit", "Nur drehen (keine Auswahl)"), ("node", "Knoten einzeln"),
                         ("face", "Fläche (zusammenhängend)"), ("box", "Rechteck aufziehen")):
            a = QG.QAction(txt, self)
            a.setCheckable(True)
            a.setChecked(key == "orbit")
            a.triggered.connect(lambda checked=False, k=key: self.set_mode(k))
            grp.addAction(a)
            m.addAction(a)
            self.mode_actions[key] = a
        m.addSeparator()
        self.vis_action = self._act(m, "Rechteck: nur sichtbare Seiten", lambda: None,
                                    checkable=True, checked=True)
        self.vis_action.toggled.connect(lambda on: setattr(self, "only_visible", on))
        self._act(m, "Winkel für Flächenwahl …", self.set_angle)
        self._act(m, "Auswahl löschen", self.clear_selection, "Esc")

        m = mb.addMenu("&Randbedingungen")
        self._act(m, "Lager aus Auswahl …", self.assign_support, "Ctrl+L")
        self._act(m, "Kraft aus Auswahl …", self.assign_force, "Ctrl+K")
        self._act(m, "Letzte entfernen", self.undo_bc)
        self._act(m, "Alle entfernen", self.clear_bc)

        m = mb.addMenu("&Material")
        self._act(m, "Materialdatenbank …", self.material_dialog, "Ctrl+T")

        m = mb.addMenu("&Berechnen")
        self._act(m, "Statische Berechnung", self.run_static, "F5")
        self._act(m, "Topologieoptimierung …", self.run_topopt, "F6")

        m = mb.addMenu("&Ansicht")
        self._act(m, "Zurück zur Setup-Ansicht", self.back_to_setup, "F2")
        self._act(m, "Schnittmodus wechseln", self.next_cut, "C")
        self._act(m, "Feld wechseln", self.next_field, "F")
        self._act(m, "Überhöhung ändern …", self.set_scale)

    def show_context_menu(self):
        m = QW.QMenu(self)
        sub = m.addMenu("Auswahlmodus")
        for a in self.mode_actions.values():
            sub.addAction(a)
        m.addAction("Auswahl löschen", self.clear_selection)
        m.addSeparator()
        m.addAction("Lager aus Auswahl …", self.assign_support)
        m.addAction("Kraft aus Auswahl …", self.assign_force)
        m.addAction("Netz lokal verfeinern (aus Auswahl) …", self.refine_from_selection)
        m.addSeparator()
        m.addAction("Netzparameter …", self.mesh_params)
        m.addAction("Vernetzen", self.do_mesh)
        m.addSeparator()
        m.addAction("Statische Berechnung", self.run_static)
        m.addAction("Topologieoptimierung …", self.run_topopt)
        if self.view == "result":
            m.addSeparator()
            m.addAction("Schnittmodus wechseln", self.next_cut)
            m.addAction("Feld wechseln", self.next_field)
            m.addAction("Zurück zur Setup-Ansicht", self.back_to_setup)
        m.exec(QG.QCursor.pos())

    # ------------------------------------------------------------ Hilfen
    def log(self, msg=""):
        msg = str(msg)
        self.logw.appendPlainText(msg)
        self.statusBar().showMessage(msg)
        print(msg)

    def error(self, msg):
        self.log("FEHLER: " + str(msg))
        QW.QMessageBox.critical(self, "Fehler", str(msg))

    def info(self, msg):
        self.log(msg)
        QW.QMessageBox.information(self, "Hinweis", str(msg))

    def show_hint(self):
        self.plotter.add_text("Datei > Bauteil oeffnen ...  (oder Demo-Balken laden)",
                              position="upper_left", font_size=12, name="hint")

    def need_model(self, selection=False):
        if self.model is None:
            self.info("Zuerst ein Bauteil öffnen und vernetzen (Rechtsklick > Vernetzen).")
            return False
        if selection and (self.sel is None or not self.sel.any()):
            self.info("Zuerst Knoten markieren (Menü Auswahl: Knoten einzeln / Fläche / Rechteck).")
            return False
        return True

    def sel_ids(self):
        return np.flatnonzero(self.sel)

    def mean_edge(self):
        P = self.model.nodes[self.model.tri]
        return float(np.mean(np.linalg.norm(P[:, 1] - P[:, 0], axis=1)))

    def volume_mm3(self):
        if self._vol is None:
            X = self.model.nodes[self.model.tets[:, :4]]
            d = np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), X[:, 3] - X[:, 0])
            self._vol = float(np.abs(d).sum() / 6.0)
        return self._vol

    # ------------------------------------------------------- Material
    def apply_material(self, name, quiet=False):
        v = self.db[name]
        self.setup["material"] = dict(name=name, E=v["E"], nu=v["nu"], rp=v.get("rp"), rho=v.get("rho"))
        if self.fem is not None:
            self.fem.set_material(v["E"], v["nu"])
        self.res = None
        if not quiet:
            self.log(f"Material: {name} (E = {v['E']:g} MPa, ν = {v['nu']:g})")
            if self.view == "result":
                self.back_to_setup()
            else:
                self.update_matinfo()

    def material_dialog(self):
        cur = self.setup["material"].get("name")
        dlg = MaterialDialog(self, self.db, cur)
        dlg.exec()
        if dlg.chosen:
            self.apply_material(dlg.chosen)
        elif cur in self.db:                    # evtl. geaenderte Werte des aktuellen Materials
            v = self.db[cur]
            m = self.setup["material"]
            if abs(v["E"] - m["E"]) > 1e-9 or abs(v["nu"] - m["nu"]) > 1e-12 \
                    or v.get("rp") != m.get("rp") or v.get("rho") != m.get("rho"):
                self.apply_material(cur)

    def update_matinfo(self):
        m = self.setup["material"]
        txt = f"Material: {m.get('name') or 'eigene Werte'}\nE = {m['E']:g} MPa, nu = {m['nu']:g}"
        if m.get("rp"):
            txt += f", Rp = {m['rp']:g} MPa"
        if self.model is not None and m.get("rho"):
            txt += f"\nMasse: {self.volume_mm3() * m['rho'] * 1e-3:.1f} g"
        self.plotter.add_text(txt, position="upper_right", font_size=8, name="matinfo")

    # --------------------------------------------------- Geometrie / Netz
    def reset_all(self):
        self.model = self.fem = self.res = self.sel = None
        self._grid4 = self._vol = None
        self.have_geometry = False
        self.view = "setup"
        keep = self.setup["material"]
        self.setup = core.default_setup()
        self.setup["material"] = keep
        self.mesh_info = dict(h=None, hmin=None)
        self.refresh_lists()
        self.plotter.clear_plane_widgets()
        self.plotter.clear()
        self.plotter.add_axes()

    def open_geometry(self, path=None):
        if not path:
            exts = " ".join("*" + e for e in core.ALL_EXT)
            path, _ = QW.QFileDialog.getOpenFileName(
                self, "Bauteil öffnen", "", f"3D-Geometrie ({exts});;Alle Dateien (*)")
        if not path:
            return
        ext = Path(path).suffix.lower()
        unit = "mm"
        if ext in core.SURF_EXT or ext in (".msh", ".brep"):
            unit, ok = QW.QInputDialog.getItem(self, "Einheit", "Längeneinheit der Datei:",
                                               list(core.UNITS), 0, False)
            if not ok:
                return
        self.reset_all()
        self.setup["geometry"] = dict(file=str(Path(path).resolve()), unit=unit)
        self.show_preview()
        if self.have_geometry and ext == ".msh":
            self.do_mesh()

    def open_demo(self):
        self.reset_all()
        self.setup["geometry"] = dict(file=None, unit="mm")
        self.show_preview()

    def show_preview(self):
        g = self.setup["geometry"]
        try:
            with Busy():
                nodes, tris = core.preview_surface(g["file"], core.UNITS[g["unit"]])
        except FemError as e:
            return self.error(e)
        except Exception as e:
            return self.error(f"Geometrie konnte nicht geladen werden: {e}")
        self.have_geometry = True
        faces = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
        self.plotter.clear()
        self.plotter.add_mesh(pv.PolyData(nodes, faces), color="lightsteelblue",
                              show_edges=False, name="body")
        self.plotter.add_axes()
        self.plotter.reset_camera()
        size = np.ptp(nodes, axis=0)
        self.log(f"Geometrie geladen: {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm. "
                 "Rechtsklick > Netzparameter / Vernetzen.")
        self.update_matinfo()

    def mesh_params(self):
        m = self.setup["mesh"]
        vals = FormDialog.ask(
            self, "Netzparameter",
            [("h", "Max. Elementgröße [mm]", "optfloat", m.get("h"),
              dict(default=self.mesh_info.get("h") or 5.0)),
             ("hmin", "Min. Elementgröße [mm]", "optfloat", m.get("hmin"),
              dict(default=self.mesh_info.get("hmin") or 1.0, auto="automatisch (h/4)")),
             ("curv", "Krümmungsverfeinerung (Elemente je 360°, 0 = aus)", "float",
              m.get("curv", 12.0), dict(min=0, max=100, decimals=1)),
             ("order", "Elementtyp", "choice", 0 if m.get("order", 2) == 2 else 1,
              dict(items=["Tet10 (quadratisch, genau)", "Tet4 (linear, schnell)"])),
             ("remesh", "Danach sofort neu vernetzen", "bool", self.have_geometry)],
            info="Die Parameter gelten für das nächste Vernetzen. Lokale Verfeinerungen "
                 "werden über Auswahl > Rechtsklick > Netz lokal verfeinern angelegt.")
        if vals is None:
            return
        if vals["h"] and vals["hmin"] and vals["hmin"] > vals["h"]:
            return self.error("Die minimale Elementgröße darf nicht größer als die maximale sein.")
        m.update(h=vals["h"], hmin=vals["hmin"], curv=vals["curv"],
                 order=2 if vals["order"] == 0 else 1)
        if vals["remesh"] and self.have_geometry:
            self.do_mesh()

    def do_mesh(self):
        if not self.have_geometry:
            return self.info("Erst ein Bauteil öffnen (Datei > Bauteil öffnen) oder den Demo-Balken laden.")
        g, m = self.setup["geometry"], self.setup["mesh"]
        old = self.model
        try:
            with Busy():
                self.log("Vernetze ... (kann etwas dauern)")
                nodes, tets, info = core.mesh_body(
                    g["file"], m.get("h"), m.get("order", 2), core.UNITS[g["unit"]],
                    m.get("hmin"), m.get("curv", 12.0), m.get("refinements") or None)
                new = core.MeshModel(nodes, tets)
                dropped = 0
                if old is not None:                      # Lager/Lasten auf das neue Netz uebertragen
                    for key in ("supports", "loads"):
                        keep = []
                        for it in self.setup[key]:
                            ids = core.transfer_ids(old, new, it["ids"]) if it.get("ids") is not None else []
                            if len(ids):
                                it["ids"] = ids
                                keep.append(it)
                            else:
                                dropped += 1
                        self.setup[key] = keep
        except FemError as e:
            return self.error(e)
        except Exception as e:
            return self.error(f"Vernetzung fehlgeschlagen: {e}")
        self.model, self.fem, self.res, self._grid4, self._vol = new, None, None, None, None
        self.mesh_info = info
        self.sel = np.zeros(new.nn, bool)
        self.show_body(reset=old is None)
        self.refresh_lists()
        msg = core.mesh_stats(nodes, tets)
        if dropped:
            msg += f" | {dropped} Zuweisung(en) nicht übertragbar"
        if len(tets) > 60000:
            msg += " | WARNUNG: sehr großes Netz"
        self.log(msg)

    # --------------------------------------------------------- Anzeige
    def show_body(self, reset=False):
        m = self.model
        self.plotter.clear_plane_widgets()
        self.plotter.clear()
        faces = np.hstack([np.full((len(m.tri), 1), 3), m.tri]).ravel()
        self.plotter.add_mesh(pv.PolyData(m.nodes, faces), color="lightsteelblue", show_edges=True,
                              edge_color="gray", line_width=0.5, name="body", reset_camera=reset)
        self.plotter.add_axes()
        if reset:
            self.plotter.reset_camera()
        self.view = "setup"
        self.redraw_markers()
        self.update_sel_actor()
        self.update_matinfo()

    def redraw_markers(self):
        for n in self._marker_names:
            self.plotter.remove_actor(n)
        self._marker_names = []
        if self.model is None or self.view != "setup":
            return
        m = self.model
        diag = m.diag
        for i, s in enumerate(self.setup["supports"]):
            col = ("dodgerblue" if np.any(s["values"] != 0) else "red" if s["mask"].all() else "orange")
            self.plotter.add_points(m.nodes[s["ids"]], color=col, point_size=9,
                                    render_points_as_spheres=True, name=f"sup{i}", reset_camera=False)
            self._marker_names.append(f"sup{i}")
        for i, ld in enumerate(self.setup["loads"]):
            f = ld["force"]
            if not np.any(f):
                continue
            d = f / np.linalg.norm(f)
            c = m.nodes[ld["ids"]].mean(0)
            self.plotter.add_arrows(c - d * 0.15 * diag, d * 0.15 * diag, color="green",
                                    name=f"load{i}", mag=1.0)
            self._marker_names.append(f"load{i}")
        for i, r in enumerate(self.setup["mesh"]["refinements"]):
            lo, hi = r["lo"], r["hi"]
            self.plotter.add_mesh(pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])),
                                  style="wireframe", color="magenta", line_width=1.5,
                                  name=f"ref{i}", reset_camera=False)
            self._marker_names.append(f"ref{i}")
        self.plotter.render()

    def update_sel_actor(self):
        if self.model is None or self.sel is None:
            return
        ids = self.sel_ids()
        if len(ids) == 0:
            self.plotter.remove_actor("sel")
        else:
            few = len(ids) < 5000
            self.plotter.add_points(self.model.nodes[ids], color="yellow", point_size=10 if few else 4,
                                    render_points_as_spheres=few, name="sel", reset_camera=False)
        self.plotter.render()
        self.statusBar().showMessage(f"{len(ids)} Knoten markiert")

    def refresh_lists(self):
        self.bc_list.clear()
        for s in self.setup["supports"]:
            self.bc_list.addItem(f"Lager {''.join(c for c, b in zip('XYZ', s['mask']) if b)} "
                                 f"({len(s['ids'])} Knoten)")
        for ld in self.setup["loads"]:
            f = ld["force"]
            self.bc_list.addItem(f"Kraft ({f[0]:g}, {f[1]:g}, {f[2]:g}) N ({len(ld['ids'])} Knoten)")
        self.ref_list.clear()
        for r in self.setup["mesh"]["refinements"]:
            self.ref_list.addItem(f"h = {r['size']:g} mm, Übergang {r['trans']:g} mm, "
                                  f"x[{r['lo'][0]:.1f}..{r['hi'][0]:.1f}]")

    def highlight_bc(self, row):
        if self.model is None or self.view != "setup":
            return
        ns = len(self.setup["supports"])
        items = self.setup["supports"] + self.setup["loads"]
        if 0 <= row < len(items):
            self.plotter.add_points(self.model.nodes[items[row]["ids"]], color="cyan", point_size=13,
                                    render_points_as_spheres=True, name="hl", reset_camera=False)
        else:
            self.plotter.remove_actor("hl")

    # ------------------------------------------------- Maus / Auswahl
    def set_mode(self, mode):
        self.mode = mode
        self.mode_actions[mode].setChecked(True)
        self.log({"orbit": "Modus: nur drehen.",
                  "node": "Modus: Knoten einzeln (Klick = setzen/entfernen, Ctrl+Klick = entfernen).",
                  "face": "Modus: Fläche (Klick wählt die zusammenhängende Fläche).",
                  "box": "Modus: Rechteck (aufziehen; Ctrl = abwählen). Zum Drehen anderen Modus wählen."}[mode])

    def set_angle(self):
        v, ok = QW.QInputDialog.getDouble(self, "Flächenwahl", "Max. Knickwinkel zwischen Dreiecken [°]:",
                                          self.angle, 1.0, 179.0, 1)
        if ok:
            self.angle = v

    def clear_selection(self):
        if self.sel is not None:
            self.sel[:] = False
            self.update_sel_actor()

    def eventFilter(self, obj, ev):
        if obj is not self.plotter.interactor:
            return False
        E = QC.QEvent
        t = ev.type()
        try:
            if t == E.MouseButtonPress:
                self._press = (ev.button(), QC.QPoint(ev.pos()))
                if (ev.button() == QC.Qt.LeftButton and self.mode == "box" and self.model is not None
                        and self.view == "setup"):
                    self._rb_origin = QC.QPoint(ev.pos())
                    self._rb.setGeometry(QC.QRect(self._rb_origin, QC.QSize()))
                    self._rb.show()
                    return True
            elif t == E.MouseMove and self._rb_origin is not None:
                self._rb.setGeometry(QC.QRect(self._rb_origin, ev.pos()).normalized())
                return True
            elif t == E.MouseButtonRelease:
                b, p = ev.button(), self._press
                if self._rb_origin is not None and b == QC.Qt.LeftButton:
                    rect = QC.QRect(self._rb_origin, ev.pos()).normalized()
                    self._rb.hide()
                    self._rb_origin = None
                    self.box_select(rect, ev.modifiers())
                    return True
                if p and p[0] == b and (ev.pos() - p[1]).manhattanLength() < 5:
                    if b == QC.Qt.LeftButton and self.mode in ("node", "face") \
                            and self.model is not None and self.view == "setup":
                        self.click_select(QC.QPoint(ev.pos()), ev.modifiers())
                    elif b == QC.Qt.RightButton:
                        QC.QTimer.singleShot(0, self.show_context_menu)
                self._press = None
        except Exception as e:                     # GUI nie durch Auswahlfehler abstuerzen lassen
            self.log(f"Auswahlfehler: {e}")
        return False

    def project(self, pts):
        """Weltpunkte -> (u, v, depth, w): u,v in [0,1] (v nach oben), w>0 = vor der Kamera."""
        ren = self.plotter.renderer
        cam = ren.GetActiveCamera()
        W = max(self.plotter.interactor.width(), 1)
        H = max(self.plotter.interactor.height(), 1)
        mat = cam.GetCompositeProjectionTransformMatrix(W / H, -1, 1)
        M = np.array([[mat.GetElement(i, j) for j in range(4)] for i in range(4)])
        P = np.c_[pts, np.ones(len(pts))] @ M.T
        w = P[:, 3]
        ws = np.where(np.abs(w) < 1e-12, 1e-12, w)
        ndc = P[:, :3] / ws[:, None]
        return (ndc[:, 0] + 1) / 2, (ndc[:, 1] + 1) / 2, ndc[:, 2], w

    def camera_vectors(self):
        cam = self.plotter.renderer.GetActiveCamera()
        return (np.array(cam.GetPosition()), np.array(cam.GetDirectionOfProjection()),
                bool(cam.GetParallelProjection()))

    def facing(self, ids):
        pos, d, parallel = self.camera_vectors()
        nrm = self.model.node_normal[ids]
        v = -d[None, :] if parallel else pos[None, :] - self.model.nodes[ids]
        return np.einsum("ij,ij->i", nrm, np.broadcast_to(v, nrm.shape)) > 0

    def pick_node(self, pos, radius=14):
        m = self.model
        ids = m.surf_nodes
        P = m.nodes[ids]
        u, v, z, w = self.project(P)
        W, H = self.plotter.interactor.width(), self.plotter.interactor.height()
        d = np.hypot(u * W - pos.x(), (1 - v) * H - pos.y())
        ok = (w > 0) & (d < radius) & self.facing(ids)
        if not ok.any():
            return None
        cand = np.flatnonzero(ok)
        campos, vdir, _ = self.camera_vectors()
        depth = (P[cand] - campos) @ vdir
        cand = cand[depth <= depth.min() + 0.03 * m.diag]
        return int(ids[cand[np.argmin(d[cand])]])

    def click_select(self, pos, mod):
        m = self.model
        remove = bool(mod & QC.Qt.ControlModifier)
        if self.mode == "node":
            n = self.pick_node(pos)
            if n is None:
                return
            self.sel[n] = False if remove else not self.sel[n]
        else:
            n = self.pick_node(pos, 22)
            if n is None:
                return
            tris = m.tris_of_node(n)
            if len(tris) == 0:
                return
            u, v, _, _ = self.project(m.nodes[m.tri[tris]].mean(1))
            W, H = self.plotter.interactor.width(), self.plotter.interactor.height()
            best = tris[np.argmin(np.hypot(u * W - pos.x(), (1 - v) * H - pos.y()))]
            lab = m.face_labels(self.angle)
            self.sel[m.nodes_of_tris(lab == lab[best])] = not remove
        self.update_sel_actor()

    def box_select(self, rect, mod):
        if self.model is None or self.view != "setup":
            return
        m = self.model
        W, H = self.plotter.interactor.width(), self.plotter.interactor.height()
        u0, u1 = rect.left() / W, rect.right() / W
        v0, v1 = 1 - rect.bottom() / H, 1 - rect.top() / H
        ids = m.surf_nodes
        u, v, _, w = self.project(m.nodes[ids])
        inside = (w > 0) & (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
        if self.only_visible:
            inside &= self.facing(ids)
        self.sel[ids[inside]] = not bool(mod & QC.Qt.ControlModifier)
        self.update_sel_actor()

    # ------------------------------------------- Randbedingungen
    def _after_bc_change(self):
        self.res = None
        self.sel[:] = False
        self.update_sel_actor()
        self.redraw_markers()
        self.refresh_lists()

    def assign_support(self):
        if not self.need_model(selection=True):
            return
        full = dict(lx=True, ly=True, lz=True)
        vals = FormDialog.ask(
            self, "Lager definieren",
            [("lx", "X gesperrt", "bool", True), ("ly", "Y gesperrt", "bool", True),
             ("lz", "Z gesperrt", "bool", True),
             ("dx", "Verschiebung X [mm]", "float", 0.0, dict(decimals=4)),
             ("dy", "Verschiebung Y [mm]", "float", 0.0, dict(decimals=4)),
             ("dz", "Verschiebung Z [mm]", "float", 0.0, dict(decimals=4))],
            info=f"Lager auf {int(self.sel.sum())} markierten Knoten (globales Koordinatensystem). "
                 "Verschiebungen wirken nur auf gesperrte Richtungen.",
            presets={"Fest (XYZ)": full, "nur X": dict(lx=True, ly=False, lz=False),
                     "nur Y": dict(lx=False, ly=True, lz=False), "nur Z": dict(lx=False, ly=False, lz=True),
                     "X+Y": dict(lx=True, ly=True, lz=False), "X+Z": dict(lx=True, ly=False, lz=True),
                     "Y+Z": dict(lx=False, ly=True, lz=True)})
        if vals is None:
            return
        mask = np.array([vals["lx"], vals["ly"], vals["lz"]], bool)
        if not mask.any():
            return self.error("Mindestens eine Richtung muss gesperrt sein.")
        d = np.array([vals["dx"], vals["dy"], vals["dz"]]) * mask
        self.setup["supports"].append(dict(ids=self.sel_ids(), mask=mask, values=d))
        self._after_bc_change()

    def assign_force(self):
        if not self.need_model(selection=True):
            return
        vals = FormDialog.ask(
            self, "Kraft vorgeben",
            [("fx", "Fx [N]", "float", 0.0, dict(decimals=3)),
             ("fy", "Fy [N]", "float", 0.0, dict(decimals=3)),
             ("fz", "Fz [N]", "float", -1000.0, dict(decimals=3))],
            info=f"Gesamtkraft, flächengewichtet auf {int(self.sel.sum())} Knoten verteilt.")
        if vals is None:
            return
        f = np.array([vals["fx"], vals["fy"], vals["fz"]])
        if not f.any():
            return self.error("Die Kraft ist null.")
        self.setup["loads"].append(dict(ids=self.sel_ids(), force=f))
        self._after_bc_change()

    def _bc_items(self):
        return [("supports", i) for i in range(len(self.setup["supports"]))] + \
               [("loads", i) for i in range(len(self.setup["loads"]))]

    def undo_bc(self):
        if self.setup["loads"]:
            self.setup["loads"].pop()
        elif self.setup["supports"]:
            self.setup["supports"].pop()
        else:
            return
        self._after_bc_change()

    def remove_bc(self):
        row = self.bc_list.currentRow()
        items = self._bc_items()
        if 0 <= row < len(items):
            key, i = items[row]
            del self.setup[key][i]
            self.plotter.remove_actor("hl")
            self._after_bc_change()

    def clear_bc(self):
        if self.model is None:
            return
        self.setup["supports"].clear()
        self.setup["loads"].clear()
        self._after_bc_change()

    # ------------------------------------------------- Verfeinerung
    def refine_from_selection(self):
        if not self.need_model(selection=True):
            return
        ids = self.sel_ids()
        pts = self.model.nodes[ids]
        cur = self.mesh_info.get("h") or self.mean_edge()
        vals = FormDialog.ask(
            self, "Lokale Verfeinerung",
            [("size", "Elementgröße im Bereich [mm]", "float", round(cur / 3, 3),
              dict(min=1e-4, max=1e6, decimals=4)),
             ("trans", "Übergangsbereich [mm]", "optfloat", None, dict(default=cur, auto="3 × Elementgröße")),
             ("remesh", "Sofort neu vernetzen", "bool", True)],
            info=f"Verfeinert wird eine Box um die {len(ids)} markierten Knoten (plus 1,5 × Elementgröße "
                 f"Rand). Aktuelle Elementgröße ca. {cur:.3g} mm.")
        if vals is None:
            return
        ref = core.make_refinement((pts.min(0), pts.max(0)), vals["size"], vals["trans"])
        self.setup["mesh"]["refinements"].append(ref)
        self.sel[:] = False
        self.update_sel_actor()
        self.refresh_lists()
        if vals["remesh"]:
            self.do_mesh()
        else:
            self.redraw_markers()
            self.log("Verfeinerung vorgemerkt - Netz > Vernetzen ausführen.")

    def remove_refinement(self):
        row = self.ref_list.currentRow()
        refs = self.setup["mesh"]["refinements"]
        if 0 <= row < len(refs):
            del refs[row]
            self.refresh_lists()
            self.redraw_markers()
            self.log("Verfeinerung entfernt - Netz > Vernetzen ausführen.")

    def clear_refinements(self):
        self.setup["mesh"]["refinements"].clear()
        self.refresh_lists()
        self.redraw_markers()
        self.log("Alle Verfeinerungen entfernt - Netz > Vernetzen ausführen.")

    # --------------------------------------------------- Berechnung
    def get_fem(self):
        mat = self.setup["material"]
        if self.fem is None:
            with Busy():
                self.log("Baue Steifigkeitsmatrizen auf ...")
                self.fem = core.FEM(self.model.nodes, self.model.tets, mat["E"], mat["nu"])
        elif abs(self.fem.E - mat["E"]) > 1e-9 or abs(self.fem.nu - mat["nu"]) > 1e-12:
            self.fem.set_material(mat["E"], mat["nu"])
        return self.fem

    def run_static(self):
        self._run("static")

    def run_topopt(self):
        if not self.need_model():
            return
        a = self.setup["analysis"]["topopt"]
        rm = core.default_rmin(self.model)
        vals = FormDialog.ask(
            self, "Topologieoptimierung",
            [("volfrac", "Volumenanteil", "float", a["volfrac"], dict(min=0.05, max=0.95, decimals=2)),
             ("rmin", "Filterradius [mm]", "optfloat", a.get("rmin"),
              dict(default=rm, auto=f"automatisch ({rm:.3g} mm)")),
             ("penal", "Penalisierung p", "float", a["penal"], dict(min=1, max=6, decimals=1)),
             ("maxit", "Max. Iterationen", "int", a["maxit"], dict(min=1, max=1000))],
            info="Minimiert die Nachgiebigkeit bei vorgegebenem Volumen (SIMP). Elemente an Lagern und "
                 "Lasten bleiben erhalten. Abbruch über den Dialog möglich.")
        if vals is None:
            return
        a.update(volfrac=vals["volfrac"], rmin=vals["rmin"], penal=vals["penal"], maxit=vals["maxit"])
        self._run("topopt")

    def grid4(self):
        if self._grid4 is None:
            t4 = self.model.tets[:, :4]
            cells = np.hstack([np.full((len(t4), 1), 4), t4]).ravel()
            self._grid4 = pv.UnstructuredGrid(cells, np.full(len(t4), 10, np.uint8), self.model.nodes)
        return self._grid4

    def _run(self, kind):
        if not self.need_model():
            return
        if self.view == "result":
            self.show_body()
        self.setup["analysis"]["type"] = kind
        prog, cb = None, None
        if kind == "topopt":
            maxit = int(self.setup["analysis"]["topopt"]["maxit"])
            prog = QW.QProgressDialog("Topologieoptimierung läuft ...", "Abbrechen", 0, maxit, self)
            prog.setWindowModality(QC.Qt.WindowModal)
            prog.setMinimumDuration(0)
            prog.setValue(0)
            g0 = self.grid4().copy()
            self.plotter.remove_actor("body")

            def cb(it, comp, vfrac, change, rt):
                if it == 1 or it % 3 == 0:
                    g0.cell_data["Dichte"] = rt
                    shown = g0.threshold(0.5, scalars="Dichte")
                    if shown.n_cells:
                        self.plotter.add_mesh(shown.extract_surface(), scalars="Dichte", cmap="gray_r",
                                              clim=(0, 1), show_scalar_bar=False, name="to",
                                              reset_camera=False)
                self.log(f"TO Iteration {it}/{maxit}: Nachgiebigkeit {comp:.4g}, Volumen {vfrac:.3f}, "
                         f"Änderung {change:.3f}")
                prog.setValue(it)
                QW.QApplication.processEvents()
                return prog.wasCanceled()
        try:
            fem = self.get_fem()
            with Busy():
                res = core.run_analysis(self.setup, self.model, fem, cb)
        except FemError as e:
            if prog:
                prog.close()
            self.plotter.remove_actor("to")
            self.show_body()
            return self.error(e)
        except Exception as e:
            if prog:
                prog.close()
            self.plotter.remove_actor("to")
            self.show_body()
            return self.error(f"Berechnung fehlgeschlagen: {e}")
        if prog:
            prog.close()
        self.res = res
        for line in core.summarize(self.setup, self.model, fem, res):
            self.log(line)
        self.enter_result("Dichte" if res["rho"] is not None else None)

    # ---------------------------------------------------- Ergebnisse
    def enter_result(self, prefer=None):
        r, m = self.res, self.model
        g = self.grid4().copy()
        umag = np.linalg.norm(r["u"], axis=1)
        g.point_data["Verschiebung"] = r["u"]
        g.point_data["u [mm]"] = umag
        g.point_data["von Mises [MPa]"] = r["vm"]
        if r["rho"] is not None:
            g.cell_data["Dichte"] = r["rho"]
        um = umag[m.corner_ids].max()
        self.scale = 0.08 * m.diag / um if um > 0 else 1.0
        self.rmesh_src = g
        self.fields = ["von Mises [MPa]", "u [mm]"] + (["Dichte"] if r["rho"] is not None else [])
        self.field_i = self.fields.index(prefer) if prefer in self.fields else 0
        self.cut_i = 0
        self.view = "result"
        self.build_rmesh()
        self.draw_result()

    def build_rmesh(self):
        self.rmesh = self.rmesh_src.warp_by_vector("Verschiebung", factor=self.scale)

    def set_scale(self):
        if self.view != "result":
            return self.info("Nur in der Ergebnisansicht.")
        v, ok = QW.QInputDialog.getDouble(self, "Überhöhung", "Verformung darstellen mit Faktor:",
                                          self.scale, 0.0, 1e9, 2)
        if ok:
            self.scale = v
            self.build_rmesh()
            self.draw_result()

    def draw_result(self):
        pl = self.plotter
        pl.clear_plane_widgets()
        try:
            pl.clear_slider_widgets()
        except Exception:
            pass
        pl.clear()
        pl.add_axes()
        mesh = self.rmesh
        m = self.model
        field, cut = self.fields[self.field_i], self.CUTS[self.cut_i]
        if field == "Dichte":
            clim, cmap = (0.0, 1.0), "gray_r"
            base = mesh.threshold(self.thr, scalars="Dichte")
        else:
            arr = mesh.point_data[field][m.corner_ids]
            clim, cmap = (float(arr.min()), float(arr.max())), "turbo"
            base = mesh
        if base.n_cells == 0:
            return self.log("Leere Auswahl (Dichte-Schwelle zu hoch).")
        kw = dict(scalars=field, cmap=cmap, clim=clim, name="result", scalar_bar_args=dict(title=field))
        if cut == "keine":
            pl.add_mesh(base, **kw)
            if field == "Dichte":
                def thr_cb(v):
                    self.thr = float(v)
                    sh = mesh.threshold(self.thr, scalars="Dichte")
                    if sh.n_cells:
                        pl.add_mesh(sh, **kw)
                pl.add_slider_widget(thr_cb, [0.05, 0.95], value=self.thr, title="Dichte-Schwelle")
        elif cut == "Clip-Ebene":
            pl.add_mesh_clip_plane(base, normal="x", **kw)
        else:
            pl.add_mesh_slice(base, normal="x", **kw)
            pl.add_mesh(mesh.extract_surface(), style="wireframe", color="gray", opacity=0.15,
                        name="ghost")
        pl.render()
        self.log(f"{field} | Schnitt: {cut} | {self.scale:.1f}-fach überhöht | "
                 "C = Schnittmodus, F = Feld, F2 = zurück")

    def next_cut(self):
        if self.view == "result":
            self.cut_i = (self.cut_i + 1) % len(self.CUTS)
            self.draw_result()

    def next_field(self):
        if self.view == "result":
            self.field_i = (self.field_i + 1) % len(self.fields)
            self.draw_result()

    def back_to_setup(self):
        if self.model is not None and self.view == "result":
            self.plotter.clear_plane_widgets()
            self.show_body()

    def export_result(self):
        if self.res is None:
            return self.info("Noch kein Ergebnis vorhanden (Berechnen > Statische Berechnung).")
        path, _ = QW.QFileDialog.getSaveFileName(self, "Ergebnis exportieren", "ergebnis.vtk",
                                                 "VTK Legacy (*.vtk);;VTK XML (*.vtu)")
        if path:
            try:
                out = core.write_results(path, self.model, self.res)
                self.log(f"Ergebnis geschrieben: {out}")
            except Exception as e:
                self.error(e)

    # ------------------------------------------------- Setup-Datei
    def save_setup(self):
        a = self.setup["analysis"]
        t = a["topopt"]
        rm = core.default_rmin(self.model) if self.model is not None else 1.0
        vals = FormDialog.ask(
            self, "Setup für Server speichern",
            [("type", "Analyse", "choice", 0 if a["type"] == "static" else 1,
              dict(items=["Statisch", "Topologieoptimierung"])),
             ("volfrac", "TO: Volumenanteil", "float", t["volfrac"], dict(min=0.05, max=0.95, decimals=2)),
             ("rmin", "TO: Filterradius [mm]", "optfloat", t.get("rmin"), dict(default=rm)),
             ("penal", "TO: Penalisierung p", "float", t["penal"], dict(min=1, max=6, decimals=1)),
             ("maxit", "TO: Max. Iterationen", "int", t["maxit"], dict(min=1, max=1000)),
             ("out", "Ergebnisdatei", "text", self.setup["output"].get("file") or "ergebnis.vtk"),
             ("mesh", "Netz mitspeichern (empfohlen)", "bool", self.model is not None)],
            info="Speichert Geometrie-Verweis, Netzparameter, Material, Lager, Lasten und Analyse als "
                 "JSON. Mit mitgespeichertem Netz rechnet der Server exakt auf diesem Netz.")
        if vals is None:
            return
        path, _ = QW.QFileDialog.getSaveFileName(self, "Setup speichern", "setup.json", "Setup (*.json)")
        if not path:
            return
        a.update(type="static" if vals["type"] == 0 else "topopt")
        t.update(volfrac=vals["volfrac"], rmin=vals["rmin"], penal=vals["penal"], maxit=vals["maxit"])
        self.setup["output"]["file"] = vals["out"].strip() or "ergebnis.vtk"
        try:
            mesh_file = core.save_setup(self.setup, path, self.model, vals["mesh"] and self.model is not None)
        except Exception as e:
            return self.error(f"Setup konnte nicht gespeichert werden: {e}")
        p = Path(path)
        self.info(f"Setup gespeichert: {p}" + (f"\nNetz: {mesh_file}" if mesh_file else "") +
                  f"\n\nAuf dem Server (ohne Grafik):\n  python fem_core.py run {p.name}\n"
                  f"Ergebnis: {self.setup['output']['file']}")

    def open_setup(self, path=None):
        if not path:
            path, _ = QW.QFileDialog.getOpenFileName(self, "Setup öffnen", "", "Setup (*.json)")
        if not path:
            return
        try:
            with Busy():
                setup, base = core.load_setup(path)
                g = setup["geometry"]
                if g.get("file"):
                    g["file"] = str(core.resolve_path(base, g["file"]))
                self.reset_all()
                self.setup = setup
                self.have_geometry = g.get("file") is None or Path(g["file"]).exists()
                model, from_file = core.build_model(setup, base)
                for key in ("supports", "loads"):
                    for it in setup[key]:
                        it["ids"] = core.resolve_ids(model, it)
        except FemError as e:
            return self.error(e)
        except Exception as e:
            return self.error(f"Setup konnte nicht geladen werden: {e}")
        self.model, self.sel = model, np.zeros(model.nn, bool)
        self.mesh_info = dict(h=setup["mesh"].get("h"), hmin=setup["mesh"].get("hmin"))
        self.show_body(reset=True)
        self.refresh_lists()
        self.log(f"Setup geladen: {Path(path).name}")

    def closeEvent(self, ev):
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(ev)


def main():
    app = QW.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        p = sys.argv[1]
        if p.lower().endswith(".json"):
            w.open_setup(p)
        else:
            w.open_geometry(p)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()