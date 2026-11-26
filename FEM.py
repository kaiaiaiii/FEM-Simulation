#!/usr/bin/env python3
"""
Interaktive lineare FEM mit Topologieoptimierung (v2)
=====================================================

Neu gegenueber v1
-----------------
* Tet10-Elemente (quadratische Tetraeder, Standard) oder Tet4 (--order 1)
* Dialogfenster zur freien Definition des Lagers (Richtungen einzeln sperren,
  optional vorgegebene Verschiebung)
* Ergebnisansicht mit Clip-/Schnittebene (verschiebbares Widget),
  umschaltbare Felder (von Mises, Verschiebung, Dichte)
* Topologieoptimierung (SIMP, Compliance-Minimierung, Dichtefilter, OC-Update)

Installation
------------
    pip install numpy scipy pyvista gmsh
    pip install pyamg        # optional, beschleunigt grosse Modelle

Aufruf
------
    python fem_interaktiv_v2.py                       # Startdialog: Bauteil,
                                                      # Material, Einheit, Netz
    python fem_interaktiv_v2.py bauteil.step --h 3    # direkt per Kommandozeile
    python fem_interaktiv_v2.py teil.stl --einheit m --material Titan
    python fem_interaktiv_v2.py --konsole             # Eingaben in der Konsole
                                                      # statt tkinter-Dialogen
Unterstuetzte Dateien: STEP/STP, IGES/IGS, BREP, STL, OBJ, PLY, VTK/VTP, MSH.
Einheiten: mm, N, MPa (Dateien in cm, m oder inch werden beim Laden in mm
umgerechnet; STEP/IGES bringen ihre Einheit selbst mit).

Bedienung im 3D-Fenster
-----------------------
  R + Rechteck ziehen : sichtbare Oberflaeche markieren (gelb)
  1  Lager definieren (Dialog) | 2  Kraft (Dialog) | 3  Undo | 4  Alles loeschen
  5  Berechnen (statisch)
  6  Topologieoptimierung (Dialog); waehrend des Laufs: 6 = Abbruch
  7  Schnittmodus wechseln: keine -> Clip-Ebene -> Schnitt-Ebene
  8  Feld wechseln: von Mises -> Verschiebung -> Dichte (nach Optimierung)
  9  Export (ergebnis.vtu, bei Optimierung zusaetzlich optimiert.stl)
  N  Netz anpassen: globale Groessen, Kruemmungsverfeinerung, lokale Verfeinerung
     um die aktuelle Markierung (R+Rechteck, dann N), Neuvernetzung; Lager/Lasten
     werden auf das neue Netz uebertragen
  M  Materialdatenbank: Materialien anlegen/aendern/loeschen und auswaehlen
     (gespeichert in materialien.json neben dem Skript)
  0  zurueck zur Lager-/Lastansicht
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import pyvista as pv

USE_TK = True
CAD_EXT = (".step", ".stp", ".iges", ".igs")
SURF_EXT = (".stl", ".obj", ".ply", ".vtk", ".vtp")
ALL_EXT = CAD_EXT + SURF_EXT + (".brep", ".msh")
# Standard-Materialien (Richtwerte - bitte fuer die eigene Anwendung pruefen):
# E [MPa], nu [-], rp = Streck-/Dehngrenze bzw. Festigkeit [MPa], rho [g/cm^3]
DEFAULT_MATERIALS = {
    "Baustahl S235":        dict(E=210000.0, nu=0.30, rp=235.0, rho=7.85),
    "Edelstahl 1.4301":     dict(E=200000.0, nu=0.30, rp=190.0, rho=7.90),
    "Aluminium EN AW-6061": dict(E=70000.0, nu=0.33, rp=240.0, rho=2.70),
    "AlSi10Mg (3D-Druck)":  dict(E=70000.0, nu=0.33, rp=230.0, rho=2.67),
    "Titan Ti-6Al-4V":      dict(E=110000.0, nu=0.34, rp=880.0, rho=4.43),
    "Inconel 718":          dict(E=200000.0, nu=0.29, rp=1030.0, rho=8.19),
    "Gusseisen GJL-250":    dict(E=110000.0, nu=0.26, rp=250.0, rho=7.15),
    "Kupfer":               dict(E=120000.0, nu=0.34, rp=200.0, rho=8.96),
}
CUSTOM = "(Eigene Werte)"


def db_path():
    """Materialdatenbank: materialien.json neben dem Skript, sonst im Home-Verzeichnis."""
    p = Path(__file__).resolve().with_name("materialien.json")
    if p.exists() or os.access(p.parent, os.W_OK):
        return p
    return Path.home() / ".fem_materialien.json"


def save_materials(db):
    try:
        with open(db_path(), "w", encoding="utf-8") as fh:
            json.dump(db, fh, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"WARNUNG: Materialdatenbank konnte nicht gespeichert werden: {e}")


def load_materials():
    db = {}
    p = db_path()
    if p.exists():
        try:
            raw = json.load(open(p, encoding="utf-8"))
            for name, v in raw.items():
                E, nu = float(v["E"]), float(v["nu"])
                if E > 0 and 0 <= nu < 0.5:
                    db[str(name)] = dict(E=E, nu=nu,
                                         rp=float(v["rp"]) if v.get("rp") else None,
                                         rho=float(v["rho"]) if v.get("rho") else None)
        except Exception as e:
            bak = p.with_suffix(".defekt.json")
            try:
                p.replace(bak)
            except OSError:
                bak = p
            print(f"WARNUNG: {p.name} nicht lesbar ({e}) - Standardwerte geladen, "
                  f"defekte Datei gesichert als {bak.name}.")
    if not db:
        db = {k: dict(v) for k, v in DEFAULT_MATERIALS.items()}
        save_materials(db)
    return db


def resolve_material(db, name, E, nu):
    """Rueckgabe (Name oder None, E, nu) fuer Kommandozeilenangaben."""
    if name:
        match = [k for k in db if k.lower() == name.lower()] or \
                [k for k in db if name.lower() in k.lower()]
        if not match:
            sys.exit(f"Material '{name}' nicht in der Datenbank ({db_path()}). "
                     "Vorhanden: " + ", ".join(db))
        v = db[match[0]]
        return match[0], (E if E else v["E"]), (nu if nu is not None else v["nu"])
    if E or nu is not None:
        return None, (E or 210000.0), (nu if nu is not None else 0.3)
    if db:
        k = next(iter(db))
        return k, db[k]["E"], db[k]["nu"]
    return None, 210000.0, 0.3


UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "inch": 25.4}   # Faktor -> mm
# Mittelknoten-Reihenfolge (VTK_QUADRATIC_TETRA): Kanten 0-1, 1-2, 2-0, 0-3, 1-3, 2-3
EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
CHUNK = 2000


# ==========================================================================
# Vernetzung
# ==========================================================================
def to_vtk_order(nodes, tets):
    """gmsh und VTK nummerieren die Mittelknoten der letzten beiden Kanten
    unterschiedlich. Es wird automatisch die Reihenfolge gewaehlt, bei der die
    Mittelknoten am besten auf den Kantenmitten liegen."""
    cands = {"gmsh": [0, 1, 2, 3, 4, 5, 6, 7, 9, 8], "vtk": list(range(10))}
    best, best_score = None, np.inf
    ia = [e[0] for e in EDGES]
    ib = [e[1] for e in EDGES]
    for perm in cands.values():
        t = tets[:, perm]
        a, b, mid = nodes[t[:, ia]], nodes[t[:, ib]], nodes[t[:, 4:]]
        dev = np.linalg.norm(mid - (a + b) / 2, axis=2) / (
            np.linalg.norm(a - b, axis=2) + 1e-30)
        score = float(np.mean(dev))
        if score < best_score:
            best, best_score = t, score
    return best


def prepare_surface(path):
    """Liest ein Oberflaechennetz (STL/OBJ/PLY/VTK/VTP), bereinigt es, schliesst
    kleine Loecher und schreibt es als temporaere STL-Datei."""
    import tempfile
    surf = pv.read(str(path))
    if not isinstance(surf, pv.PolyData):
        surf = surf.extract_surface()
    surf = surf.triangulate().clean()
    if surf.n_open_edges > 0:
        print(f"Oberflaeche hat {surf.n_open_edges} offene Kanten - versuche Loecher zu schliessen ...")
        surf = surf.fill_holes(hole_size=surf.length).triangulate().clean()
    if surf.n_open_edges > 0:
        sys.exit("FEHLER: Das Oberflaechennetz ist nicht geschlossen (kein Volumenkoerper). "
                 "Bitte in der CAD-Software bzw. mit einem Mesh-Reparaturwerkzeug "
                 "(z.B. MeshLab, Meshmixer) reparieren oder als STEP exportieren.")
    try:
        surf = surf.compute_normals(consistent_normals=True, auto_orient_normals=True,
                                    split_vertices=False)
    except Exception:
        pass
    out = Path(tempfile.mkdtemp()) / "surface.stl"
    surf.save(str(out))
    return str(out)


_PREP_CACHE = {}


def mesh_stats(nodes, tets):
    """Kurzbericht ueber das Netz: Elementzahl, Kantenlaengen, schlechtestes Seitenverhaeltnis."""
    X = nodes[tets[:, :4]]
    L = np.stack([np.linalg.norm(X[:, i] - X[:, j], axis=1)
                  for i, j in EDGES], axis=1)
    asp = L.max(1) / L.min(1)
    return (f"{len(tets)} Elemente, {len(nodes)} Knoten | Kantenlaenge min/mittel/max "
            f"{L.min():.2f}/{L.mean():.2f}/{L.max():.2f} mm | Seitenverhaeltnis mittel/max "
            f"{asp.mean():.1f}/{asp.max():.1f}")


def mesh_body(path, h=None, order=2, scale=1.0, hmin=None, curv=12.0, refinements=None):
    """Tetraedernetz mit gmsh. Groessen in mm, scale = Faktor Dateieinheit -> mm.
    refinements: Liste von dict(lo=[x,y,z], hi=[x,y,z], size=mm, trans=mm) in mm.
    Rueckgabe: nodes (n,3) in mm, tets (m,4|10), info dict(h, hmin).""" 
    ext = Path(path).suffix.lower() if path else ""
    if path is not None and ext not in ALL_EXT:
        sys.exit(f"Dateityp {ext} nicht unterstuetzt. Erlaubt: " + ", ".join(ALL_EXT))
    if ext in CAD_EXT or path is None:
        scale = 1.0                              # STEP/IGES: Einheit wird von gmsh nach mm gewandelt
    if ext in SURF_EXT:
        key = (str(path), Path(path).stat().st_mtime)
        if key not in _PREP_CACHE:                 # Aufbereitung nur einmal pro Datei
            _PREP_CACHE[key] = prepare_surface(path)
        path, ext = _PREP_CACHE[key], ".stl"
    info = dict(h=None, hmin=None)
    h_file = None if h is None else h / scale    # Elementgroesse in Dateieinheiten

    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        if path is None:
            gmsh.model.add("demo")
            gmsh.model.occ.addBox(0, 0, 0, 100, 20, 20)
            gmsh.model.occ.synchronize()
        elif ext in CAD_EXT or ext == ".brep":
            try:
                gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
            except Exception:
                pass
            gmsh.model.occ.importShapes(str(path))
            gmsh.model.occ.synchronize()
        elif ext == ".stl":
            gmsh.merge(str(path))
            gmsh.model.mesh.classifySurfaces(math.pi / 4, True, True)
            gmsh.model.mesh.createGeometry()
            surfs = [e[1] for e in gmsh.model.getEntities(2)]
            loop = gmsh.model.geo.addSurfaceLoop(surfs)
            gmsh.model.geo.addVolume([loop])
            gmsh.model.geo.synchronize()
        elif ext == ".msh":
            gmsh.open(str(path))

        if ext == ".msh":                        # fertiges Volumennetz uebernehmen
            if order == 2:
                gmsh.model.mesh.setOrder(2)
        else:
            if h_file is None:
                x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(-1, -1)
                h_file = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) / (
                    12 if order == 2 else 15)
            hmin_file = h_file / 4 if hmin is None else hmin / scale
            refs = refinements or []
            if refs:
                hmin_file = min(hmin_file, min(r["size"] for r in refs) / scale)
                fids = []
                for r in refs:
                    fid = gmsh.model.mesh.field.add("Box")
                    lo, hi = np.asarray(r["lo"], float) / scale, np.asarray(r["hi"], float) / scale
                    for nm, val in (("VIn", r["size"] / scale), ("VOut", h_file),
                                    ("XMin", lo[0]), ("XMax", hi[0]), ("YMin", lo[1]),
                                    ("YMax", hi[1]), ("ZMin", lo[2]), ("ZMax", hi[2]),
                                    ("Thickness", r["trans"] / scale)):
                        gmsh.model.mesh.field.setNumber(fid, nm, float(val))
                    fids.append(fid)
                bg = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(bg, "FieldsList", fids)
                gmsh.model.mesh.field.setAsBackgroundMesh(bg)
                gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            info = dict(h=h_file * scale, hmin=hmin_file * scale)
            gmsh.option.setNumber("Mesh.MeshSizeMax", h_file)
            gmsh.option.setNumber("Mesh.MeshSizeMin", hmin_file)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", float(curv))
            gmsh.option.setNumber("Mesh.ElementOrder", order)
            try:
                gmsh.model.mesh.generate(3)
            except Exception as e:
                sys.exit(f"Vernetzung fehlgeschlagen: {e}\n"
                         "Tipp: Elementgroesse (--h) anpassen; bei STL sicherstellen, "
                         "dass der Koerper geschlossen ist.")

        ntags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords).reshape(-1, 3)
        lookup = np.full(int(ntags.max()) + 1, -1, dtype=np.int64)
        lookup[ntags.astype(np.int64)] = np.arange(len(ntags))

        etype, npe = (4, 4) if order == 1 else (11, 10)
        etypes, _, enodes = gmsh.model.mesh.getElements(3)
        if etype not in etypes:
            sys.exit("Kein Tetraedernetz in passender Ordnung erzeugt "
                     "(ist der Koerper geschlossen / enthaelt die Datei Tetraeder?).")
        tets = np.asarray(enodes[list(etypes).index(etype)], dtype=np.int64)
        tets = lookup[tets.reshape(-1, npe)]
    finally:
        gmsh.finalize()

    used = np.unique(tets)
    remap = np.full(len(coords), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    nodes, tets = coords[used] * scale, remap[tets]
    if order == 2:
        tets = to_vtk_order(nodes, tets)
    return nodes, tets, info


# ==========================================================================
# Elemente
# ==========================================================================
def shape_derivs(order, L):
    """dN/d(xi,eta,zeta) an den baryzentrischen Punkten L (P,4) -> (P,n,3)."""
    L = np.atleast_2d(L)
    dL = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    P = len(L)
    if order == 1:
        return np.broadcast_to(dL, (P, 4, 3)).copy()
    d = np.zeros((P, 10, 3))
    for i in range(4):
        d[:, i, :] = (4 * L[:, i, None] - 1) * dL[i]
    for k, (i, j) in enumerate(EDGES):
        d[:, 4 + k, :] = 4 * (L[:, i, None] * dL[j] + L[:, j, None] * dL[i])
    return d


def gauss_points(order):
    if order == 1:
        return np.full((1, 4), 0.25), np.array([1 / 6])
    a, b = 0.1381966011250105, 0.5854101966249685
    L = np.full((4, 4), a)
    np.fill_diagonal(L, b)
    return L, np.full(4, 1 / 24)


def node_bary(order):
    c = np.eye(4)
    if order == 1:
        return c
    return np.vstack([c, [0.5 * (c[i] + c[j]) for i, j in EDGES]])


def elasticity_matrix(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[range(3), range(3)] += 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def linear_solve(A, b):
    n = A.shape[0]
    if n <= 60000:
        x = spla.spsolve(A.tocsc(), b)
    else:
        A = A.tocsr()
        try:
            import pyamg
            M = pyamg.smoothed_aggregation_solver(A).aspreconditioner()
        except ImportError:
            dg = A.diagonal()
            M = spla.LinearOperator(A.shape, matvec=lambda v: v / dg)
        try:
            x, info = spla.cg(A, b, M=M, rtol=1e-8, maxiter=5000)
        except TypeError:                      # aeltere SciPy-Versionen
            x, info = spla.cg(A, b, M=M, tol=1e-8, maxiter=5000)
        if info != 0:
            print("WARNUNG: CG nicht konvergiert (info=%d)" % info)
    if not np.all(np.isfinite(x)):
        raise RuntimeError("Loesung ungueltig - Lagerung ausreichend? "
                           "(Starrkoerperbewegung moeglich)")
    return x


class FEM:
    """Lineare Elastizitaet mit Tet4/Tet10 (isoparametrisch)."""

    def __init__(self, nodes, tets, E, nu):
        self.nodes, self.tets = nodes, tets
        self.nen = tets.shape[1]
        self.order = 1 if self.nen == 4 else 2
        self.ne = 3 * self.nen
        self.m = len(tets)
        self.ndof = 3 * len(nodes)
        self.E, self.nu = E, nu
        self.D = elasticity_matrix(E, nu)
        self.edofs = (3 * tets[:, :, None] + np.arange(3)).reshape(self.m, self.ne)
        self.rows = self.cols = None
        self._element_stiffness()

    def set_material(self, E, nu):
        """Materialwechsel ohne Neuvernetzung."""
        if abs(nu - self.nu) < 1e-12:
            f = E / self.E
            self.Ke0 *= f
            self.D = self.D * f
        else:
            self.D = elasticity_matrix(E, nu)
            self.E, self.nu = E, nu
            self._element_stiffness()
        self.E, self.nu = E, nu

    def _B(self, sl, L):
        Xe = self.nodes[self.tets[sl]]                          # (c,n,3)
        dN = shape_derivs(self.order, L)                        # (P,n,3)
        J = np.einsum("pak,cad->cpkd", dN, Xe)                  # J[k,d]=dx_d/dxi_k
        detJ = np.linalg.det(J)
        Jinv = np.linalg.inv(J)
        dNdx = np.einsum("cpdk,pak->cpad", Jinv, dN)            # (c,P,n,3)
        c, P = dNdx.shape[:2]
        B = np.zeros((c, P, 6, self.ne))
        for a in range(self.nen):
            gx, gy, gz = dNdx[..., a, 0], dNdx[..., a, 1], dNdx[..., a, 2]
            k = 3 * a
            B[..., 0, k] = gx
            B[..., 1, k + 1] = gy
            B[..., 2, k + 2] = gz
            B[..., 3, k], B[..., 3, k + 1] = gy, gx
            B[..., 4, k + 1], B[..., 4, k + 2] = gz, gy
            B[..., 5, k], B[..., 5, k + 2] = gz, gx
        return B, detJ

    def _element_stiffness(self):
        Lg, wg = gauss_points(self.order)
        self.Ke0 = np.empty((self.m, self.ne, self.ne))
        self.vol = np.empty(self.m)
        dmin = np.inf
        for s in range(0, self.m, CHUNK):
            sl = slice(s, min(s + CHUNK, self.m))
            B, detJ = self._B(sl, Lg)
            dmin = min(dmin, detJ.min())
            wd = np.abs(detJ) * wg[None, :]
            DB = np.matmul(self.D, B)
            self.Ke0[sl] = np.einsum("cp,cpji,cpjl->cil", wd, B, DB, optimize=True)
            self.vol[sl] = wd.sum(1)
        if dmin <= 0:
            print("WARNUNG: Elemente mit nicht-positiver Jacobi-Determinante "
                  "(Netz/Knotenreihenfolge pruefen).")

    def assemble(self, factor=None):
        if self.rows is None:
            self.rows = np.repeat(self.edofs, self.ne, axis=1).ravel().astype(np.int32)
            self.cols = np.tile(self.edofs, (1, self.ne)).ravel().astype(np.int32)
        Ke = self.Ke0 if factor is None else self.Ke0 * factor[:, None, None]
        return sp.coo_matrix((Ke.ravel(), (self.rows, self.cols)),
                             shape=(self.ndof, self.ndof)).tocsr()

    def solve_system(self, K, F, cmask, cval):
        free = np.flatnonzero(~cmask)
        con = np.flatnonzero(cmask)
        u = np.zeros(self.ndof)
        u[con] = cval[con]
        rhs = F[free].copy()
        if np.any(cval[con] != 0):
            rhs -= K[free][:, con] @ u[con]
        u[free] = linear_solve(K[free][:, free], rhs)
        return u

    def static(self, cmask, cval, F):
        K = self.assemble()
        u = self.solve_system(K, F, cmask, cval)
        print("Summe Lasten (x,y,z):      ", F.reshape(-1, 3).sum(0).round(3))
        Rall = np.zeros(self.ndof)
        Rall[cmask] = (K @ u - F)[cmask]
        print("Summe Reaktionskraefte:    ", Rall.reshape(-1, 3).sum(0).round(3))
        return u

    def vonmises_nodes(self, u):
        L = node_bary(self.order)
        s = np.zeros(len(self.nodes))
        cnt = np.zeros(len(self.nodes))
        for a in range(0, self.m, CHUNK):
            sl = slice(a, min(a + CHUNK, self.m))
            B, _ = self._B(sl, L)
            sig = np.einsum("ij,cpjk,ck->cpi", self.D, B, u[self.edofs[sl]])
            sx, sy, sz, txy, tyz, tzx = (sig[..., i] for i in range(6))
            vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                         + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))
            np.add.at(s, self.tets[sl], vm)
            np.add.at(cnt, self.tets[sl], 1)
        return s / np.maximum(cnt, 1)

    # ---------------------------------------------------------------- SIMP
    EMIN = 1e-6

    def density_filter(self, rmin):
        cen = self.nodes[self.tets[:, :4]].mean(1)
        pairs = cKDTree(cen).query_pairs(rmin, output_type="ndarray")
        d = np.linalg.norm(cen[pairs[:, 0]] - cen[pairs[:, 1]], axis=1)
        w = rmin - d
        ar = np.arange(self.m)
        H = sp.csr_matrix((np.concatenate([w, w, np.full(self.m, rmin)]),
                           (np.concatenate([pairs[:, 0], pairs[:, 1], ar]),
                            np.concatenate([pairs[:, 1], pairs[:, 0], ar]))),
                          shape=(self.m, self.m))
        return (sp.diags(1 / np.asarray(H.sum(1)).ravel()) @ H).tocsr()

    def stiffness_factor(self, rho, p):
        return self.EMIN + (1 - self.EMIN) * rho ** p

    def topopt(self, cmask, F, vf, rmin, p, maxit, passive, callback=None):
        """Compliance-Minimierung (SIMP, Dichtefilter, OC). Rueckgabe: rho_filtered."""
        Hn = self.density_filter(rmin)
        HnT = Hn.T.tocsr()
        vol = self.vol
        zero = np.zeros(self.ndof)
        rho = np.full(self.m, vf)
        rho[passive] = 1.0
        move = 0.2
        for it in range(1, maxit + 1):
            rt = Hn @ rho
            rt[passive] = 1.0
            fac = self.stiffness_factor(rt, p)
            u = self.solve_system(self.assemble(fac), F, cmask, zero)
            ue = u[self.edofs]
            ce = np.einsum("ci,cij,cj->c", ue, self.Ke0, ue, optimize=True)
            comp = float((fac * ce).sum())
            dc = HnT @ (-p * (1 - self.EMIN) * rt ** (p - 1) * ce)
            dv = HnT @ vol
            dc = dc / max(np.abs(dc).max(), 1e-300)
            dv = dv / dv.max()
            lo_b = np.maximum(rho - move, 1e-3)
            hi_b = np.minimum(rho + move, 1.0)
            l1, l2 = 0.0, 1e9
            for _ in range(200):
                lm = 0.5 * (l1 + l2)
                rn = np.clip(rho * np.sqrt(np.maximum(-dc, 0) / (dv * lm + 1e-300)),
                             lo_b, hi_b)
                rn[passive] = 1.0
                rtn = Hn @ rn
                rtn[passive] = 1.0
                if (vol * rtn).sum() > vf * vol.sum():
                    l1 = lm
                else:
                    l2 = lm
                if (l2 - l1) / (l1 + l2 + 1e-300) < 1e-5:
                    break
            change = float(np.abs(rn - rho).max())
            rho = rn
            rt_new = Hn @ rho
            rt_new[passive] = 1.0
            vfrac = float((vol * rt_new).sum() / vol.sum())
            stop = callback(it, comp, vfrac, change, rt_new) if callback else False
            if stop or (change < 0.01 and it >= 15):
                break
        return rt_new


# ==========================================================================
# Dialoge (tkinter, mit Konsolen-Fallback)
# ==========================================================================
def _parse3(s, default=0.0):
    v = [float(t) for t in s.replace(";", ",").replace(" ", ",").split(",") if t]
    if len(v) != 3:
        raise ValueError
    return v


def ask_force():
    prompt = "Gesamtkraft Fx, Fy, Fz in N (z.B. 0, 0, -1000):"
    if USE_TK:
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()
            s = simpledialog.askstring("Kraft vorgeben", prompt, parent=root)
            root.destroy()
        except Exception:
            s = input(prompt + " ")
    else:
        s = input(prompt + " ")
    if not s:
        return None
    try:
        return np.array(_parse3(s))
    except ValueError:
        return None


def ask_support():
    """Rueckgabe: (mask[3] bool, vals[3] float) oder None."""
    if USE_TK:
        try:
            import tkinter as tk
            return _ask_support_tk(tk)
        except Exception as e:
            print("tkinter nicht nutzbar (%s) - Konsole." % e)
    s = input("Gesperrte Richtungen (z.B. xyz, xz, y): ").lower()
    mask = np.array([c in s for c in "xyz"])
    vals = np.zeros(3)
    v = input("Vorgegebene Verschiebung dx,dy,dz in mm [0,0,0]: ").strip()
    if v:
        try:
            vals = np.array(_parse3(v))
        except ValueError:
            print("Ungueltig - 0 verwendet.")
    return (mask, vals) if mask.any() else None


def _ask_support_tk(tk):
    root = tk.Tk()
    root.title("Lager definieren")
    root.attributes("-topmost", True)
    bv = [tk.BooleanVar(value=True) for _ in range(3)]
    ev = [tk.StringVar(value="0") for _ in range(3)]
    out = {}
    tk.Label(root, text="Gesperrte Richtungen (globales Koordinatensystem)\n"
                        "und optional vorgegebene Verschiebung in mm:"
             ).grid(row=0, column=0, columnspan=3, padx=10, pady=8)
    for r, name in enumerate("XYZ"):
        tk.Checkbutton(root, text=f"{name} gesperrt", variable=bv[r]
                       ).grid(row=r + 1, column=0, sticky="w", padx=10)
        tk.Label(root, text="Verschiebung [mm]:").grid(row=r + 1, column=1)
        tk.Entry(root, textvariable=ev[r], width=8).grid(row=r + 1, column=2, padx=10)

    def preset(mask):
        for v, b in zip(bv, mask):
            v.set(bool(b))

    pf = tk.Frame(root)
    pf.grid(row=4, column=0, columnspan=3, pady=6)
    for txt, mask in (("Fest (XYZ)", (1, 1, 1)), ("nur X", (1, 0, 0)),
                      ("nur Y", (0, 1, 0)), ("nur Z", (0, 0, 1)),
                      ("X+Y", (1, 1, 0)), ("X+Z", (1, 0, 1)), ("Y+Z", (0, 1, 1))):
        tk.Button(pf, text=txt, command=lambda m=mask: preset(m)).pack(side="left", padx=2)
    err = tk.Label(root, text="", fg="red")
    err.grid(row=5, column=0, columnspan=3)

    def ok():
        try:
            vals = [float(e.get().replace(",", ".")) for e in ev]
        except ValueError:
            err.config(text="Verschiebung: bitte Zahlen eingeben.")
            return
        mask = [v.get() for v in bv]
        if not any(mask):
            err.config(text="Mindestens eine Richtung sperren.")
            return
        out["r"] = (np.array(mask), np.array(vals))
        root.destroy()

    bf = tk.Frame(root)
    bf.grid(row=6, column=0, columnspan=3, pady=8)
    tk.Button(bf, text="OK", width=10, command=ok).pack(side="left", padx=6)
    tk.Button(bf, text="Abbrechen", width=10, command=root.destroy).pack(side="left", padx=6)
    root.mainloop()
    return out.get("r")


def ask_topopt(default_rmin):
    """Rueckgabe: (volfrac, rmin, penal, maxit) oder None."""
    defaults = (0.4, default_rmin, 3.0, 60)
    if USE_TK:
        try:
            import tkinter as tk
            root = tk.Tk()
            root.title("Topologieoptimierung")
            root.attributes("-topmost", True)
            labels = ("Volumenanteil (0.05-0.95)", "Filterradius [mm]",
                      "Penalisierung p", "Max. Iterationen")
            vs = [tk.StringVar(value=f"{d:.4g}") for d in defaults]
            for r, (lb, v) in enumerate(zip(labels, vs)):
                tk.Label(root, text=lb).grid(row=r, column=0, sticky="w", padx=10, pady=3)
                tk.Entry(root, textvariable=v, width=10).grid(row=r, column=1, padx=10)
            out = {}
            err = tk.Label(root, text="", fg="red")
            err.grid(row=4, column=0, columnspan=2)

            def ok():
                try:
                    a = [float(v.get().replace(",", ".")) for v in vs]
                except ValueError:
                    err.config(text="Bitte Zahlen eingeben.")
                    return
                out["r"] = (a[0], a[1], a[2], int(a[3]))
                root.destroy()

            bf = tk.Frame(root)
            bf.grid(row=5, column=0, columnspan=2, pady=8)
            tk.Button(bf, text="Start", width=10, command=ok).pack(side="left", padx=6)
            tk.Button(bf, text="Abbrechen", width=10, command=root.destroy).pack(side="left", padx=6)
            root.mainloop()
            res = out.get("r")
        except Exception as e:
            print("tkinter nicht nutzbar (%s) - Konsole." % e)
            res = _topopt_console(defaults)
    else:
        res = _topopt_console(defaults)
    if res is None:
        return None
    vf, rmin, p, it = res
    if not (0.02 < vf < 0.98) or rmin <= 0 or p < 1 or it < 1:
        print("Ungueltige Parameter.")
        return None
    return res


def _topopt_console(defaults):
    try:
        raw = input("Volumenanteil, Filterradius, p, Iterationen "
                    "[%s]: " % ", ".join(f"{d:.4g}" for d in defaults)).strip()
        if not raw:
            return defaults
        a = [float(t) for t in raw.replace(";", ",").split(",")]
        return a[0], a[1], a[2], int(a[3])
    except Exception:
        return None


def _ref_text(r):
    return (f"Box x[{r['lo'][0]:.1f},{r['hi'][0]:.1f}] y[{r['lo'][1]:.1f},{r['hi'][1]:.1f}] "
            f"z[{r['lo'][2]:.1f},{r['hi'][2]:.1f}]  h={r['size']:g}  Uebergang={r['trans']:g}")


def make_refinement(sel_box, size, trans=None):
    """Verfeinerungsbox um die markierten Knoten (mm). Rand = 1.5 x Elementgroesse."""
    pad = 1.5 * size
    return dict(lo=[float(v) for v in sel_box[0] - pad], hi=[float(v) for v in sel_box[1] + pad],
                size=float(size), trans=float(trans if trans else 3 * size))


def mesh_dialog(cfg, sel_box=None, n_sel=0):
    """Netzeinstellungen: global und lokale Verfeinerungen (aus der aktuellen Markierung).
    Rueckgabe: neue Einstellungen (dict) oder None."""
    import tkinter as tk

    root = tk.Tk()
    root.title("Netz anpassen")
    root.attributes("-topmost", True)
    pad = dict(padx=8, pady=3)
    out = {}
    refs = [dict(r) for r in cfg["refs"]]
    h0 = cfg.get("h")
    v_h = tk.StringVar(value="" if h0 is None else f"{h0:.4g}")
    v_hmin = tk.StringVar(value="" if cfg.get("hmin") is None else f"{cfg['hmin']:.4g}")
    v_curv = tk.StringVar(value=f"{cfg.get('curv', 12):g}")
    v_ord = tk.IntVar(value=cfg.get("order", 2))
    v_size, v_trans = tk.StringVar(), tk.StringVar()

    tk.Label(root, text="Globale Einstellungen", font=("TkDefaultFont", 10, "bold")
             ).grid(row=0, column=0, columnspan=3, sticky="w", **pad)
    rows = (("Max. Elementgroesse [mm]", v_h), ("Min. Elementgroesse [mm]", v_hmin),
            ("Kruemmungsverfeinerung (Elemente je 360 Grad, 0 = aus)", v_curv))
    for r, (lab, var) in enumerate(rows, start=1):
        tk.Label(root, text=lab).grid(row=r, column=0, sticky="w", **pad)
        tk.Entry(root, textvariable=var, width=10).grid(row=r, column=1, sticky="w", **pad)
    tk.Label(root, text="Elementtyp").grid(row=4, column=0, sticky="w", **pad)
    rf = tk.Frame(root)
    rf.grid(row=4, column=1, sticky="w", **pad)
    tk.Radiobutton(rf, text="Tet10", variable=v_ord, value=2).pack(side="left")
    tk.Radiobutton(rf, text="Tet4", variable=v_ord, value=1).pack(side="left")

    tk.Label(root, text="Lokale Verfeinerungen", font=("TkDefaultFont", 10, "bold")
             ).grid(row=5, column=0, columnspan=3, sticky="w", **pad)
    lb = tk.Listbox(root, height=6, width=78, exportselection=False)
    lb.grid(row=6, column=0, columnspan=3, **pad)

    def fill():
        lb.delete(0, "end")
        for r in refs:
            lb.insert("end", _ref_text(r))

    def remove():
        for i in reversed(lb.curselection()):
            del refs[i]
        fill()

    def remove_all():
        refs.clear()
        fill()

    bf = tk.Frame(root)
    bf.grid(row=7, column=0, columnspan=3, **pad)
    tk.Button(bf, text="Auswahl entfernen", command=remove).pack(side="left", padx=3)
    tk.Button(bf, text="Alle entfernen", command=remove_all).pack(side="left", padx=3)

    err = tk.Label(root, text="", fg="red", wraplength=520, justify="left")

    nf = tk.LabelFrame(root, text=("Neue Verfeinerung um die Markierung "
                                   f"({n_sel} Knoten)" if sel_box is not None
                                   else "Neue Verfeinerung: erst Flaeche mit R+Rechteck markieren"))
    nf.grid(row=8, column=0, columnspan=3, sticky="ew", **pad)
    tk.Label(nf, text="Elementgroesse [mm]").grid(row=0, column=0, **pad)
    e1 = tk.Entry(nf, textvariable=v_size, width=8)
    e1.grid(row=0, column=1, **pad)
    tk.Label(nf, text="Uebergangsbereich [mm] (leer = 3 x h)").grid(row=0, column=2, **pad)
    e2 = tk.Entry(nf, textvariable=v_trans, width=8)
    e2.grid(row=0, column=3, **pad)

    def add():
        try:
            size = float(v_size.get().replace(",", "."))
            trans = float(v_trans.get().replace(",", ".")) if v_trans.get().strip() else None
            assert size > 0 and (trans is None or trans > 0)
        except Exception:
            err.config(text="Bitte gueltige positive Zahlen eingeben.")
            return
        refs.append(make_refinement(sel_box, size, trans))
        err.config(text="")
        fill()

    btn = tk.Button(nf, text="Hinzufuegen", command=add)
    btn.grid(row=0, column=4, **pad)
    if sel_box is None:
        for w in (e1, e2, btn):
            w.config(state="disabled")
    err.grid(row=9, column=0, columnspan=3)

    def ok():
        try:
            h = float(v_h.get().replace(",", "."))
            hmin = float(v_hmin.get().replace(",", ".")) if v_hmin.get().strip() else None
            curv = float(v_curv.get().replace(",", "."))
            assert h > 0 and curv >= 0 and (hmin is None or 0 < hmin <= h)
        except Exception:
            err.config(text="Ungueltige Eingabe (h > 0, 0 < hmin <= h, Kruemmung >= 0).")
            return
        out["r"] = dict(h=h, hmin=hmin, curv=curv, order=v_ord.get(), refs=refs)
        root.destroy()

    bb = tk.Frame(root)
    bb.grid(row=10, column=0, columnspan=3, pady=8)
    tk.Button(bb, text="Neu vernetzen", command=ok).pack(side="left", padx=6)
    tk.Button(bb, text="Abbrechen", command=root.destroy).pack(side="left", padx=6)
    fill()
    root.mainloop()
    return out.get("r")


def mesh_console(cfg, sel_box=None, n_sel=0):
    """Konsolenvariante von mesh_dialog."""
    def ask(prompt, default, conv=float):
        t = input(f"{prompt} [{default}]: ").strip()
        return conv(t.replace(",", ".")) if t else default

    try:
        h = ask("Max. Elementgroesse [mm]", cfg.get("h") or 5.0)
        hmin = ask("Min. Elementgroesse [mm]", cfg.get("hmin") or h / 4)
        curv = ask("Kruemmungsverfeinerung (0 = aus)", cfg.get("curv", 12.0))
        order = ask("Elementtyp (1 = Tet4, 2 = Tet10)", cfg.get("order", 2), int)
        refs = [dict(r) for r in cfg["refs"]]
        if refs and input(f"{len(refs)} Verfeinerung(en) vorhanden - alle entfernen? [j/N]: "
                          ).strip().lower() == "j":
            refs = []
        if sel_box is not None:
            size = input(f"Verfeinerung um die Markierung ({n_sel} Knoten): Elementgroesse "
                         "[mm] (Enter = keine): ").strip()
            if size:
                refs.append(make_refinement(sel_box, float(size.replace(",", "."))))
        if h <= 0 or hmin <= 0 or hmin > h or order not in (1, 2):
            raise ValueError
        return dict(h=h, hmin=hmin, curv=curv, order=order, refs=refs)
    except ValueError:
        print("Ungueltige Eingabe.")
        return None


# ==========================================================================
# Startdialog
# ==========================================================================
def material_dialog(db, current=None, parent=None):
    """Materialdatenbank bearbeiten (neu, aendern, loeschen) und auswaehlen.
    Aenderungen werden sofort in materialien.json gespeichert.
    Rueckgabe: Name des uebernommenen Materials oder None."""
    import tkinter as tk
    from tkinter import messagebox

    own = parent is None
    win = tk.Tk() if own else tk.Toplevel(parent)
    win.title("Materialdatenbank")
    win.attributes("-topmost", True)
    pad = dict(padx=6, pady=3)
    out = {}

    lb = tk.Listbox(win, height=13, width=28, exportselection=False)
    lb.grid(row=0, column=0, rowspan=8, **pad)
    fields = [("Name", tk.StringVar()), ("E-Modul [MPa]", tk.StringVar()),
              ("Querkontraktion nu", tk.StringVar()),
              ("Streckgrenze/Festigkeit Rp [MPa] (optional)", tk.StringVar()),
              ("Dichte [g/cm3] (optional)", tk.StringVar())]
    for r, (lab, var) in enumerate(fields):
        tk.Label(win, text=lab).grid(row=r, column=1, sticky="w", **pad)
        tk.Entry(win, textvariable=var, width=22).grid(row=r, column=2, **pad)
    err = tk.Label(win, text="", fg="red", wraplength=330, justify="left")
    err.grid(row=5, column=1, columnspan=2)

    def refresh(sel=None):
        lb.delete(0, "end")
        for n in db:
            lb.insert("end", n)
        if sel in db:
            i = list(db).index(sel)
            lb.selection_clear(0, "end")
            lb.selection_set(i)
            lb.see(i)

    def fill(name):
        v = db[name]
        vals = [name, f"{v['E']:g}", f"{v['nu']:g}",
                "" if not v.get("rp") else f"{v['rp']:g}",
                "" if not v.get("rho") else f"{v['rho']:g}"]
        for (_, var), t in zip(fields, vals):
            var.set(t)

    def on_select(_=None):
        sel = lb.curselection()
        if sel:
            fill(lb.get(sel[0]))
            err.config(text="")

    lb.bind("<<ListboxSelect>>", on_select)

    def parse():
        name = fields[0][1].get().strip()
        if not name:
            raise ValueError("Bitte einen Namen eingeben.")
        num = lambda i: float(fields[i][1].get().replace(",", ".").strip())
        try:
            E, nu = num(1), num(2)
            rp = num(3) if fields[3][1].get().strip() else None
            rho = num(4) if fields[4][1].get().strip() else None
        except ValueError:
            raise ValueError("Bitte gueltige Zahlen eingeben.")
        if E <= 0 or not (0 <= nu < 0.5):
            raise ValueError("E > 0 und 0 <= nu < 0.5 erforderlich.")
        if (rp is not None and rp <= 0) or (rho is not None and rho <= 0):
            raise ValueError("Rp und Dichte muessen positiv sein.")
        return name, dict(E=E, nu=nu, rp=rp, rho=rho)

    def save():
        try:
            name, v = parse()
        except ValueError as e:
            err.config(text=str(e))
            return None
        db[name] = v
        save_materials(db)
        refresh(name)
        err.config(text=f"'{name}' gespeichert.", fg="darkgreen")
        return name

    def new():
        for _, var in fields:
            var.set("")
        lb.selection_clear(0, "end")
        err.config(text="Neues Material: Werte eintragen und 'Speichern'.", fg="black")

    def delete():
        sel = lb.curselection()
        if not sel:
            return
        name = lb.get(sel[0])
        if messagebox.askyesno("Loeschen", f"'{name}' aus der Datenbank loeschen?", parent=win):
            del db[name]
            save_materials(db)
            refresh()
            for _, var in fields:
                var.set("")

    def use():
        err.config(fg="red")
        name = save()
        if name:
            out["r"] = name
            win.destroy()

    bf = tk.Frame(win)
    bf.grid(row=8, column=1, columnspan=2, pady=8)
    for txt, cmd in (("Neu", new), ("Speichern", save), ("Loeschen", delete),
                     ("Uebernehmen", use), ("Schliessen", win.destroy)):
        tk.Button(bf, text=txt, command=cmd, width=10).pack(side="left", padx=2)

    refresh(current)
    if current in db:
        fill(current)
    if own:
        win.mainloop()
    else:
        win.grab_set()
        win.wait_window()
    return out.get("r")


def material_console(db):
    """Konsolenvariante der Materialdatenbank. Rueckgabe: Name oder None."""
    while True:
        names = list(db)
        for i, n in enumerate(names, 1):
            v = db[n]
            print(f"  {i}: {n}  E={v['E']:g} MPa  nu={v['nu']:g}"
                  + (f"  Rp={v['rp']:g}" if v.get("rp") else "")
                  + (f"  rho={v['rho']:g}" if v.get("rho") else ""))
        c = input("Nummer waehlen, n = neu, d = loeschen, Enter = abbrechen: ").strip().lower()
        if not c:
            return None
        if c.isdigit() and 1 <= int(c) <= len(names):
            return names[int(c) - 1]
        try:
            if c == "n":
                name = input("Name: ").strip()
                E = float(input("E-Modul [MPa]: ").replace(",", "."))
                nu = float(input("Querkontraktion nu: ").replace(",", "."))
                rp = input("Rp [MPa] (optional): ").strip()
                rho = input("Dichte [g/cm3] (optional): ").strip()
                if name and E > 0 and 0 <= nu < 0.5:
                    db[name] = dict(E=E, nu=nu, rp=float(rp) if rp else None,
                                    rho=float(rho) if rho else None)
                    save_materials(db)
            elif c == "d":
                k = int(input("Nummer loeschen: "))
                del db[names[k - 1]]
                save_materials(db)
        except (ValueError, IndexError, KeyError):
            print("Ungueltige Eingabe.")


def ask_startup(cfg, db):
    """tkinter-Startdialog: Bauteil, Einheit, Material, Elementtyp, Netzgroesse.
    Rueckgabe: dict oder None (Abbruch)."""
    import tkinter as tk
    from tkinter import filedialog, ttk

    root = tk.Tk()
    root.title("FEM - Bauteil laden")
    root.attributes("-topmost", True)
    out = {}
    pad = dict(padx=8, pady=4)

    path_v = tk.StringVar(value=cfg.get("path") or "")
    unit_v = tk.StringVar(value=cfg.get("unit", "mm"))
    mat_v = tk.StringVar(value=cfg.get("material") or (next(iter(db)) if db else CUSTOM))
    E_v, nu_v = tk.StringVar(), tk.StringVar()
    ord_v = tk.IntVar(value=cfg.get("order", 2))
    h_v = tk.StringVar(value="" if cfg.get("h") is None else f"{cfg['h']:g}")

    tk.Label(root, text="Bauteil (leer = Demo-Balken):").grid(row=0, column=0, sticky="w", **pad)
    tk.Entry(root, textvariable=path_v, width=48).grid(row=0, column=1, **pad)

    def browse():
        p = filedialog.askopenfilename(
            parent=root, title="Bauteil auswaehlen",
            filetypes=[("3D-Geometrie", " ".join("*" + e for e in ALL_EXT)),
                       ("Alle Dateien", "*.*")])
        if p:
            path_v.set(p)

    tk.Button(root, text="Durchsuchen ...", command=browse).grid(row=0, column=2, **pad)

    tk.Label(root, text="Einheit der Datei (STL/OBJ/...):").grid(row=1, column=0, sticky="w", **pad)
    ttk.Combobox(root, textvariable=unit_v, values=list(UNITS), width=8,
                 state="readonly").grid(row=1, column=1, sticky="w", **pad)

    tk.Label(root, text="Material:").grid(row=2, column=0, sticky="w", **pad)
    mf = tk.Frame(root)
    mf.grid(row=2, column=1, sticky="w", **pad)
    cb = ttk.Combobox(mf, textvariable=mat_v, width=26, state="readonly")
    cb.pack(side="left")
    ent = {}

    def on_mat(_=None):
        v = db.get(mat_v.get())
        if v:
            E_v.set(f"{v['E']:g}")
            nu_v.set(f"{v['nu']:g}")
        state = "normal" if mat_v.get() == CUSTOM else "disabled"
        for e in ent.values():
            e.config(state=state)

    def refresh_list():
        cb["values"] = list(db) + [CUSTOM]
        if mat_v.get() not in db and mat_v.get() != CUSTOM:
            mat_v.set(next(iter(db)) if db else CUSTOM)
        on_mat()

    def edit_db():
        n = material_dialog(db, mat_v.get(), parent=root)
        if n:
            mat_v.set(n)
        refresh_list()

    tk.Button(mf, text="Datenbank ...", command=edit_db).pack(side="left", padx=6)
    cb.bind("<<ComboboxSelected>>", on_mat)

    tk.Label(root, text="E-Modul [MPa]:").grid(row=3, column=0, sticky="w", **pad)
    ent["E"] = tk.Entry(root, textvariable=E_v, width=12)
    ent["E"].grid(row=3, column=1, sticky="w", **pad)
    tk.Label(root, text="Querkontraktionszahl:").grid(row=4, column=0, sticky="w", **pad)
    ent["nu"] = tk.Entry(root, textvariable=nu_v, width=12)
    ent["nu"].grid(row=4, column=1, sticky="w", **pad)
    if cfg.get("E") and not cfg.get("material"):
        mat_v.set(CUSTOM)
        E_v.set(f"{cfg['E']:g}")
        nu_v.set(f"{cfg.get('nu', 0.3):g}")
    refresh_list()

    tk.Label(root, text="Elementtyp:").grid(row=5, column=0, sticky="w", **pad)
    rf = tk.Frame(root)
    rf.grid(row=5, column=1, sticky="w", **pad)
    tk.Radiobutton(rf, text="Tet10 (genau)", variable=ord_v, value=2).pack(side="left")
    tk.Radiobutton(rf, text="Tet4 (schnell)", variable=ord_v, value=1).pack(side="left")

    tk.Label(root, text="Elementgroesse [mm] (leer = automatisch):").grid(
        row=6, column=0, sticky="w", **pad)
    tk.Entry(root, textvariable=h_v, width=12).grid(row=6, column=1, sticky="w", **pad)

    err = tk.Label(root, text="", fg="red")
    err.grid(row=7, column=0, columnspan=3)

    def ok():
        try:
            E = float(E_v.get().replace(",", "."))
            nu = float(nu_v.get().replace(",", "."))
            h = float(h_v.get().replace(",", ".")) if h_v.get().strip() else None
        except ValueError:
            err.config(text="Bitte gueltige Zahlen eingeben.")
            return
        if E <= 0 or not (0.0 <= nu < 0.5) or (h is not None and h <= 0):
            err.config(text="E > 0, 0 <= nu < 0.5 und h > 0 erforderlich.")
            return
        p = path_v.get().strip()
        if p and not Path(p).is_file():
            err.config(text="Datei nicht gefunden.")
            return
        out["r"] = dict(path=p or None, unit=unit_v.get(), E=E, nu=nu,
                        material=None if mat_v.get() == CUSTOM else mat_v.get(),
                        order=ord_v.get(), h=h)
        root.destroy()

    bf = tk.Frame(root)
    bf.grid(row=8, column=0, columnspan=3, pady=8)
    tk.Button(bf, text="Vernetzen und starten", command=ok).pack(side="left", padx=6)
    tk.Button(bf, text="Abbrechen", command=root.destroy).pack(side="left", padx=6)
    root.mainloop()
    return out.get("r")


# ==========================================================================
# Interaktive Oberflaeche
# ==========================================================================
class App:
    CUTS = ["keine", "Clip-Ebene", "Schnitt-Ebene"]
    BAR_TITLES = ["von Mises [MPa]", "u [mm]", "Dichte"]

    def __init__(self, nodes, tets, E, nu, title="Interaktive FEM", db=None, mat_name=None,
                 src=None, mesh_cfg=None):
        self.src = src or dict(path=None, scale=1.0)
        self.mesh_cfg = mesh_cfg or dict(h=None, hmin=None, curv=12.0,
                                         order=1 if tets.shape[1] == 4 else 2, refs=[])
        self.db = db if db is not None else {}
        self.mat_name = mat_name
        base = self.db.get(mat_name, {})
        self.props = dict(E=E, nu=nu, rp=base.get("rp"), rho=base.get("rho"))
        self._setup_mesh(nodes, tets)

        self.selection = np.array([], dtype=int)
        self.history = []        # ("support", ids, mask, vals) | ("force", ids, F)
        self._marker_names = []
        self.mode = "setup"
        self.busy = False
        self.stop = False
        self.res = None
        self.cut_i = 0
        self.field_i = 0
        self.thr = 0.5

        self.pl = pv.Plotter(title=title)
        self.pl.add_mesh(self.surf, color="lightsteelblue", show_edges=True,
                         edge_color="gray", line_width=0.5, name="body")
        self.pl.add_axes()
        self._enable_picking()
        for key, fn in (("1", self.assign_support), ("2", self.assign_force),
                        ("3", self.undo), ("4", self.clear), ("5", self.run),
                        ("6", self.run_topopt), ("7", self.next_cut),
                        ("8", self.next_field), ("9", self.export), ("m", self.choose_material), ("n", self.adjust_mesh),
                        ("0", self.back)):
            self.pl.add_key_event(key, fn)
        self.pl.add_text("R+Rechteck: markieren | 1 Lager | 2 Kraft | 3 Undo | 4 Loeschen | "
                         "5 Rechnen | 6 Topologieopt. | M Material | N Netz", position="upper_left",
                         font_size=8, name="help")
        self.update_matinfo()
        self.status(f"{self.nn} Knoten, {len(tets)} Tet{'10' if self.order == 2 else '4'}. "
                    "Mit R und Rechteck Oberflaeche markieren.")

    def _setup_mesh(self, nodes, tets):
        """Alle vom Netz abhaengigen Datenstrukturen (neu) aufbauen."""
        self.nodes, self.tets = nodes, tets
        self.nn = len(nodes)
        self.fem = FEM(nodes, tets, self.props["E"], self.props["nu"])
        self.order = self.fem.order
        t4 = tets[:, :4]
        self.corner_ids = np.unique(t4)
        self.grid = pv.UnstructuredGrid(
            np.hstack([np.full((len(t4), 1), 4), t4]).ravel(),
            np.full(len(t4), pv.CellType.TETRA), nodes)
        self.grid.point_data["node_id"] = np.arange(self.nn)
        self.surf = self.grid.extract_surface().triangulate()
        sid = np.asarray(self.surf.point_data["node_id"])
        self.tri = sid[self.surf.faces.reshape(-1, 4)[:, 1:]]          # (T,3) Eckknoten
        p = nodes[self.tri]
        self.tri_area = 0.5 * np.linalg.norm(
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        self.diag = float(np.linalg.norm(np.ptp(nodes, axis=0)))
        self._edge_maps()
        self.surf_corner = np.unique(self.tri)

    def _enable_picking(self):
        try:
            self.pl.disable_picking()
        except Exception:
            pass
        self.pl.enable_cell_picking(mesh=self.surf, callback=self.on_pick, through=False,
                                    show=False, show_message=False)

    # ---- Netz anpassen ----
    def adjust_mesh(self):
        if self.busy:
            return
        p = self.src.get("path")
        if p and Path(p).suffix.lower() == ".msh":
            return self.status("Fertige MSH-Volumennetze koennen nicht neu vernetzt werden.")
        if self.mode == "result":
            self.back()
        sel_box = None
        if len(self.selection):
            pts = self.nodes[self.selection]
            sel_box = (pts.min(0), pts.max(0))
        cfg = None
        try:
            cfg = (mesh_dialog if USE_TK else mesh_console)(self.mesh_cfg, sel_box, len(self.selection))
        except Exception as e:
            print(f"Dialog nicht verfuegbar ({e}) - Konsole.")
            cfg = mesh_console(self.mesh_cfg, sel_box, len(self.selection))
        if cfg is None:
            return self.status("Netz unveraendert.")
        self.apply_mesh(cfg)

    def apply_mesh(self, cfg):
        """Neu vernetzen; Lager und Lasten werden auf das neue Netz uebertragen."""
        self.busy = True
        self.status("Vernetze neu ... (kann etwas dauern)")
        try:
            nodes, tets, info = mesh_body(self.src.get("path"), cfg["h"], cfg["order"],
                                          self.src.get("scale", 1.0), cfg["hmin"],
                                          cfg["curv"], cfg["refs"])
        except SystemExit as e:
            self.busy = False
            return self.status(f"Vernetzung fehlgeschlagen: {e}")
        except Exception as e:
            self.busy = False
            return self.status(f"Vernetzung fehlgeschlagen: {e}")

        # Zuordnung alter -> neuer Oberflaechenknoten (naechster alter Knoten)
        old_corner, old_nodes, old_hist = self.surf_corner, self.nodes, self.history
        tree = cKDTree(old_nodes[old_corner])
        self._setup_mesh(nodes, tets)
        _, nearest = tree.query(self.nodes[self.surf_corner])
        new_hist, dropped = [], 0
        for h in old_hist:
            keep = np.isin(old_corner, h[1])[nearest]
            if not keep.any():
                dropped += 1
                continue
            new_hist.append((h[0], self.expand_selection(self.surf_corner[keep])) + tuple(h[2:]))
        self.history = new_hist
        self.mesh_cfg = dict(h=info["h"] or cfg["h"], hmin=info["hmin"] or cfg["hmin"],
                             curv=cfg["curv"], order=cfg["order"], refs=cfg["refs"])
        self.res = None
        self.selection = np.array([], dtype=int)
        self.pl.remove_actor("body")
        self.pl.add_mesh(self.surf, color="lightsteelblue", show_edges=True,
                         edge_color="gray", line_width=0.5, name="body")
        self._enable_picking()
        self.busy = False
        self.redraw()
        self.update_matinfo()
        msg = mesh_stats(nodes, tets)
        if dropped:
            msg += f" | {dropped} Zuweisung(en) nicht uebertragbar"
        if len(tets) > 60000:
            msg += " | WARNUNG: sehr gross"
        self.status(msg)

    # ---- Material ----
    def mass_g(self, rho_el=None):
        d = self.props.get("rho")
        if not d:
            return None
        v = self.fem.vol.sum() if rho_el is None else (self.fem.vol * rho_el).sum()
        return v * d * 1e-3                      # mm^3 * g/cm^3 -> g

    def update_matinfo(self):
        p = self.props
        txt = f"Material: {self.mat_name or 'eigene Werte'}\nE = {p['E']:g} MPa, nu = {p['nu']:g}"
        if p.get("rp"):
            txt += f", Rp = {p['rp']:g} MPa"
        m = self.mass_g()
        if m is not None:
            txt += f"\nMasse: {m:.1f} g"
        self.pl.add_text(txt, position="upper_right", font_size=8, name="matinfo")

    def choose_material(self):
        if self.busy:
            return
        if self.mode == "result":
            self.back()
        name = None
        try:
            if USE_TK:
                name = material_dialog(self.db, self.mat_name)
            else:
                name = material_console(self.db)
        except Exception as e:
            print(f"Dialog nicht verfuegbar ({e}) - Konsole.")
            name = material_console(self.db)
        name = name or self.mat_name           # evtl. geaenderte Werte des aktuellen Materials
        if name not in self.db:
            return self.status("Material unveraendert.")
        v = self.db[name]
        changed = (abs(v["E"] - self.fem.E) > 1e-9 or abs(v["nu"] - self.fem.nu) > 1e-12)
        if changed:
            self.fem.set_material(v["E"], v["nu"])
            self.res = None
        self.mat_name = name
        self.props = dict(E=v["E"], nu=v["nu"], rp=v.get("rp"), rho=v.get("rho"))
        self.update_matinfo()
        self.status(f"Material: {name}" + (" (Berechnung neu starten mit 5)" if changed else ""))

    # ---- Netz-Topologie fuer Auswahl / Lastverteilung ----
    def _edge_maps(self):
        N = self.nn

        def key(a, b):
            return (np.minimum(a, b).astype(np.int64) * N
                    + np.maximum(a, b).astype(np.int64))

        if self.order == 2:
            t = self.tets
            ks = np.concatenate([key(t[:, i], t[:, j]) for i, j in EDGES])
            ms = np.concatenate([t[:, 4 + k] for k in range(6)])
            self.ekeys, idx = np.unique(ks, return_index=True)
            self.emids = ms[idx]
        tk = np.concatenate([key(self.tri[:, 0], self.tri[:, 1]),
                             key(self.tri[:, 1], self.tri[:, 2]),
                             key(self.tri[:, 2], self.tri[:, 0])])
        uk = np.unique(tk)
        self.se_lo, self.se_hi = uk // N, uk % N
        if self.order == 2:
            self.se_mid = self.emids[np.searchsorted(self.ekeys, uk)]
            self.tri_mid = np.stack(
                [self.emids[np.searchsorted(self.ekeys, key(self.tri[:, i], self.tri[:, j]))]
                 for i, j in ((0, 1), (1, 2), (2, 0))], axis=1)

    def expand_selection(self, corner_ids):
        sel = np.unique(corner_ids)
        if self.order == 2:
            inS = np.zeros(self.nn, bool)
            inS[sel] = True
            m = inS[self.se_lo] & inS[self.se_hi]
            sel = np.unique(np.concatenate([sel, self.se_mid[m]]))
        return sel

    def load_weights(self, ids):
        """Konsistente Flaechenlast-Gewichte (Summe 1) fuer die Knoten ids."""
        inS = np.zeros(self.nn, bool)
        inS[ids] = True
        tm = inS[self.tri].all(1)
        w = np.zeros(self.nn)
        if tm.any():
            a = self.tri_area[tm] / 3.0
            src = self.tri[tm] if self.order == 1 else self.tri_mid[tm]
            for k in range(3):
                np.add.at(w, src[:, k], a)
        else:
            w[ids] = 1.0
        w = w[ids]
        return w / w.sum()

    # ---- Anzeige-Hilfen ----
    def status(self, msg):
        self.pl.add_text(msg, position="lower_left", font_size=9, name="status")
        self.pl.render()
        print(msg)

    def on_pick(self, picked):
        if self.mode != "setup" or self.busy or picked is None or picked.n_points == 0:
            return
        corner = np.asarray(picked.point_data["node_id"])
        self.selection = self.expand_selection(corner)
        self.pl.add_points(self.nodes[self.selection], color="yellow", point_size=8,
                           render_points_as_spheres=True, name="sel", reset_camera=False)
        self.status(f"{len(self.selection)} Knoten markiert -> 1 Lager / 2 Kraft")

    def clear_markers(self):
        for n in self._marker_names + ["sel"]:
            self.pl.remove_actor(n)
        self._marker_names = []

    def redraw(self):
        self.clear_markers()
        for i, h in enumerate(self.history):
            if h[0] == "support":
                _, ids, mask, vals = h
                color = ("dodgerblue" if np.any(vals != 0)
                         else "red" if mask.all() else "orange")
                name = f"mark{i}"
                self.pl.add_points(self.nodes[ids], color=color, point_size=9,
                                   render_points_as_spheres=True, name=name,
                                   reset_camera=False)
            else:
                _, ids, F = h
                d = F / np.linalg.norm(F)
                c = self.nodes[ids].mean(0)
                name = f"mark{i}"
                self.pl.add_arrows(c - d * 0.15 * self.diag, d * 0.15 * self.diag,
                                   color="green", name=name, mag=1.0)
            self._marker_names.append(name)
        for i, r in enumerate(self.mesh_cfg["refs"]):
            lo, hi = r["lo"], r["hi"]
            name = f"mark_ref{i}"
            self.pl.add_mesh(pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])),
                             style="wireframe", color="magenta", line_width=1.5,
                             name=name, reset_camera=False)
            self._marker_names.append(name)
        self.selection = np.array([], dtype=int)
        self.pl.render()

    # ---- Zuweisungen ----
    def _ok_setup(self):
        return self.mode == "setup" and not self.busy

    def _need_selection(self):
        if not self._ok_setup():
            return False
        if len(self.selection) == 0:
            self.status("Nichts markiert - erst mit R + Rechteck auswaehlen.")
            return False
        return True

    def assign_support(self):
        if not self._need_selection():
            return
        r = ask_support()
        if r is None:
            return self.status("Lager abgebrochen.")
        mask, vals = r
        ids = self.selection.copy()
        self.history.append(("support", ids, mask, vals))
        self.redraw()
        self.status("Lager (" + "".join(c for c, m in zip("XYZ", mask) if m)
                    + f" gesperrt) auf {len(ids)} Knoten.")

    def assign_force(self):
        if not self._need_selection():
            return
        F = ask_force()
        if F is None or not np.any(F):
            return self.status("Keine gueltige Kraft eingegeben.")
        ids = self.selection.copy()
        self.history.append(("force", ids, F))
        self.redraw()
        self.status(f"Kraft {F} N auf {len(ids)} Knoten verteilt.")

    def undo(self):
        if self._ok_setup() and self.history:
            self.history.pop()
            self.redraw()
            self.status("Letzte Zuweisung entfernt.")

    def clear(self):
        if self._ok_setup():
            self.history.clear()
            self.redraw()
            self.status("Alles geloescht.")

    # ---- Randbedingungen / Lasten ----
    def build_bc(self):
        n = 3 * self.nn
        cmask = np.zeros(n, bool)
        cval = np.zeros(n)
        F = np.zeros(n)
        for h in self.history:
            if h[0] == "support":
                _, ids, mask, vals = h
                for k in range(3):
                    if mask[k]:
                        cmask[3 * ids + k] = True
                        cval[3 * ids + k] = vals[k]
            else:
                _, ids, f = h
                w = self.load_weights(ids)
                for k in range(3):
                    np.add.at(F, 3 * ids + k, w * f[k])
        return cmask, cval, F

    def _check_bc(self, need_force=True):
        cmask, cval, F = self.build_bc()
        if not cmask.any():
            self.status("Kein Lager definiert.")
            return None
        if need_force and not np.any(F) and not np.any(cval):
            self.status("Keine Kraft bzw. vorgegebene Verschiebung definiert.")
            return None
        return cmask, cval, F

    # ---- Berechnung ----
    def run(self):
        if not self._ok_setup():
            return
        bc = self._check_bc()
        if bc is None:
            return
        cmask, cval, F = bc
        self.status("Berechne ...")
        try:
            u = self.fem.static(cmask, cval, F)
            vm = self.fem.vonmises_nodes(u)
        except Exception as e:
            return self.status(f"Fehler: {e}")
        self.res = dict(u=u.reshape(-1, 3), vm=vm, rho=None)
        self.enter_result()

    def run_topopt(self):
        if self.busy:                      # waehrend des Laufs: Abbruch
            self.stop = True
            return
        if not self._ok_setup():
            return
        bc = self._check_bc(need_force=True)
        if bc is None:
            return
        cmask, cval, F = bc
        if np.any(cval != 0):
            return self.status("Topologieopt.: vorgegebene Verschiebungen nicht unterstuetzt.")
        if not np.any(F):
            return self.status("Topologieopt. braucht mindestens eine Kraft.")
        fem = self.fem
        cen = self.nodes[self.tets[:, :4]].mean(1)
        spacing = cKDTree(cen).query(cen, k=2)[0][:, 1].mean()
        prm = ask_topopt(2.5 * spacing)
        if prm is None:
            return self.status("Optimierung abgebrochen.")
        vf, rmin, p, maxit = prm

        loaded = np.zeros(self.nn, bool)
        for h in self.history:
            if h[0] == "force":
                loaded[h[1]] = True
        frozen = cmask.reshape(-1, 3).any(1) | loaded
        passive = frozen[self.tets].any(1)
        if (fem.vol[passive].sum() > vf * fem.vol.sum()):
            self.status("WARNUNG: Fest eingefrorene Bereiche an Lager/Last "
                        "ueberschreiten den Zielvolumenanteil.")
        if fem.m > 25000:
            print("Hinweis: grosse Modelle (>25000 Elemente) sind pro Iteration langsam; "
                  "groebere Vernetzung (--h) oder --order 1 empfohlen.")

        g0 = self.grid.copy()
        self.busy, self.stop = True, False
        self.pl.remove_actor("body")
        self.pl.remove_actor("sel")

        def cb(it, comp, vfrac, change, rt):
            if it == 1 or it % 3 == 0:
                g0.cell_data["Dichte"] = rt
                shown = g0.threshold(0.5, scalars="Dichte")
                if shown.n_cells:
                    self.pl.add_mesh(shown.extract_surface(), scalars="Dichte",
                                     cmap="gray_r", clim=(0, 1), show_scalar_bar=False,
                                     name="to", reset_camera=False)
            self.status(f"TO Iteration {it}/{maxit}: Nachgiebigkeit {comp:.4g}, "
                        f"Volumen {vfrac:.3f}, Aenderung {change:.3f}  (6 = Abbruch)")
            try:
                self.pl.update()
            except Exception:
                pass
            return self.stop

        try:
            rt = fem.topopt(cmask, F, vf, rmin, p, maxit, passive, cb)
            fac = fem.stiffness_factor(rt, p)
            u = fem.solve_system(fem.assemble(fac), F, cmask, np.zeros(3 * self.nn))
            vm = fem.vonmises_nodes(u)
        except Exception as e:
            self.busy = False
            self.pl.remove_actor("to")
            self.pl.add_mesh(self.surf, color="lightsteelblue", show_edges=True,
                             edge_color="gray", line_width=0.5, name="body")
            return self.status(f"Fehler: {e}")
        self.busy = False
        self.pl.remove_actor("to")
        self.res = dict(u=u.reshape(-1, 3), vm=vm, rho=rt)
        self.enter_result(prefer="Dichte")

    # ---- Ergebnisansicht ----
    def enter_result(self, prefer=None):
        r = self.res
        g = self.grid.copy()
        g.point_data["Verschiebung"] = r["u"]
        umag = np.linalg.norm(r["u"], axis=1)
        g.point_data["u [mm]"] = umag
        g.point_data["von Mises [MPa]"] = r["vm"]
        if r["rho"] is not None:
            g.cell_data["Dichte"] = r["rho"]
        self.scale = (0.08 * self.diag / umag[self.corner_ids].max()
                      if umag.max() > 0 else 1.0)
        self.rmesh = g.warp_by_vector("Verschiebung", factor=self.scale)
        self.fields = ["von Mises [MPa]", "u [mm]"] + (["Dichte"] if r["rho"] is not None else [])
        self.field_i = self.fields.index(prefer) if prefer in self.fields else 0
        self.cut_i = 0
        self.mode = "result"
        self.pl.remove_actor("body")
        self.clear_markers()
        self.draw_result()

    def draw_result(self):
        pl = self.pl
        for fn in (pl.clear_plane_widgets, pl.clear_slider_widgets):
            try:
                fn()
            except Exception:
                pass
        for n in ("result", "ghost"):
            pl.remove_actor(n)
        for t in self.BAR_TITLES:
            try:
                pl.remove_scalar_bar(t)
            except Exception:
                pass
        mesh = self.rmesh
        field, cut = self.fields[self.field_i], self.CUTS[self.cut_i]
        if field == "Dichte":
            clim, cmap = (0.0, 1.0), "gray_r"
            base = mesh.threshold(self.thr, scalars="Dichte")
        else:
            arr = mesh.point_data[field][self.corner_ids]
            clim, cmap = (float(arr.min()), float(arr.max())), "turbo"
            base = mesh
        if base.n_cells == 0:
            return self.status("Leere Auswahl (Dichte-Schwelle zu hoch).")
        kw = dict(scalars=field, cmap=cmap, clim=clim, name="result",
                  scalar_bar_args=dict(title=field))
        if cut == "keine":
            pl.add_mesh(base, **kw)
            if field == "Dichte":
                def thr_cb(v):
                    self.thr = float(v)
                    sh = mesh.threshold(self.thr, scalars="Dichte")
                    if sh.n_cells:
                        pl.add_mesh(sh, **kw)
                pl.add_slider_widget(thr_cb, [0.05, 0.95], value=self.thr,
                                     title="Dichte-Schwelle")
        elif cut == "Clip-Ebene":
            pl.add_mesh_clip_plane(base, normal="x", **kw)
        else:
            pl.add_mesh_slice(base, normal="x", **kw)
            pl.add_mesh(mesh.extract_surface(), style="wireframe", color="gray",
                        opacity=0.15, name="ghost")
        pl.render()
        extra = ""
        if self.res["rho"] is None:
            vmax = self.res["vm"][self.corner_ids].max()
            extra = (f"max von Mises {vmax:.2f} MPa, "
                     f"max u {np.linalg.norm(self.res['u'], axis=1).max():.4f} mm, ")
            if self.props.get("rp") and vmax > 0:
                extra += f"Sicherheit Rp/vM {self.props['rp'] / vmax:.2f} (Spitzenwert), "
        else:
            m, m0 = self.mass_g(self.res["rho"]), self.mass_g()
            if m is not None:
                extra = f"Masse {m:.1f} g (vorher {m0:.1f} g), "
        self.status(f"{field} | Schnitt: {cut} | {extra}{self.scale:.1f}x ueberhoeht | "
                    "7 Schnitt, 8 Feld, 9 Export, 0 zurueck")

    def next_cut(self):
        if self.mode == "result" and not self.busy:
            self.cut_i = (self.cut_i + 1) % len(self.CUTS)
            self.draw_result()

    def next_field(self):
        if self.mode == "result" and not self.busy:
            self.field_i = (self.field_i + 1) % len(self.fields)
            self.draw_result()

    def export(self):
        if self.mode != "result" or self.busy:
            return
        self.rmesh.save("ergebnis.vtu")
        msg = "ergebnis.vtu gespeichert"
        if self.res["rho"] is not None:
            g0 = self.grid.copy()
            g0.cell_data["Dichte"] = self.res["rho"]
            sh = g0.threshold(self.thr, scalars="Dichte")
            if sh.n_cells:
                sh.extract_surface().triangulate().clean().save("optimiert.stl")
                msg += f", optimiert.stl (Schwelle {self.thr:.2f})"
        self.status(msg)

    def back(self):
        if self.mode != "result" or self.busy:
            return
        for fn in (self.pl.clear_plane_widgets, self.pl.clear_slider_widgets):
            try:
                fn()
            except Exception:
                pass
        for n in ("result", "ghost"):
            self.pl.remove_actor(n)
        for t in self.BAR_TITLES:
            try:
                self.pl.remove_scalar_bar(t)
            except Exception:
                pass
        self.pl.add_mesh(self.surf, color="lightsteelblue", show_edges=True,
                         edge_color="gray", line_width=0.5, name="body")
        self.mode = "setup"
        self.redraw()
        self.status("Zurueck in der Lager-/Lastansicht.")

    def show(self):
        self.pl.show()


# ==========================================================================
def main():
    global USE_TK
    ap = argparse.ArgumentParser(description="Interaktive lineare FEM + Topologieoptimierung")
    ap.add_argument("datei", nargs="?",
                    help="Bauteil: " + "/".join(e[1:] for e in ALL_EXT) + " (leer = Startdialog)")
    ap.add_argument("--h", type=float, default=None, help="Elementgroesse in mm")
    ap.add_argument("--hmin", type=float, default=None,
                    help="Minimale Elementgroesse in mm (Standard: h/4)")
    ap.add_argument("--kruemmung", type=float, default=12.0,
                    help="Kruemmungsverfeinerung: Elemente je 360 Grad (0 = aus)")
    ap.add_argument("--order", type=int, choices=(1, 2), default=2,
                    help="1 = Tet4, 2 = Tet10 (Standard)")
    ap.add_argument("--material", default=None,
                    help="Name (oder Teil davon) aus der Materialdatenbank materialien.json")
    ap.add_argument("--E", type=float, default=None, help="E-Modul in MPa (ueberschreibt Material)")
    ap.add_argument("--nu", type=float, default=None, help="Querkontraktionszahl (ueberschreibt Material)")
    ap.add_argument("--einheit", choices=list(UNITS), default="mm",
                    help="Laengeneinheit der Datei (STL/OBJ/...; STEP/IGES automatisch)")
    ap.add_argument("--konsole", action="store_true",
                    help="Eingaben in der Konsole statt in tkinter-Dialogen")
    args = ap.parse_args()
    USE_TK = not args.konsole

    db = load_materials()
    mname, E, nu = resolve_material(db, args.material, args.E, args.nu)
    cfg = dict(path=args.datei, h=args.h, order=args.order, E=E, nu=nu,
               unit=args.einheit, material=mname)
    if args.datei is None and USE_TK and len(sys.argv) == 1:
        try:
            cfg = ask_startup(cfg, db)
        except Exception as e:
            print(f"Startdialog nicht verfuegbar ({e}) - Demo-Balken.")
        if cfg is None:
            return

    title = Path(cfg["path"]).name if cfg["path"] else "Demo-Balken"
    print(f"Vernetze {title} ...")
    nodes, tets, info = mesh_body(cfg["path"], cfg["h"], cfg["order"], UNITS[cfg["unit"]],
                                  args.hmin, args.kruemmung)
    size = np.ptp(nodes, axis=0)
    print(f"{len(nodes)} Knoten, {len(tets)} Tetraeder (Ordnung {cfg['order']}); "
          f"Abmessungen {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm")
    if size.max() < 1.0 or size.max() > 20000:
        print("WARNUNG: Ungewoehnliche Bauteilgroesse - stimmt die Einheit der Datei?")
    if len(tets) > 60000:
        print("WARNUNG: Sehr grosses Netz - Rechenzeit und Speicherbedarf beachten "
              "(groessere --h oder --order 1 waehlen).")
    print(mesh_stats(nodes, tets))
    mesh_cfg = dict(h=info["h"], hmin=info["hmin"], curv=args.kruemmung,
                    order=cfg["order"], refs=[])
    src = dict(path=cfg["path"], scale=UNITS[cfg["unit"]])
    App(nodes, tets, cfg["E"], cfg["nu"], title=f"FEM - {title}",
        db=db, mat_name=cfg.get("material"), src=src, mesh_cfg=mesh_cfg).show()


if __name__ == "__main__":
    main()