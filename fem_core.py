"""
fem_core.py - Rechenkern fuer die interaktive FEM (ohne Grafik, ohne GUI)
=========================================================================

Enthaelt: Vernetzung (gmsh), Tet4/Tet10-FEM, Topologieoptimierung (SIMP),
Materialdatenbank, Setup-Datei (JSON) und VTK-Ausgabe.
Das Grafikprogramm ``fem_gui.py`` nutzt dieses Modul; es laeuft aber auch
allein auf einem Server ohne Display:

    python fem_core.py run  setup.json                 # rechnen -> setup_ergebnis.vtk
    python fem_core.py run  setup.json --out erg.vtk    # Ausgabedatei festlegen
    python fem_core.py check setup.json                 # nur Netz/Randbedingungen pruefen

Benoetigt auf dem Server nur: numpy, scipy, gmsh  (optional: pyamg, pyvista).

Einheiten: mm, N, MPa, g/cm^3.

Aufbau der Setup-Datei (JSON)
-----------------------------
geometry.file       Bauteil (STEP/STL/...) oder null = Demo-Balken 100x20x20 mm
geometry.unit       Einheit bei STL/OBJ/... (mm, cm, m, inch)
mesh.*              h, hmin, curv, order (1|2), refinements, file (gespeichertes Netz .npz)
material            name, E [MPa], nu, rp [MPa], rho [g/cm^3]
supports[]          Lager:  mask [x,y,z], values [mm] und eine Knotenauswahl (siehe unten)
loads[]             Lasten: force [Fx,Fy,Fz] in N, flaechengewichtet verteilt
analysis.type       "static" oder "topopt" (Parameter in analysis.topopt; mit save_dir werden
                    Zwischenergebnisse je save_every Iterationen als VTK + verlauf.csv gespeichert)
output.file         Ergebnisdatei (.vtk oder .vtu)

Knotenauswahl einer Randbedingung (eine der Varianten):
  "ids" + "points"  von der GUI gespeichert (ids gelten fuer das mitgespeicherte Netz)
  "points"          Knotenkoordinaten (naechster Netzknoten)
  "select"          von Hand schreibbar, z.B.
                    {"plane": {"axis": "x", "value": 0}}
                    {"box": {"lo": [0,0,0], "hi": [10,20,20]}}
                    {"sphere": {"center": [50,10,10], "radius": 5}}
                    (mehrere Bedingungen werden verundet; gewaehlt werden Oberflaechenknoten)
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


class FemError(Exception):
    """Fehler mit verstaendlicher Meldung (kein Programmierfehler)."""


_log = print


def set_logger(fn):
    global _log
    _log = fn


def log(msg=""):
    _log(msg)


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
        log(f"WARNUNG: Materialdatenbank konnte nicht gespeichert werden: {e}")


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
            log(f"WARNUNG: {p.name} nicht lesbar ({e}) - Standardwerte geladen, "
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
            raise FemError(f"Material '{name}' nicht in der Datenbank ({db_path()}). "
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


_PREP_CACHE = {}


def prepare_surface(path):
    """Liest ein Oberflaechennetz (STL/OBJ/PLY/VTK/VTP), bereinigt es, schliesst kleine
    Loecher und schreibt es als temporaere STL-Datei (benoetigt pyvista)."""
    import tempfile
    try:
        import pyvista as pv
    except ImportError:
        if Path(path).suffix.lower() == ".stl":
            log("pyvista nicht installiert - STL wird ohne Bereinigung verwendet.")
            return str(path)
        raise FemError("Fuer dieses Dateiformat wird pyvista benoetigt; bitte als STL/STEP speichern.")
    surf = pv.read(str(path))
    if not isinstance(surf, pv.PolyData):
        surf = surf.extract_surface()
    surf = surf.triangulate().clean()
    if surf.n_open_edges > 0:
        log(f"Oberflaeche hat {surf.n_open_edges} offene Kanten - versuche Loecher zu schliessen ...")
        surf = surf.fill_holes(hole_size=surf.length).triangulate().clean()
    if surf.n_open_edges > 0:
        raise FemError("Das Oberflaechennetz ist nicht geschlossen (kein Volumenkoerper). "
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


def mesh_stats(nodes, tets):
    """Kurzbericht ueber das Netz."""
    X = nodes[tets[:, :4]]
    L = np.stack([np.linalg.norm(X[:, i] - X[:, j], axis=1) for i, j in EDGES], axis=1)
    asp = L.max(1) / L.min(1)
    return (f"{len(tets)} Elemente, {len(nodes)} Knoten | Kantenlaenge min/mittel/max "
            f"{L.min():.2f}/{L.mean():.2f}/{L.max():.2f} mm | Seitenverhaeltnis mittel/max "
            f"{asp.mean():.1f}/{asp.max():.1f}")


def _resolve_geometry(path, scale):
    ext = Path(path).suffix.lower() if path else ""
    if path is not None and ext not in ALL_EXT:
        raise FemError(f"Dateityp {ext} nicht unterstuetzt. Erlaubt: " + ", ".join(ALL_EXT))
    if ext in CAD_EXT or path is None:
        scale = 1.0                         # STEP/IGES: Einheit wird von gmsh nach mm gewandelt
    if ext in SURF_EXT:
        key = (str(path), Path(path).stat().st_mtime)
        if key not in _PREP_CACHE:          # Aufbereitung nur einmal pro Datei
            _PREP_CACHE[key] = prepare_surface(path)
        path, ext = _PREP_CACHE[key], ".stl"
    return path, ext, scale


def _import_gmsh():
    try:
        import gmsh
    except ImportError:
        raise FemError("gmsh ist nicht installiert (pip install gmsh).")
    return gmsh


def _import_geometry(gmsh, path, ext):
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


def preview_surface(path, scale=1.0):
    """Grobes Oberflaechennetz zur Anzeige der Geometrie vor dem Vernetzen.
    Rueckgabe: nodes (n,3) in mm, tris (m,3)."""
    path, ext, scale = _resolve_geometry(path, scale)
    gmsh = _import_gmsh()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        _import_geometry(gmsh, path, ext)
        if ext != ".msh":
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(-1, -1)
            diag = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
            gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 30)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 10)
            gmsh.model.mesh.generate(2)
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords).reshape(-1, 3)
        lookup = np.full(int(ntags.max()) + 1, -1, dtype=np.int64)
        lookup[ntags.astype(np.int64)] = np.arange(len(ntags))
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        if 2 not in etypes:
            raise FemError("Keine Oberflaechendreiecke gefunden.")
        tris = lookup[np.asarray(enodes[list(etypes).index(2)], dtype=np.int64).reshape(-1, 3)]
    finally:
        gmsh.finalize()
    used = np.unique(tris)
    remap = np.full(len(coords), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return coords[used] * scale, remap[tris]


def mesh_body(path, h=None, order=2, scale=1.0, hmin=None, curv=12.0, refinements=None):
    """Tetraedernetz mit gmsh. Groessen in mm, scale = Faktor Dateieinheit -> mm.
    refinements: Liste von dict(lo=[x,y,z], hi=[x,y,z], size=mm, trans=mm) in mm.
    Rueckgabe: nodes (n,3) in mm, tets (m,4|10), info dict(h, hmin)."""
    path, ext, scale = _resolve_geometry(path, scale)
    h_file = None if h is None else h / scale
    info = dict(h=None, hmin=None)
    gmsh = _import_gmsh()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        _import_geometry(gmsh, path, ext)
        if ext == ".msh":                        # fertiges Volumennetz uebernehmen
            if order == 2:
                gmsh.model.mesh.setOrder(2)
            if refinements:
                log("Hinweis: MSH-Netze koennen nicht lokal verfeinert werden.")
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
                raise FemError(f"Vernetzung fehlgeschlagen: {e}. Tipp: Elementgroesse anpassen; "
                               "bei STL sicherstellen, dass der Koerper geschlossen ist.")

        ntags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords).reshape(-1, 3)
        lookup = np.full(int(ntags.max()) + 1, -1, dtype=np.int64)
        lookup[ntags.astype(np.int64)] = np.arange(len(ntags))
        etype, npe = (4, 4) if order == 1 else (11, 10)
        etypes, _, enodes = gmsh.model.mesh.getElements(3)
        if etype not in etypes:
            raise FemError("Kein Tetraedernetz in passender Ordnung erzeugt "
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


def make_refinement(sel_box, size, trans=None):
    """Verfeinerungsbox um die markierten Knoten (mm). Rand = 1.5 x Elementgroesse."""
    pad = 1.5 * size
    return dict(lo=[float(v) for v in np.asarray(sel_box[0]) - pad],
                hi=[float(v) for v in np.asarray(sel_box[1]) + pad],
                size=float(size), trans=float(trans if trans else 3 * size))


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
            log("WARNUNG: CG nicht konvergiert (info=%d)" % info)
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
            log("WARNUNG: Elemente mit nicht-positiver Jacobi-Determinante "
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
        log("Summe Lasten (x,y,z):       " + str(F.reshape(-1, 3).sum(0).round(3)))
        Rall = np.zeros(self.ndof)
        Rall[cmask] = (K @ u - F)[cmask]
        log("Summe Reaktionskraefte:     " + str(Rall.reshape(-1, 3).sum(0).round(3)))
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
            stop = callback(it, comp, vfrac, change, rt_new, u, rt) if callback else False
            if stop or (change < 0.01 and it >= 15):
                break
        return rt_new



# ==========================================================================
# Netzmodell mit Oberflaechentopologie (ohne Grafikbibliothek)
# ==========================================================================
class MeshModel:
    """Tet4/Tet10-Netz mit Randflaechen, Knotennormalen und Auswahlhilfen."""

    def __init__(self, nodes, tets):
        self.nodes = np.asarray(nodes, float)
        self.tets = np.asarray(tets, np.int64)
        self.nn = len(self.nodes)
        self.order = 1 if self.tets.shape[1] == 4 else 2
        self.corner_ids = np.unique(self.tets[:, :4])
        self.diag = float(np.linalg.norm(np.ptp(self.nodes, axis=0)))
        self._tree = None
        self._labels = {}
        self._n2t = None
        self._boundary()
        self._edge_maps()
        self._normals()

    # ---- Randflaechen (nach aussen orientiert) ----
    def _boundary(self):
        t = self.tets[:, :4]
        F = np.concatenate([t[:, [0, 1, 2]], t[:, [0, 1, 3]], t[:, [0, 2, 3]], t[:, [1, 2, 3]]])
        opp = np.concatenate([t[:, 3], t[:, 2], t[:, 1], t[:, 0]])
        _, idx, cnt = np.unique(np.sort(F, axis=1), axis=0, return_index=True, return_counts=True)
        sel = idx[cnt == 1]
        tri, opp = F[sel].copy(), opp[sel]
        P = self.nodes[tri]
        n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        flip = np.einsum("ij,ij->i", n, self.nodes[opp] - P.mean(1)) > 0
        tri[flip] = tri[flip][:, [0, 2, 1]]
        P = self.nodes[tri]
        n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        area = 0.5 * np.linalg.norm(n, axis=1)
        self.tri = tri
        self.tri_area = area
        self.tri_normal = n / np.maximum(2 * area, 1e-300)[:, None]
        self.tri_centroid = P.mean(1)
        self.surf_corner = np.unique(tri)

    def _edge_maps(self):
        N = self.nn

        def key(a, b):
            return (np.minimum(a, b).astype(np.int64) * N + np.maximum(a, b).astype(np.int64))

        tri = self.tri
        if self.order == 2:
            t = self.tets
            ks = np.concatenate([key(t[:, i], t[:, j]) for i, j in EDGES])
            ms = np.concatenate([t[:, 4 + k] for k in range(6)])
            self.ekeys, idx = np.unique(ks, return_index=True)
            self.emids = ms[idx]
        tk = np.concatenate([key(tri[:, 0], tri[:, 1]), key(tri[:, 1], tri[:, 2]),
                             key(tri[:, 2], tri[:, 0])])
        uk, inv = np.unique(tk, return_inverse=True)
        T = len(tri)
        ti = np.tile(np.arange(T), 3)
        o = np.argsort(inv, kind="stable")
        inv_s, ti_s = inv[o], ti[o]
        start = np.searchsorted(inv_s, np.arange(len(uk)))
        nxt = np.minimum(start + 1, len(inv_s) - 1)
        two = (inv_s[nxt] == np.arange(len(uk))) & (nxt != start)
        self.edge_tris = np.stack([ti_s[start], np.where(two, ti_s[nxt], -1)], axis=1)
        self.se_lo, self.se_hi = uk // N, uk % N
        if self.order == 2:
            self.se_mid = self.emids[np.searchsorted(self.ekeys, uk)]
            self.tri_mid = np.stack(
                [self.emids[np.searchsorted(self.ekeys, key(tri[:, i], tri[:, j]))]
                 for i, j in ((0, 1), (1, 2), (2, 0))], axis=1)
            self.tri_all = np.hstack([tri, self.tri_mid])
        else:
            self.tri_all = tri

    def _normals(self):
        nrm = np.zeros((self.nn, 3))
        w = self.tri_normal * self.tri_area[:, None]
        for k in range(3):
            np.add.at(nrm, self.tri[:, k], w)
        if self.order == 2:
            nrm[self.se_mid] = nrm[self.se_lo] + nrm[self.se_hi]
            self.surf_nodes = np.unique(np.concatenate([self.surf_corner, self.se_mid]))
        else:
            self.surf_nodes = self.surf_corner
        ln = np.linalg.norm(nrm, axis=1)
        ln[ln == 0] = 1.0
        self.node_normal = nrm / ln[:, None]

    # ---- Auswahlhilfen ----
    def expand_selection(self, corner_ids):
        """Eckknoten -> inkl. Mittelknoten aller Oberflaechenkanten zwischen gewaehlten Ecken."""
        sel = np.unique(corner_ids)
        if self.order == 2:
            inS = np.zeros(self.nn, bool)
            inS[sel] = True
            m = inS[self.se_lo] & inS[self.se_hi]
            sel = np.unique(np.concatenate([sel, self.se_mid[m]]))
        return sel

    def tris_of_node(self, i):
        if self._n2t is None:
            T = len(self.tri)
            cols = np.repeat(np.arange(T), self.tri_all.shape[1])
            self._n2t = sp.csr_matrix((np.ones(len(cols)), (self.tri_all.ravel(), cols)),
                                      shape=(self.nn, T)).tocsr()
        return self._n2t[i].indices

    def face_labels(self, angle_deg):
        """Zusammenhaengende Dreiecksgebiete, getrennt durch Kanten mit Knickwinkel > angle."""
        key = round(float(angle_deg), 3)
        if key in self._labels:
            return self._labels[key]
        T, N = len(self.tri), self.nn
        tri = self.tri
        tk = np.concatenate([np.minimum(tri[:, k], tri[:, (k + 1) % 3]).astype(np.int64) * N
                             + np.maximum(tri[:, k], tri[:, (k + 1) % 3]) for k in range(3)])
        ti = np.tile(np.arange(T), 3)
        o = np.argsort(tk, kind="stable")
        ks, tis = tk[o], ti[o]
        same = ks[1:] == ks[:-1]
        a, b = tis[:-1][same], tis[1:][same]
        ok = np.einsum("ij,ij->i", self.tri_normal[a], self.tri_normal[b]) >= math.cos(
            math.radians(angle_deg))
        g = sp.coo_matrix((np.ones(ok.sum()), (a[ok], b[ok])), shape=(T, T))
        _, labels = connected_components(g, directed=False)
        self._labels[key] = labels
        return labels

    def nodes_of_edges(self, edge_idx):
        """Knoten (inkl. Mittelknoten bei Tet10) der Oberflaechenkanten mit den Indizes edge_idx."""
        e = np.asarray(edge_idx).ravel()
        parts = [self.se_lo[e], self.se_hi[e]]
        if self.order == 2:
            parts.append(self.se_mid[e])
        return np.unique(np.concatenate(parts))

    def feature_edges(self, angle_deg):
        """Kanten, an denen die beiden Nachbardreiecke um mehr als angle_deg abknicken."""
        t1, t2 = self.edge_tris[:, 0], self.edge_tris[:, 1]
        dots = np.einsum("ij,ij->i", self.tri_normal[t1], self.tri_normal[np.where(t2 >= 0, t2, 0)])
        return (t2 >= 0) & (dots < math.cos(math.radians(angle_deg)))

    def edge_chains(self, angle_deg):
        """Kantenzuege (Geometriekanten): Label je Oberflaechenkante, -1 = keine Merkmalskante.
        Ein Zug endet an Knoten, an denen sich mehr/weniger als zwei Merkmalskanten treffen
        oder die Richtung um mehr als angle_deg knickt."""
        key = ("chain", round(float(angle_deg), 3))
        if key in self._labels:
            return self._labels[key]
        fe = np.flatnonzero(self.feature_edges(angle_deg))
        labels = np.full(len(self.se_lo), -1, dtype=np.int64)
        if len(fe):
            lo, hi = self.se_lo[fe], self.se_hi[fe]
            deg = np.bincount(np.concatenate([lo, hi]), minlength=self.nn)
            cat_n = np.concatenate([lo, hi])
            cat_e = np.concatenate([np.arange(len(fe)), np.arange(len(fe))])
            o = np.argsort(cat_n, kind="stable")
            ns, es = cat_n[o], cat_e[o]
            i = np.flatnonzero((ns[:-1] == ns[1:]) & (deg[ns[:-1]] == 2))
            v, e1, e2 = ns[i], es[i], es[i + 1]
            o1 = np.where(lo[e1] == v, hi[e1], lo[e1])
            o2 = np.where(lo[e2] == v, hi[e2], lo[e2])
            d1, d2 = self.nodes[o1] - self.nodes[v], self.nodes[o2] - self.nodes[v]
            cosang = np.einsum("ij,ij->i", d1, d2) / np.maximum(
                np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1), 1e-300)
            cont = cosang <= -math.cos(math.radians(angle_deg))     # fast gerade Fortsetzung
            g = sp.coo_matrix((np.ones(int(cont.sum())), (e1[cont], e2[cont])),
                              shape=(len(fe), len(fe)))
            _, comp = connected_components(g, directed=False)
            labels[fe] = comp
        self._labels[key] = labels
        return labels

    def nodes_of_tris(self, tri_mask):
        return np.unique(self.tri_all[tri_mask].ravel())

    def map_points(self, points):
        """Naechster Netzknoten je Punkt. Rueckgabe: (ids, max. Abstand)."""
        if self._tree is None:
            self._tree = cKDTree(self.nodes)
        d, idx = self._tree.query(np.asarray(points, float))
        return np.unique(idx), float(d.max()) if len(d) else 0.0

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
            inE = inS[self.se_lo] & inS[self.se_hi]
            if inE.any():                          # Linienlast entlang gewaehlter Kanten
                lo, hi = self.se_lo[inE], self.se_hi[inE]
                L = np.linalg.norm(self.nodes[hi] - self.nodes[lo], axis=1)
                if self.order == 1:
                    np.add.at(w, lo, L / 2)
                    np.add.at(w, hi, L / 2)
                else:
                    np.add.at(w, lo, L / 6)
                    np.add.at(w, hi, L / 6)
                    np.add.at(w, self.se_mid[inE], 2 * L / 3)
            else:
                w[ids] = 1.0
        w = w[ids]
        if w.sum() <= 0:
            w = np.ones(len(ids))
        return w / w.sum()


def _seg_dist(P, X, Y):
    d = Y - X
    t = np.clip(np.einsum("ij,ij->i", P - X, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-300), 0, 1)
    return np.linalg.norm(P - (X + t[:, None] * d), axis=1)


def point_triangle_distance(P, A, B, C):
    """Abstand Punkt-Dreieck (vektorisiert ueber Paare)."""
    ab, ac, ap = B - A, C - A, P - A
    n = np.cross(ab, ac)
    nn = np.maximum(np.einsum("ij,ij->i", n, n), 1e-300)
    d00, d01, d11 = (np.einsum("ij,ij->i", ab, ab), np.einsum("ij,ij->i", ab, ac),
                     np.einsum("ij,ij->i", ac, ac))
    d20, d21 = np.einsum("ij,ij->i", ap, ab), np.einsum("ij,ij->i", ap, ac)
    den = np.maximum(d00 * d11 - d01 ** 2, 1e-300)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)
    plane = np.abs(np.einsum("ij,ij->i", ap, n)) / np.sqrt(nn)
    edge = np.minimum(np.minimum(_seg_dist(P, A, B), _seg_dist(P, B, C)), _seg_dist(P, C, A))
    return np.where(inside, plane, edge)


def transfer_ids(old, new, ids):
    """Knotenauswahl auf ein neues Netz uebertragen (z.B. nach Neuvernetzung).
    Flaechen: neue Oberflaechenknoten auf den markierten Dreiecken; Kanten: neue Knoten auf den
    markierten Kanten; sonst jeweils der naechste neue Knoten."""
    ids = np.asarray(ids)
    inS = np.zeros(old.nn, bool)
    inS[ids] = True
    P = new.nodes[new.surf_corner]
    Pn = new.nodes[new.tri]
    tol = 0.4 * float(np.mean(np.linalg.norm(Pn[:, 1] - Pn[:, 0], axis=1)))
    tm = inS[old.tri].all(1)
    keep = np.zeros(len(P), bool)
    if tm.any():
        sel = old.tri[tm]
        A, B, C = (old.nodes[sel[:, k]] for k in range(3))
        cen = (A + B + C) / 3
        rad = np.maximum.reduce([np.linalg.norm(A - cen, axis=1), np.linalg.norm(B - cen, axis=1),
                                 np.linalg.norm(C - cen, axis=1)])
        cand = cKDTree(cen).query_ball_point(P, rad.max() + tol)
        pi = np.repeat(np.arange(len(P)), [len(c) for c in cand])
        ti = np.concatenate([np.asarray(c, np.int64) for c in cand]) if len(pi) else np.zeros(0, np.int64)
        if len(pi):
            keep[pi[point_triangle_distance(P[pi], A[ti], B[ti], C[ti]) <= tol]] = True
    else:
        inE = inS[old.se_lo] & inS[old.se_hi]
        if inE.any():
            A, B = old.nodes[old.se_lo[inE]], old.nodes[old.se_hi[inE]]
            mid, rad = (A + B) / 2, np.linalg.norm(B - A, axis=1) / 2
            cand = cKDTree(mid).query_ball_point(P, rad.max() + tol)
            pi = np.repeat(np.arange(len(P)), [len(c) for c in cand])
            ti = np.concatenate([np.asarray(c, np.int64) for c in cand]) if len(pi) else np.zeros(0, np.int64)
            if len(pi):
                keep[pi[_seg_dist(P[pi], A[ti], B[ti]) <= tol]] = True
    if keep.any():
        return new.expand_selection(new.surf_corner[keep])
    return new.map_points(old.nodes[ids])[0]


# ==========================================================================
# Setup (JSON), Randbedingungen, Analyse, Ausgabe
# ==========================================================================
def default_setup():
    return dict(
        version=1,
        geometry=dict(file=None, unit="mm"),
        mesh=dict(h=None, hmin=None, curv=12.0, order=2, refinements=[], file=None),
        material=dict(name=None, E=210000.0, nu=0.3, rp=None, rho=None),
        supports=[], loads=[],
        analysis=dict(type="static",
                      topopt=dict(volfrac=0.4, rmin=None, penal=3.0, maxit=60,
                                  save_dir=None, save_every=5, save_vtk=True)),
        output=dict(file=None))


def _merge(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def resolve_path(base_dir, p):
    p = Path(p)
    return p if p.is_absolute() else Path(base_dir) / p


def _rel(base_dir, p):
    try:
        return os.path.relpath(p, base_dir)
    except ValueError:
        return str(p)


def setup_to_dict(setup, model=None, base_dir=".", mesh_file=None):
    """JSON-faehige Kopie des Setups. ids/points werden aus dem Netz bestimmt."""
    d = json.loads(json.dumps(
        {k: v for k, v in setup.items() if k not in ("supports", "loads")}, default=float))
    g = d["geometry"]
    if g.get("file"):
        g["file"] = _rel(base_dir, g["file"])
    d["mesh"]["file"] = mesh_file
    for key in ("supports", "loads"):
        out = []
        for it in setup.get(key, []):
            e = {k: v for k, v in it.items() if k not in ("ids", "points")}
            e = json.loads(json.dumps(e, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
            ids = it.get("ids")
            if ids is not None and model is not None and len(ids):
                ids = np.asarray(ids, np.int64)
                e["ids"] = ids.tolist()
                e["points"] = model.nodes[ids].tolist()
            elif it.get("points") is not None:
                e["points"] = np.asarray(it["points"]).tolist()
            out.append(e)
        d[key] = out
    return d


def save_setup(setup, path, model=None, save_mesh=True):
    """Setup als JSON speichern; optional das Netz daneben als .mesh.npz."""
    path = Path(path)
    mesh_file = None
    if model is not None and save_mesh:
        mp = path.with_name(path.stem + ".mesh.npz")
        np.savez_compressed(mp, nodes=model.nodes, tets=model.tets)
        mesh_file = mp.name
    d = setup_to_dict(setup, model, path.parent, mesh_file)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1, ensure_ascii=False)
    return mesh_file


def load_setup(path):
    """Rueckgabe: (setup, base_dir). Listen werden zu NumPy-Arrays."""
    path = Path(path)
    try:
        raw = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise FemError(f"Setup-Datei nicht lesbar: {e}")
    setup = _merge(default_setup(), {k: v for k, v in raw.items() if k not in ("supports", "loads")})
    for key in ("supports", "loads"):
        items = []
        for it in raw.get(key, []):
            it = dict(it)
            if it.get("ids") is not None:
                it["ids"] = np.asarray(it["ids"], np.int64)
            if it.get("points") is not None:
                it["points"] = np.asarray(it["points"], float).reshape(-1, 3)
            if key == "supports":
                it["mask"] = np.asarray(it.get("mask", [True, True, True]), bool)
                it["values"] = np.asarray(it.get("values", [0.0, 0.0, 0.0]), float)
            else:
                if "force" not in it:
                    raise FemError("Last ohne 'force'.")
                it["force"] = np.asarray(it["force"], float)
            items.append(it)
        setup[key] = items
    return setup, path.parent


def build_model(setup, base_dir="."):
    """Netz laden (mesh.file) oder neu vernetzen. Rueckgabe: (MeshModel, aus_datei)."""
    m, g = setup["mesh"], setup["geometry"]
    if m.get("file"):
        p = resolve_path(base_dir, m["file"])
        if p.exists():
            d = np.load(p)
            model = MeshModel(d["nodes"], d["tets"])
            log(f"Netz aus {p.name} geladen: " + mesh_stats(model.nodes, model.tets))
            return model, True
        log(f"Netzdatei {p} nicht gefunden - vernetze neu.")
    gp = resolve_path(base_dir, g["file"]) if g.get("file") else None
    if gp is not None and not gp.exists():
        raise FemError(f"Geometriedatei nicht gefunden: {gp}")
    scale = UNITS.get(g.get("unit", "mm"), 1.0)
    log("Vernetze " + (gp.name if gp else "Demo-Balken") + " ...")
    nodes, tets, _ = mesh_body(str(gp) if gp else None, m.get("h"), m.get("order", 2), scale,
                               m.get("hmin"), m.get("curv", 12.0), m.get("refinements") or None)
    model = MeshModel(nodes, tets)
    log(mesh_stats(nodes, tets))
    return model, False


def resolve_ids(model, item):
    """Knoten-IDs einer Randbedingung im aktuellen Netz bestimmen."""
    sel = item.get("select")
    if sel:
        P = model.nodes[model.surf_nodes]
        tol = float(sel.get("tol", 1e-6 * model.diag + 1e-9))
        mask = np.ones(len(P), bool)
        if "plane" in sel:
            pl = sel["plane"]
            ax = {"x": 0, "y": 1, "z": 2}[str(pl["axis"]).lower()]
            mask &= np.abs(P[:, ax] - float(pl["value"])) <= tol
        if "box" in sel:
            lo, hi = np.asarray(sel["box"]["lo"], float), np.asarray(sel["box"]["hi"], float)
            mask &= np.all((P >= lo - tol) & (P <= hi + tol), axis=1)
        if "sphere" in sel:
            c, r = np.asarray(sel["sphere"]["center"], float), float(sel["sphere"]["radius"])
            mask &= np.linalg.norm(P - c, axis=1) <= r + tol
        ids = model.surf_nodes[mask]
        if len(ids) == 0:
            raise FemError(f"Auswahl {sel} trifft keine Oberflaechenknoten.")
        return ids
    ids, pts = item.get("ids"), item.get("points")
    if ids is not None and len(ids) and ids.max() < model.nn:
        if pts is None or np.allclose(model.nodes[ids], pts, atol=1e-6 * model.diag + 1e-9):
            return np.asarray(ids, np.int64)
    if pts is None or len(pts) == 0:
        raise FemError("Randbedingung ohne gueltige Knotenauswahl (ids/points/select).")
    ids, dmax = model.map_points(pts)
    log(f"Hinweis: Knoten ueber Koordinaten zugeordnet (max. Abstand {dmax:.3g} mm). "
        "Fuer exakte Zuordnung das Netz mitspeichern.")
    return ids


def build_bc(model, setup):
    """Rueckgabe: cmask (n*3 bool), cval (n*3), F (n*3)."""
    n = 3 * model.nn
    cmask, cval, F = np.zeros(n, bool), np.zeros(n), np.zeros(n)
    for s in setup["supports"]:
        ids = resolve_ids(model, s)
        for k in range(3):
            if s["mask"][k]:
                cmask[3 * ids + k] = True
                cval[3 * ids + k] = s["values"][k]
    for ld in setup["loads"]:
        ids = resolve_ids(model, ld)
        w = model.load_weights(ids)
        for k in range(3):
            np.add.at(F, 3 * ids + k, w * ld["force"][k])
    return cmask, cval, F


def default_rmin(model):
    cen = model.nodes[model.tets[:, :4]].mean(1)
    return 2.5 * float(cKDTree(cen).query(cen, k=2)[0][:, 1].mean())


class IterationSaver:
    """Schreibt waehrend der Topologieoptimierung Zwischenergebnisse in einen Ordner:
    verlauf.csv (jede Iteration) und iter_NNNN.vtk (Spannung, Verschiebung, Dichte)."""

    def __init__(self, folder, every=5, vtk=True):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every, self.vtk = max(1, int(every)), bool(vtk)
        self.csv = self.dir / "verlauf.csv"
        self.csv.write_text("iteration;nachgiebigkeit;volumenanteil;aenderung\n")

    def wanted(self, it):
        return it == 1 or it % self.every == 0

    def row(self, it, comp, vfrac, change):
        with open(self.csv, "a") as fh:
            fh.write(f"{it};{comp:.6g};{vfrac:.6f};{change:.6f}\n")

    def save_vtk(self, name, model, fem, u, rho):
        u3 = np.asarray(u).reshape(-1, 3)
        pd = {"Verschiebung": u3, "u_mag_mm": np.linalg.norm(u3, axis=1),
              "von_Mises_MPa": fem.vonmises_nodes(np.asarray(u).ravel())}
        write_vtk(self.dir / f"iter_{name}.vtk", model.nodes, model.tets, pd, {"Dichte": rho})


def run_analysis(setup, model, fem, callback=None, base_dir="."):
    """Statische Rechnung oder Topologieoptimierung gemaess Setup."""
    cmask, cval, F = build_bc(model, setup)
    if not cmask.any():
        raise FemError("Kein Lager definiert.")
    kind = setup["analysis"]["type"]
    if kind == "static":
        if not np.any(F) and not np.any(cval):
            raise FemError("Keine Kraft bzw. vorgegebene Verschiebung definiert.")
        log("Statische Rechnung ...")
        u = fem.static(cmask, cval, F)
        return dict(u=u.reshape(-1, 3), vm=fem.vonmises_nodes(u), rho=None, cmask=cmask, F=F)
    if kind != "topopt":
        raise FemError(f"Unbekannter Analysetyp: {kind}")
    if np.any(cval != 0):
        raise FemError("Topologieoptimierung unterstuetzt keine vorgegebenen Verschiebungen.")
    if not np.any(F):
        raise FemError("Topologieoptimierung braucht mindestens eine Kraft.")
    prm = setup["analysis"]["topopt"]
    vf, p, maxit = float(prm["volfrac"]), float(prm["penal"]), int(prm["maxit"])
    rmin = float(prm["rmin"]) if prm.get("rmin") else default_rmin(model)
    frozen = cmask.reshape(-1, 3).any(1) | (F.reshape(-1, 3) != 0).any(1)
    passive = frozen[model.tets].any(1)
    if fem.vol[passive].sum() > vf * fem.vol.sum():
        log("WARNUNG: Fest eingefrorene Bereiche an Lager/Last ueberschreiten den Zielvolumenanteil.")
    log(f"Topologieoptimierung: Volumenanteil {vf}, Filterradius {rmin:.3g} mm, p={p}, "
        f"max. {maxit} Iterationen, {int(passive.sum())} eingefrorene Elemente")
    saver = None
    if prm.get("save_dir"):
        saver = IterationSaver(resolve_path(base_dir, prm["save_dir"]), prm.get("save_every", 5),
                               prm.get("save_vtk", True))
        log(f"Zwischenergebnisse werden gespeichert in {saver.dir} (alle {saver.every} Iterationen)")

    def cb(it, comp, vfrac, change, rt_new, u, rt_used):
        if saver is not None:
            saver.row(it, comp, vfrac, change)
            if saver.vtk and saver.wanted(it):
                saver.save_vtk(f"{it:04d}", model, fem, u, rt_used)
        return callback(it, comp, vfrac, change, rt_new, u, rt_used) if callback else False

    rt = fem.topopt(cmask, F, vf, rmin, p, maxit, passive, cb)
    u = fem.solve_system(fem.assemble(fem.stiffness_factor(rt, p)), F, cmask, np.zeros(3 * model.nn))
    if saver is not None and saver.vtk:
        saver.save_vtk("final", model, fem, u, rt)
    return dict(u=u.reshape(-1, 3), vm=fem.vonmises_nodes(u), rho=rt, cmask=cmask, F=F)


def mass_g(setup, fem, rho_el=None):
    d = setup["material"].get("rho")
    if not d:
        return None
    v = fem.vol.sum() if rho_el is None else (fem.vol * rho_el).sum()
    return float(v * d * 1e-3)                  # mm^3 * g/cm^3 -> g


def summarize(setup, model, fem, res):
    mat = setup["material"]
    vmax = float(res["vm"][model.corner_ids].max())
    umax = float(np.linalg.norm(res["u"], axis=1).max())
    lines = [f"max. von Mises: {vmax:.2f} MPa | max. Verschiebung: {umax:.4f} mm"]
    if res["rho"] is None and mat.get("rp") and vmax > 0:
        lines.append(f"Sicherheit Rp/von Mises: {mat['rp'] / vmax:.2f} (Spitzenwert)")
    m = mass_g(setup, fem, res["rho"])
    if m is not None:
        lines.append(f"Masse: {m:.1f} g" + (f" (vorher {mass_g(setup, fem):.1f} g)"
                                              if res["rho"] is not None else ""))
    return lines


# ---- VTK-Ausgabe ----------------------------------------------------------
def _vtk_name(s):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in str(s))


def write_vtk(path, nodes, tets, point_data=None, cell_data=None, title="FEM Ergebnis"):
    """Schreibt ein VTK-Legacy-Binaerfile (Unstructured Grid, ohne VTK-Bibliothek).
    Tet4 -> Zelltyp 10, Tet10 -> Zelltyp 24. ParaView/VisIt lesen die Datei direkt."""
    nodes = np.asarray(nodes, float)
    m, nen = tets.shape
    ctype = 10 if nen == 4 else 24
    be = lambda a, dt: np.ascontiguousarray(a, dtype=dt).tobytes()

    def block(fh, name, arr):
        arr = np.asarray(arr, float)
        name = _vtk_name(name)
        if arr.ndim == 1:
            fh.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n".encode())
        else:
            fh.write(f"VECTORS {name} double\n".encode())
        fh.write(be(arr, ">f8") + b"\n")

    with open(path, "wb") as fh:
        fh.write(b"# vtk DataFile Version 3.0\n" + title.encode("ascii", "replace")[:200]
                 + b"\nBINARY\nDATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(nodes)} double\n".encode() + be(nodes, ">f8") + b"\n")
        cells = np.hstack([np.full((m, 1), nen), tets])
        fh.write(f"CELLS {m} {m * (nen + 1)}\n".encode() + be(cells, ">i4") + b"\n")
        fh.write(f"CELL_TYPES {m}\n".encode() + be(np.full(m, ctype), ">i4") + b"\n")
        if point_data:
            fh.write(f"POINT_DATA {len(nodes)}\n".encode())
            for k, v in point_data.items():
                block(fh, k, v)
        if cell_data:
            fh.write(f"CELL_DATA {m}\n".encode())
            for k, v in cell_data.items():
                block(fh, k, v)


def result_fields(model, res):
    pd = {"Verschiebung": res["u"],
          "u_mag_mm": np.linalg.norm(res["u"], axis=1),
          "von_Mises_MPa": res["vm"],
          "Lager_gesperrte_FG": res["cmask"].reshape(-1, 3).sum(1).astype(float),
          "Last_N": res["F"].reshape(-1, 3)}
    cd = {"Dichte": res["rho"]} if res["rho"] is not None else None
    return pd, cd


def write_results(path, model, res):
    """Ergebnis schreiben. .vtk = Legacy (ohne Abhaengigkeiten), .vtu = XML (benoetigt pyvista)."""
    path = Path(path)
    pd, cd = result_fields(model, res)
    if path.suffix.lower() == ".vtu":
        try:
            import pyvista as pv
        except ImportError:
            path = path.with_suffix(".vtk")
            log("pyvista fehlt - schreibe stattdessen " + path.name)
        else:
            m, nen = model.tets.shape
            cells = np.hstack([np.full((m, 1), nen), model.tets]).ravel()
            g = pv.UnstructuredGrid(cells, np.full(m, 10 if nen == 4 else 24, np.uint8), model.nodes)
            for k, v in pd.items():
                g.point_data[_vtk_name(k)] = v
            for k, v in (cd or {}).items():
                g.cell_data[_vtk_name(k)] = v
            g.save(str(path))
            return path
    write_vtk(path, model.nodes, model.tets, pd, cd)
    return path


# ---- Komplettlauf fuer Server ---------------------------------------------
def run_setup_file(json_path, out=None):
    setup, base = load_setup(json_path)
    model, _ = build_model(setup, base)
    mat = setup["material"]
    fem = FEM(model.nodes, model.tets, float(mat["E"]), float(mat["nu"]))
    last = [0.0]

    def cb(it, comp, vfrac, change, rt, u=None, rt_used=None):
        if time.time() - last[0] > 5 or it == 1:
            log(f"  TO Iteration {it}: Nachgiebigkeit {comp:.4g}, Volumen {vfrac:.3f}, "
                f"Aenderung {change:.3f}")
            last[0] = time.time()
        return False

    t0 = time.time()
    res = run_analysis(setup, model, fem, cb, base)
    for line in summarize(setup, model, fem, res):
        log(line)
    if out is None:
        out = setup["output"].get("file") or (Path(json_path).stem + "_ergebnis.vtk")
    out = resolve_path(base, out)
    written = write_results(out, model, res)
    log(f"Ergebnis geschrieben: {written} (Rechenzeit {time.time() - t0:.1f} s)")
    return written


def cli(argv=None):
    ap = argparse.ArgumentParser(prog="fem_core.py",
                                 description="FEM ohne Grafikfenster: Setup-Datei -> VTK-Ergebnis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Setup rechnen und Ergebnis als VTK schreiben")
    r.add_argument("setup")
    r.add_argument("--out", help="Ausgabedatei (.vtk oder .vtu)")
    c = sub.add_parser("check", help="Netz und Randbedingungen pruefen, ohne zu rechnen")
    c.add_argument("setup")
    args = ap.parse_args(argv)
    set_logger(lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True))
    try:
        if args.cmd == "run":
            out = Path(args.out).resolve() if args.out else None
            run_setup_file(args.setup, out)
        else:
            setup, base = load_setup(args.setup)
            model, _ = build_model(setup, base)
            cmask, cval, F = build_bc(model, setup)
            log(f"{len(setup['supports'])} Lager ({int(cmask.sum())} gesperrte Freiheitsgrade), "
                f"{len(setup['loads'])} Lasten (Summe {F.reshape(-1, 3).sum(0).round(3)} N)")
            log("Setup in Ordnung." if cmask.any() else "WARNUNG: kein Lager definiert.")
    except FemError as e:
        print(f"FEHLER: {e}", file=sys.stderr)
        return 1
    except Exception as e:                       # unerwartet: Traceback zeigen
        import traceback
        traceback.print_exc()
        print(f"UNERWARTETER FEHLER: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(cli())