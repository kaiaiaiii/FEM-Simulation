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
    python fem_interaktiv_v2.py                       # Demo-Balken
    python fem_interaktiv_v2.py bauteil.step --h 3    # STEP/IGES/STL/MSH
    python fem_interaktiv_v2.py teil.stl --order 1 --E 70000 --nu 0.33
    python fem_interaktiv_v2.py --konsole             # Eingaben in der Konsole
                                                      # statt tkinter-Dialogen
Einheiten: mm, N, MPa.

Bedienung im 3D-Fenster
-----------------------
  R + Rechteck ziehen : sichtbare Oberflaeche markieren (gelb)
  1  Lager definieren (Dialog) | 2  Kraft (Dialog) | 3  Undo | 4  Alles loeschen
  5  Berechnen (statisch)
  6  Topologieoptimierung (Dialog); waehrend des Laufs: 6 = Abbruch
  7  Schnittmodus wechseln: keine -> Clip-Ebene -> Schnitt-Ebene
  8  Feld wechseln: von Mises -> Verschiebung -> Dichte (nach Optimierung)
  9  Export (ergebnis.vtu, bei Optimierung zusaetzlich optimiert.stl)
  0  zurueck zur Lager-/Lastansicht
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import pyvista as pv

USE_TK = True
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


def mesh_body(path, h=None, order=2):
    """Tetraedernetz mit gmsh. Rueckgabe: nodes (n,3), tets (m,4|10)."""
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        if path is None:
            gmsh.model.add("demo")
            gmsh.model.occ.addBox(0, 0, 0, 100, 20, 20)
            gmsh.model.occ.synchronize()
        else:
            ext = Path(path).suffix.lower()
            if ext in (".step", ".stp", ".iges", ".igs", ".brep"):
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
            else:
                sys.exit(f"Dateityp {ext} nicht unterstuetzt.")

        if h is None:
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(-1, -1)
            h = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) / (
                12 if order == 2 else 15)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h / 4)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.model.mesh.generate(3)

        ntags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords).reshape(-1, 3)
        lookup = np.full(int(ntags.max()) + 1, -1, dtype=np.int64)
        lookup[ntags.astype(np.int64)] = np.arange(len(ntags))

        etype, npe = (4, 4) if order == 1 else (11, 10)
        etypes, _, enodes = gmsh.model.mesh.getElements(3)
        if etype not in etypes:
            sys.exit("Kein Tetraedernetz erzeugt (ist der Koerper geschlossen?).")
        tets = np.asarray(enodes[list(etypes).index(etype)], dtype=np.int64)
        tets = lookup[tets.reshape(-1, npe)]
    finally:
        gmsh.finalize()

    used = np.unique(tets)
    remap = np.full(len(coords), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    nodes, tets = coords[used], remap[tets]
    if order == 2:
        tets = to_vtk_order(nodes, tets)
    return nodes, tets


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
        self.D = elasticity_matrix(E, nu)
        self.edofs = (3 * tets[:, :, None] + np.arange(3)).reshape(self.m, self.ne)
        self.rows = self.cols = None
        self._element_stiffness()

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


# ==========================================================================
# Interaktive Oberflaeche
# ==========================================================================
class App:
    CUTS = ["keine", "Clip-Ebene", "Schnitt-Ebene"]
    BAR_TITLES = ["von Mises [MPa]", "u [mm]", "Dichte"]

    def __init__(self, nodes, tets, E, nu):
        self.nodes, self.tets = nodes, tets
        self.nn = len(nodes)
        self.fem = FEM(nodes, tets, E, nu)
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

        self.pl = pv.Plotter(title="Interaktive FEM")
        self.pl.add_mesh(self.surf, color="lightsteelblue", show_edges=True,
                         edge_color="gray", line_width=0.5, name="body")
        self.pl.add_axes()
        self.pl.enable_cell_picking(callback=self.on_pick, through=False,
                                    show=False, show_message=False)
        for key, fn in (("1", self.assign_support), ("2", self.assign_force),
                        ("3", self.undo), ("4", self.clear), ("5", self.run),
                        ("6", self.run_topopt), ("7", self.next_cut),
                        ("8", self.next_field), ("9", self.export),
                        ("0", self.back)):
            self.pl.add_key_event(key, fn)
        self.pl.add_text("R+Rechteck: markieren | 1 Lager | 2 Kraft | 3 Undo | 4 Loeschen | "
                         "5 Rechnen | 6 Topologieopt.", position="upper_left",
                         font_size=8, name="help")
        self.status(f"{self.nn} Knoten, {len(tets)} Tet{'10' if self.order == 2 else '4'}. "
                    "Mit R und Rechteck Oberflaeche markieren.")

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
            extra = (f"max von Mises {self.res['vm'][self.corner_ids].max():.2f} MPa, "
                     f"max u {np.linalg.norm(self.res['u'], axis=1).max():.4f} mm, ")
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
    ap.add_argument("datei", nargs="?", help="STEP/IGES/STL/MSH (leer = Demo-Balken)")
    ap.add_argument("--h", type=float, default=None, help="Elementgroesse in mm")
    ap.add_argument("--order", type=int, choices=(1, 2), default=2,
                    help="1 = Tet4, 2 = Tet10 (Standard)")
    ap.add_argument("--E", type=float, default=210000.0, help="E-Modul in MPa")
    ap.add_argument("--nu", type=float, default=0.3, help="Querkontraktionszahl")
    ap.add_argument("--konsole", action="store_true",
                    help="Eingaben in der Konsole statt in tkinter-Dialogen")
    args = ap.parse_args()
    USE_TK = not args.konsole

    print("Vernetze ...")
    nodes, tets = mesh_body(args.datei, args.h, args.order)
    print(f"{len(nodes)} Knoten, {len(tets)} Tetraeder (Ordnung {args.order})")
    App(nodes, tets, args.E, args.nu).show()


if __name__ == "__main__":
    main()