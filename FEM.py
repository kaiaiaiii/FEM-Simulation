"""
Interaktive lineare FEM-Berechnung eines beliebigen Koerpers.

Ablauf
------
1. Geometrie laden (STEP/IGES/STL/MSH) -> Tetraedernetz (gmsh).
   Ohne Datei wird ein Demo-Balken (100 x 20 x 20 mm) erzeugt.
2. Interaktives Fenster (PyVista): Flaechen/Knoten auf der Oberflaeche
   markieren und als Lager oder Kraft zuweisen.
3. Loesen (linear-elastisch, Tet4) und Ergebnis anzeigen
   (von-Mises-Vergleichsspannung auf der verformten Geometrie).

Installation
------------
    pip install numpy scipy pyvista gmsh

Aufruf
------
    python fem_interaktiv.py                      # Demo-Balken
    python fem_interaktiv.py bauteil.step --h 2   # STEP, Elementgroesse 2 mm
    python fem_interaktiv.py teil.stl --E 70000 --nu 0.33

Einheiten: mm, N, MPa (E in MPa).

Bedienung im Fenster
--------------------
    R + Rechteck ziehen : sichtbare Oberflaeche markieren (gelb)
    1 : Markierung als FESTES LAGER (alle 3 Freiheitsgrade gesperrt)
    2 : Markierung als KRAFT (Dialog: Fx, Fy, Fz in N, Gesamtkraft
        wird flaechengewichtet auf die Knoten verteilt)
    3 : letzte Zuweisung zuruecknehmen
    4 : alles loeschen
    5 : BERECHNEN
    0 : zurueck zur Lager-/Lastansicht
    Maus: drehen / Mausrad zoomen / Shift+Linksklick schieben
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyvista as pv


# --------------------------------------------------------------------------
# Vernetzung
# --------------------------------------------------------------------------
def mesh_body(path, h=None):
    """Erzeugt ein lineares Tetraedernetz. Rueckgabe: (nodes[n,3], tets[m,4])."""
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
            h = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) / 15
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h / 4)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.model.mesh.generate(3)

        ntags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords).reshape(-1, 3)
        lookup = np.full(int(ntags.max()) + 1, -1, dtype=np.int64)
        lookup[ntags.astype(np.int64)] = np.arange(len(ntags))

        etypes, _, enodes = gmsh.model.mesh.getElements(3)
        if 4 not in etypes:
            sys.exit("Kein Tetraedernetz erzeugt (ist der Koerper geschlossen?).")
        tets = np.asarray(enodes[list(etypes).index(4)], dtype=np.int64).reshape(-1, 4)
        tets = lookup[tets]
    finally:
        gmsh.finalize()

    # nur verwendete Knoten behalten
    used = np.unique(tets)
    remap = np.full(len(coords), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return coords[used], remap[tets]


# --------------------------------------------------------------------------
# FEM (lineare Tetraeder, isotrop linear-elastisch)
# --------------------------------------------------------------------------
def elasticity_matrix(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[range(3), range(3)] += 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def element_B(nodes, tets):
    """B-Matrizen (m,6,12) und Volumina (m,) aller Tetraeder."""
    X = nodes[tets]                                    # (m,4,3)
    M = np.concatenate([np.ones((len(tets), 4, 1)), X], axis=2)  # (m,4,4)
    vol = np.abs(np.linalg.det(M)) / 6.0
    grad = np.linalg.inv(M)[:, 1:, :]                  # (m,3,4): dN_a/dx_j
    B = np.zeros((len(tets), 6, 12))
    for a in range(4):
        gx, gy, gz = grad[:, 0, a], grad[:, 1, a], grad[:, 2, a]
        c = 3 * a
        B[:, 0, c] = gx
        B[:, 1, c + 1] = gy
        B[:, 2, c + 2] = gz
        B[:, 3, c], B[:, 3, c + 1] = gy, gx
        B[:, 4, c + 1], B[:, 4, c + 2] = gz, gy
        B[:, 5, c], B[:, 5, c + 2] = gz, gx
    return B, vol


def assemble(nodes, tets, E, nu):
    D = elasticity_matrix(E, nu)
    B, vol = element_B(nodes, tets)
    Ke = np.einsum("mji,jk,mkl->mil", B, D, B) * vol[:, None, None]
    dofs = (3 * tets[:, :, None] + np.arange(3)).reshape(len(tets), 12)
    rows = np.repeat(dofs, 12, axis=1).ravel()
    cols = np.tile(dofs, (1, 12)).ravel()
    n = 3 * len(nodes)
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    return K, D, B, vol


def solve(nodes, tets, E, nu, fixed_nodes, F):
    K, D, B, vol = assemble(nodes, tets, E, nu)
    n = 3 * len(nodes)
    fixed = np.zeros(n, dtype=bool)
    for nd in fixed_nodes:
        fixed[3 * nd:3 * nd + 3] = True
    free = np.flatnonzero(~fixed)
    Kff = K[free][:, free].tocsc()
    u = np.zeros(n)
    if len(free) < 80000:
        u[free] = spla.spsolve(Kff, F[free])
    else:
        diag = Kff.diagonal()
        M = spla.LinearOperator(Kff.shape, lambda x: x / diag)
        sol, info = spla.cg(Kff, F[free], M=M, rtol=1e-8, maxiter=20000)
        if info != 0:
            print("WARNUNG: CG nicht konvergiert (info=%d)" % info)
        u[free] = sol
    if not np.all(np.isfinite(u)):
        raise RuntimeError("Loesung ungueltig - Lagerung ausreichend? "
                           "(Starrkoerperbewegung moeglich)")

    R = K @ u - F                                   # Reaktionskraefte
    print("Summe Kraefte (Last):      ", F.reshape(-1, 3).sum(0).round(3))
    print("Summe Reaktionskraefte:    ", R[fixed].reshape(-1, 3).sum(0).round(3))

    ue = u[(3 * tets[:, :, None] + np.arange(3)).reshape(len(tets), 12)]
    sig = np.einsum("ij,mjk,mk->mi", D, B, ue)      # (m,6) sxx syy szz txy tyz tzx
    sx, sy, sz, txy, tyz, tzx = sig.T
    vm_e = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                   + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))
    # Knotenwerte: volumengewichtetes Mittel
    w = np.zeros(len(nodes))
    s = np.zeros(len(nodes))
    for a in range(4):
        np.add.at(w, tets[:, a], vol)
        np.add.at(s, tets[:, a], vol * vm_e)
    return u.reshape(-1, 3), s / w, vm_e


# --------------------------------------------------------------------------
# Interaktive Oberflaeche
# --------------------------------------------------------------------------
def ask_force():
    prompt = "Gesamtkraft Fx, Fy, Fz in N (z.B. 0, 0, -1000):"
    s = None
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        s = simpledialog.askstring("Kraft vorgeben", prompt, parent=root)
        root.destroy()
    except Exception:
        s = input(prompt + " ")
    if not s:
        return None
    try:
        v = [float(t) for t in s.replace(";", ",").replace(" ", ",").split(",") if t]
        return np.array(v) if len(v) == 3 else None
    except ValueError:
        return None


class App:
    def __init__(self, nodes, tets, E, nu):
        self.nodes, self.tets, self.E, self.nu = nodes, tets, E, nu
        self.grid = pv.UnstructuredGrid(
            np.hstack([np.full((len(tets), 1), 4), tets]).ravel(),
            np.full(len(tets), pv.CellType.TETRA),
            nodes,
        )
        self.grid.point_data["node_id"] = np.arange(len(nodes))
        self.surf = self.grid.extract_surface().triangulate()
        self.surf_ids = np.asarray(self.surf.point_data["node_id"])
        self.diag = float(np.linalg.norm(np.ptp(nodes, axis=0)))

        # Knotenflaechen (je 1/3 der Dreiecksflaeche) fuer Lastverteilung
        area = self.surf.compute_cell_sizes(length=False, volume=False)["Area"]
        tri = self.surf.faces.reshape(-1, 4)[:, 1:]
        a_pt = np.zeros(self.surf.n_points)
        for k in range(3):
            np.add.at(a_pt, tri[:, k], area / 3)
        self.node_area = np.zeros(len(nodes))
        self.node_area[self.surf_ids] = a_pt

        self.selection = np.array([], dtype=int)   # globale Knoten-IDs
        self.history = []                          # [("fix", ids) | ("force", ids, F)]
        self.mode = "setup"

        self.pl = pv.Plotter(title="Interaktive FEM")
        self.pl.add_mesh(self.surf, color="lightsteelblue", show_edges=True,
                         edge_color="gray", line_width=0.5, name="body")
        self.pl.add_axes()
        self.pl.enable_cell_picking(callback=self.on_pick, through=False,
                                    show=False, show_message=False,
                                    font_size=10)
        for key, fn in (("1", self.assign_fix), ("2", self.assign_force),
                        ("3", self.undo), ("4", self.clear),
                        ("5", self.run), ("0", self.back)):
            self.pl.add_key_event(key, fn)
        self.help = ("R + Rechteck: markieren | 1 Lager | 2 Kraft | 3 Undo | "
                     "4 Alles loeschen | 5 Berechnen")
        self.pl.add_text(self.help, position="upper_left", font_size=8,
                         name="help")
        self.status("Bereit. Zuerst mit R und Rechteck Oberflaeche markieren.")

    # ---- Hilfsfunktionen ----
    def status(self, msg):
        self.pl.add_text(msg, position="lower_left", font_size=9, name="status")
        self.pl.render()
        print(msg)

    def on_pick(self, picked):
        if self.mode != "setup" or picked is None or picked.n_points == 0:
            return
        self.selection = np.unique(np.asarray(picked.point_data["node_id"]))
        self.pl.add_points(self.nodes[self.selection], color="yellow",
                           point_size=8, render_points_as_spheres=True,
                           name="sel", reset_camera=False)
        self.status(f"{len(self.selection)} Knoten markiert -> 1 Lager / 2 Kraft")

    def redraw(self):
        fixed = [h[1] for h in self.history if h[0] == "fix"]
        if fixed:
            ids = np.unique(np.concatenate(fixed))
            self.pl.add_points(self.nodes[ids], color="red", point_size=9,
                               render_points_as_spheres=True, name="fixed",
                               reset_camera=False)
        else:
            self.pl.remove_actor("fixed")
        forces = [h for h in self.history if h[0] == "force"]
        if forces:
            centers = np.array([self.nodes[h[1]].mean(0) for h in forces])
            dirs = np.array([h[2] / np.linalg.norm(h[2]) for h in forces])
            self.pl.add_arrows(centers - dirs * 0.15 * self.diag,
                               dirs * 0.15 * self.diag, color="green",
                               name="forces", mag=1.0)
        else:
            self.pl.remove_actor("forces")
        self.pl.remove_actor("sel")
        self.selection = np.array([], dtype=int)
        self.pl.render()

    # ---- Aktionen ----
    def need_selection(self):
        if self.mode != "setup":
            return False
        if len(self.selection) == 0:
            self.status("Nichts markiert - erst mit R + Rechteck auswaehlen.")
            return False
        return True

    def assign_fix(self):
        if not self.need_selection():
            return
        self.history.append(("fix", self.selection.copy()))
        n = len(self.selection)
        self.redraw()
        self.status(f"Festes Lager auf {n} Knoten gesetzt.")

    def assign_force(self):
        if not self.need_selection():
            return
        F = ask_force()
        if F is None or not np.any(F):
            self.status("Keine gueltige Kraft eingegeben.")
            return
        self.history.append(("force", self.selection.copy(), F))
        n = len(self.selection)
        self.redraw()
        self.status(f"Kraft {F} N auf {n} Knoten verteilt.")

    def undo(self):
        if self.mode == "setup" and self.history:
            self.history.pop()
            self.redraw()
            self.status("Letzte Zuweisung entfernt.")

    def clear(self):
        if self.mode == "setup":
            self.history.clear()
            self.redraw()
            self.status("Alle Zuweisungen geloescht.")

    def build_loads(self):
        F = np.zeros(3 * len(self.nodes))
        for h in self.history:
            if h[0] != "force":
                continue
            ids, f = h[1], h[2]
            w = self.node_area[ids]
            w = w / w.sum() if w.sum() > 0 else np.full(len(ids), 1 / len(ids))
            for k in range(3):
                np.add.at(F, 3 * ids + k, w * f[k])
        return F

    def run(self):
        if self.mode != "setup":
            return
        fixed = [h[1] for h in self.history if h[0] == "fix"]
        if not fixed:
            return self.status("Kein Lager definiert.")
        if not any(h[0] == "force" for h in self.history):
            return self.status("Keine Kraft definiert.")
        fixed_ids = np.unique(np.concatenate(fixed))
        self.status("Berechne ...")
        try:
            u, vm_n, vm_e = solve(self.nodes, self.tets, self.E, self.nu,
                                  fixed_ids, self.build_loads())
        except Exception as e:
            return self.status(f"Fehler: {e}")

        umag = np.linalg.norm(u, axis=1)
        scale = 0.08 * self.diag / umag.max() if umag.max() > 0 else 1.0
        g = self.grid.copy()
        g.point_data["Verschiebung"] = u
        g.point_data["u_mag [mm]"] = umag
        g.point_data["von Mises [MPa]"] = vm_n
        warped = g.warp_by_vector("Verschiebung", factor=scale)
        for name in ("body", "fixed", "forces", "sel"):
            self.pl.remove_actor(name)
        self.pl.add_mesh(warped, scalars="von Mises [MPa]", cmap="turbo",
                         show_edges=False, name="result",
                         scalar_bar_args={"title": "von Mises [MPa]"})
        self.pl.add_mesh(self.grid.extract_surface(), style="wireframe",
                         color="gray", opacity=0.25, name="undeformed")
        self.mode = "result"
        self.status(f"max. von Mises: {vm_n.max():.2f} MPa | "
                    f"max. Verschiebung: {umag.max():.4f} mm | "
                    f"Darstellung {scale:.1f}x ueberhoeht | 0 = zurueck")

    def back(self):
        if self.mode != "result":
            return
        self.pl.remove_actor("result")
        self.pl.remove_actor("undeformed")
        try:
            self.pl.remove_scalar_bar()
        except Exception:
            pass
        self.pl.add_mesh(self.surf, color="lightsteelblue", show_edges=True,
                         edge_color="gray", line_width=0.5, name="body")
        self.mode = "setup"
        self.redraw()
        self.status("Zurueck in der Lager-/Lastansicht.")

    def show(self):
        self.pl.show()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Interaktive lineare FEM")
    ap.add_argument("datei", nargs="?", help="STEP/IGES/STL/MSH (leer = Demo-Balken)")
    ap.add_argument("--h", type=float, default=None, help="Elementgroesse in mm")
    ap.add_argument("--E", type=float, default=210000.0, help="E-Modul in MPa")
    ap.add_argument("--nu", type=float, default=0.3, help="Querkontraktionszahl")
    args = ap.parse_args()

    print("Vernetze ...")
    nodes, tets = mesh_body(args.datei, args.h)
    print(f"{len(nodes)} Knoten, {len(tets)} Tetraeder")
    App(nodes, tets, args.E, args.nu).show()


if __name__ == "__main__":
    main()