from __future__ import annotations

import numpy as np
import pyvista as pv

from meshforge.meshing.base import BoundaryGroup, WallMesh
from meshforge.meshing.lattice import WallLattice, build_wall_lattice
from meshforge.quality.volume import volume_quality

# D87: station layouts. "centreline" (default, unchanged) casts rays square
# to a centreline; "planes" takes sections of planes parallel to two flat
# parallel end caps (meshing/planes.py); "auto" keeps the centreline sweep
# when its wall is clean and switches to the planes when it is not.
SWEEPS = ("centreline", "planes", "auto")


class SweptBackend:
    """Native structured hexahedra: stations × periodic θ ring × radial layers."""

    name = "swept"

    def mesh_wall(
        self,
        surface: pv.PolyData,
        thickness: float,
        *,
        centerline=None,
        n_theta: int = 32,
        n_axial: int = 60,
        n_layers: int = 4,
        layer_ratio: float = 1.0,
        curvature_cap: float = 0.5,
        floor_frac: float = 0.2,
        sweep: str = "centreline",
        side: str = "outward",
        max_tilt: float | None = None,
        section_smoothing: float = 0.0,
    ) -> WallMesh:
        from meshforge.meshing.planes import MAX_TILT_DEG, SIDES

        if sweep not in SWEEPS:
            raise ValueError(f"sweep must be one of {SWEEPS}, got {sweep!r}")
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        if side == "inward" and sweep == "centreline":
            raise ValueError(
                "an inward wall needs the plane sweep (sweep='planes' or 'auto'); the "
                "centreline sweep only grows the wall outward"
            )
        if sweep == "centreline" and (max_tilt is not None or section_smoothing):
            raise ValueError(
                "max_tilt and section_smoothing apply to the plane sweep only "
                "(sweep='planes' or 'auto')"
            )
        tilt = MAX_TILT_DEG if max_tilt is None else float(max_tilt)
        common = dict(
            n_theta=n_theta, n_axial=n_axial, n_layers=n_layers, layer_ratio=layer_ratio,
            curvature_cap=curvature_cap, floor_frac=floor_frac,
        )

        def centreline_mesh() -> WallMesh:
            lat = build_wall_lattice(surface, thickness, centerline=centerline, **common)
            return self._hex_mesh(lat, layer_ratio, curvature_cap)

        def plane_mesh(caps=None) -> WallMesh:
            from meshforge.meshing.planes import build_plane_lattice

            lat = build_plane_lattice(surface, thickness, centerline=centerline, side=side,
                                      max_tilt=tilt, section_smoothing=section_smoothing,
                                      caps=caps, **common)
            return self._hex_mesh(lat, layer_ratio, curvature_cap)

        if sweep == "centreline":
            return centreline_mesh()
        if sweep == "planes":
            return plane_mesh()
        unused = "" if max_tilt is None and not section_smoothing else (
            "; max_tilt/section_smoothing unused"
        )
        return self._auto(surface, centerline, side, centreline_mesh, plane_mesh, unused)

    @staticmethod
    def _auto(surface, centerline, side, centreline_mesh, plane_mesh, unused="") -> WallMesh:
        """sweep="auto": the centreline sweep when its wall is clean (0 inverted,
        0 capped, 0 floored); otherwise — or when it fails — the plane sweep
        if the surface ends in two flat caps parallel within
        ``planes.PARALLEL_TOL_DEG``. Every decision lands in the notes."""
        from meshforge.meshing.planes import PARALLEL_TOL_DEG, parallel_caps

        try:
            caps, caps_why = parallel_caps(surface, centerline=centerline), ""
        except ValueError as exc:
            caps, caps_why = None, str(exc)
        if side == "inward":
            if caps is None:
                raise ValueError(f"sweep=auto with side=inward needs the plane sweep: {caps_why}")
            mesh = plane_mesh(caps)
            mesh.notes.insert(0, "auto: plane sweep (an inward wall needs it)")
            return mesh
        try:
            cl, cl_err = centreline_mesh(), ""
        except ValueError as exc:
            cl, cl_err = None, str(exc)
        if cl is not None:
            lat = cl.lattice
            clean = (cl.quality.n_inverted == 0 and lat["n_capped"] == 0
                     and lat["n_floored"] == 0)
            why = (f"centreline sweep: inverted {cl.quality.n_inverted}, capped "
                   f"{lat['n_capped']}, floored {lat['n_floored']}, fold number "
                   f"{lat.get('fold_number', float('nan')):.3g}")
            if clean:
                cl.notes.insert(0, f"auto: centreline sweep (clean wall; {why}{unused})")
                return cl
            if caps is None:
                cl.notes.insert(0, f"auto: centreline sweep kept — the plane sweep does not "
                                   f"apply ({caps_why}); {why}{unused}")
                return cl
        else:
            why = f"centreline sweep failed: {cl_err}"
            if caps is None:
                raise ValueError(f"{why}; the plane sweep does not apply either: {caps_why}")
        try:
            mesh = plane_mesh(caps)
        except ValueError as exc:
            if cl is None:
                raise ValueError(f"{why}; the plane sweep refused: {exc}") from None
            cl.notes.insert(0, f"auto: centreline sweep kept — the plane sweep refused "
                               f"({exc}); {why}{unused}")
            return cl
        mesh.notes.insert(0, f"auto: plane sweep (caps parallel within "
                             f"{PARALLEL_TOL_DEG:g} deg; {why})")
        if cl is not None:
            mesh.lattice["auto_centreline"] = {
                "inverted": int(cl.quality.n_inverted),
                "capped": int(cl.lattice["n_capped"]),
                "floored": int(cl.lattice["n_floored"]),
                "fold_number": cl.lattice.get("fold_number"),
            }
        return mesh

    def _hex_mesh(self, lattice: WallLattice, layer_ratio: float, curvature_cap: float) -> WallMesh:
        s, t, layers = lattice.shape
        points = lattice.points()
        nid = lattice.node_id
        closed = lattice.closed_s
        n_seg = s if closed else s - 1  # periodic in s on closed loops (V9)

        cells = np.empty(n_seg * t * layers * 9, dtype=np.int64)
        k = 0
        for i in range(n_seg):
            i1 = (i + 1) % s
            for j in range(t):
                for l in range(layers):
                    # Bottom quad ordered so (p1−p0)×(p3−p0)·(p4−p0) > 0.
                    cells[k : k + 9] = [
                        8,
                        nid(i, j, l),
                        nid(i, j + 1, l),
                        nid(i1, j + 1, l),
                        nid(i1, j, l),
                        nid(i, j, l + 1),
                        nid(i, j + 1, l + 1),
                        nid(i1, j + 1, l + 1),
                        nid(i1, j, l + 1),
                    ]
                    k += 9
        celltypes = np.full(n_seg * t * layers, pv.CellType.HEXAHEDRON, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cells, celltypes, points)
        grid.cell_data["region"] = np.ones(grid.n_cells, dtype=np.int32)

        def cell_id(i: int, j: int, l: int) -> int:
            return (i * t + (j % t)) * layers + l

        def group(name: str, items: list[tuple[int, tuple[int, int, int, int]]]) -> BoundaryGroup:
            return BoundaryGroup(
                name=name,
                cell_ids=np.array([c for c, _ in items], dtype=np.int64),
                face_nodes=np.array([f for _, f in items], dtype=np.int64),
            )

        inner = [
            (
                cell_id(i, j, 0),
                (nid(i, j, 0), nid(i, j + 1, 0), nid((i + 1) % s, j + 1, 0), nid((i + 1) % s, j, 0)),
            )
            for i in range(n_seg)
            for j in range(t)
        ]
        outer = [
            (
                cell_id(i, j, layers - 1),
                (
                    nid(i, j, layers),
                    nid((i + 1) % s, j, layers),
                    nid((i + 1) % s, j + 1, layers),
                    nid(i, j + 1, layers),
                ),
            )
            for i in range(n_seg)
            for j in range(t)
        ]

        boundary = {"inner": group("inner", inner), "outer": group("outer", outer)}
        if not closed:  # closed loops have no end faces
            inlet = [
                (cell_id(0, j, l), (nid(0, j, l), nid(0, j + 1, l), nid(0, j + 1, l + 1), nid(0, j, l + 1)))
                for j in range(t)
                for l in range(layers)
            ]
            outlet = [
                (
                    cell_id(s - 2, j, l),
                    (nid(s - 1, j, l), nid(s - 1, j, l + 1), nid(s - 1, j + 1, l + 1), nid(s - 1, j + 1, l)),
                )
                for j in range(t)
                for l in range(layers)
            ]
            boundary["inlet"] = group("inlet", inlet)
            boundary["outlet"] = group("outlet", outlet)
        quality = volume_quality(grid)
        if lattice.sweep == "planes":
            notes = [
                f"plane lattice {s} stations × {t} samples × {layers} layers, "
                f"nominal thickness {lattice.thickness:.4g}, layer_ratio {layer_ratio:g}"
            ]
        else:
            notes = [
                f"lattice {s} stations × {t} azimuth × {layers} layers"
                f"{' (periodic s)' if closed else ''}, "
                f"nominal thickness {lattice.thickness:.4g}, layer_ratio {layer_ratio:g}"
            ]
        notes.extend(lattice.notes)
        notes.extend(quality.notes)
        counts = {
            "n_capped": int(lattice.n_capped),
            "n_floored": int(lattice.n_floored),
            "n_rays": int(lattice.thickness_field.size),
            "thickness_min": float(lattice.thickness_field.min()),
            "thickness_max": float(lattice.thickness_field.max()),
            "curvature_cap": float(curvature_cap),
            "sweep": lattice.sweep,
        }
        if lattice.fold_number is not None:
            counts["fold_number"] = round(float(lattice.fold_number), 4)
        counts.update(lattice.info)
        return WallMesh(
            grid=grid, boundary=boundary, backend=self.name, quality=quality, notes=notes,
            lattice=counts,
        )
