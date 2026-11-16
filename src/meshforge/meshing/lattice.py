from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyvista as pv

from meshforge.geometry import parallel_transport_frames


@dataclass
class WallLattice:
    """(s, θ, r) node lattice of the wall region, shared by both backends.

    ``nodes[i, j, l]`` = station i, azimuth j (periodic), layer l
    (0 = inner wall, n_layers = outer wall). Radial spacing follows
    ``layer_ratio`` grading; per-ray thickness is curvature-capped and
    floored (hardening series).
    """

    nodes: np.ndarray  # (S, T, L+1, 3)
    inner_radii: np.ndarray  # (S, T) ray-measured lumen radii
    thickness: float  # nominal
    thickness_field: np.ndarray  # (S, T) applied per-ray thickness
    n_capped: int
    n_floored: int
    notes: list[str]
    closed_s: bool = False  # periodic in s (closed centerline loop, V9)
    # D87: how the stations were laid out — "centreline" (rays square to a
    # centreline, this module) or "planes" (meshing/planes.py: sections of
    # planes parallel to two flat caps); ``fold_number`` = max r·κ·cosα of
    # the centreline lattice's own rays (>= 1: the ray fan crosses itself);
    # ``info`` carries the plane sweep's own counts.
    sweep: str = "centreline"
    fold_number: float | None = None
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        s, t, l1, _ = self.nodes.shape
        return s, t, l1 - 1

    def node_id(self, i: int, j: int, l: int) -> int:
        s, t, layers = self.shape
        return (i * t + (j % t)) * (layers + 1) + l

    def points(self) -> np.ndarray:
        return self.nodes.reshape(-1, 3)

    def boundary_shell(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Watertight triangulated boundary of the wall solid.

        Returns (points, triangles, patch_labels) with labels
        0=inner, 1=outer, 2=inlet, 3=outlet — a volume mesher can mesh this
        shell directly (no reparametrization) and reuse the labels to
        classify its own boundary faces.
        """
        s, t, layers = self.shape
        pts = self.points()
        tris: list[tuple[int, int, int]] = []
        labels: list[int] = []

        def quad(a: int, b: int, c: int, d: int, label: int) -> None:
            tris.append((a, b, c))
            tris.append((a, c, d))
            labels.append(label)
            labels.append(label)

        n_seg = s if self.closed_s else s - 1
        for i in range(n_seg):
            i1 = (i + 1) % s
            for j in range(t):
                # Inner wall (normal pointing into the lumen = outward from solid).
                quad(
                    self.node_id(i, j, 0),
                    self.node_id(i1, j, 0),
                    self.node_id(i1, j + 1, 0),
                    self.node_id(i, j + 1, 0),
                    0,
                )
                # Outer wall.
                quad(
                    self.node_id(i, j, layers),
                    self.node_id(i, j + 1, layers),
                    self.node_id(i1, j + 1, layers),
                    self.node_id(i1, j, layers),
                    1,
                )
        if not self.closed_s:  # closed loops have no end annuli (V9)
            for j in range(t):
                for l in range(layers):
                    # Inlet annulus (station 0, outward normal = −tangent).
                    quad(
                        self.node_id(0, j, l),
                        self.node_id(0, j + 1, l),
                        self.node_id(0, j + 1, l + 1),
                        self.node_id(0, j, l + 1),
                        2,
                    )
                    # Outlet annulus (last station, outward normal = +tangent).
                    quad(
                        self.node_id(s - 1, j, l),
                        self.node_id(s - 1, j, l + 1),
                        self.node_id(s - 1, j + 1, l + 1),
                        self.node_id(s - 1, j + 1, l),
                        3,
                    )
        return pts, np.asarray(tris, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def build_wall_lattice(
    surface: pv.PolyData,
    thickness,  # float or ThicknessSpec (measured/field walls)
    *,
    centerline=None,
    n_theta: int = 32,
    n_axial: int = 60,
    n_layers: int = 4,
    layer_ratio: float = 1.0,
    curvature_cap: float = 0.5,
    floor_frac: float = 0.2,
) -> WallLattice:
    """Ray-cast the lumen radius per (station, azimuth) and extrude radially.

    A lattice node at radial distance ρ on the concave side of a bend has
    local axial metric 1 − ρκ·cosα (α between the ray and the curvature
    direction); it folds when that reaches 0, so per-ray thickness is capped
    at ρ_max = ``curvature_cap``/(κ·cosα) minus the lumen radius, and floored
    at ``floor_frac``·t so no ray collapses to a zero slab. ``layer_ratio``
    grades radial layer thicknesses geometrically (outer/inner).
    """
    from meshforge.geometry.thickness import ThicknessSpec

    spec = thickness if isinstance(thickness, ThicknessSpec) else None
    if spec is None and thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    if n_theta < 8 or n_axial < 2 or n_layers < 1:
        raise ValueError(
            f"lattice too coarse: n_theta={n_theta}, n_axial={n_axial}, n_layers={n_layers}"
        )
    if layer_ratio <= 0:
        raise ValueError(f"layer_ratio must be positive, got {layer_ratio}")
    if not 0.0 < floor_frac <= 1.0:
        raise ValueError(f"floor_frac must be in (0,1], got {floor_frac}")
    if centerline is None:
        # V40: reuse a centerline embedded in the surface (e.g. the interior
        # contour centerline of an imported contour set) before falling
        # back to recomputing the skeleton centerline.
        from meshforge.geometry import centerline_from_surface, compute_centerline

        centerline = centerline_from_surface(surface)
        if centerline is None:
            centerline = compute_centerline(surface)
    closed_s = bool(centerline.closed)
    if closed_s:
        # Closed centerlines duplicate the wrap sample (points[-1]==points[0]);
        # pick n_axial distinct stations over one period.
        idx = np.unique(
            np.floor(
                np.linspace(0, len(centerline.points) - 1, n_axial, endpoint=False)
            ).astype(int)
        )
    else:
        idx = np.unique(
            np.linspace(0, len(centerline.points) - 1, n_axial).round().astype(int)
        )
    stations = centerline.points[idx]
    tangents = centerline.tangents[idx]
    normals, binormals = parallel_transport_frames(tangents, closed=closed_s)

    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Rays only need to reach the INNER surface (lumen wall), not the outer.
    ray_len = 5.0 * float(centerline.radii.max())

    s_count = len(stations)
    inner_radii = np.empty((s_count, n_theta))
    for i in range(s_count):
        for j in range(n_theta):
            d = cos_t[j] * normals[i] + sin_t[j] * binormals[i]
            hits, _ = surface.ray_trace(stations[i], stations[i] + d * ray_len, first_point=True)
            if len(hits) == 0:
                raise ValueError(
                    f"lumen ray missed the surface at station {i}, θ index {j}; "
                    "centerline may exit the surface or ray length is too short"
                )
            inner_radii[i, j] = np.linalg.norm(hits - stations[i])

    # Per-ray thickness: scalar, or a sampled field/measured outer (H-C).
    d_all = cos_t[None, :, None] * normals[:, None, :] + sin_t[None, :, None] * binormals[:, None, :]
    spec_notes: list[str] = []
    if spec is None:
        nominal = float(thickness)
        t_field = np.full((s_count, n_theta), nominal)
    else:
        hit_pts = stations[:, None, :] + inner_radii[:, :, None] * d_all
        t_field = spec.sample(
            hit_pts.reshape(-1, 3), normals=d_all.reshape(-1, 3)
        ).reshape(s_count, n_theta)
        nominal = float(np.median(t_field))
        spec_notes.extend(spec.notes)
        spec_notes.append(
            f"thickness field ({spec.kind}): sampled "
            f"{float(t_field.min()):.4g}..{float(t_field.max()):.4g}, median {nominal:.4g}"
        )
    if closed_s:
        # Periodic curvature: wrapped central differences at uniform spacing.
        ds_loop = centerline.length / len(stations)
        kvec = (np.roll(tangents, -1, axis=0) - np.roll(tangents, 1, axis=0)) / (
            2.0 * ds_loop
        )
    else:
        ds = np.gradient(centerline.arc_lengths[idx])
        kvec = np.gradient(tangents, axis=0) / ds[:, None]
    kappa = np.linalg.norm(kvec, axis=1)
    n_capped = 0
    fold_number = 0.0
    for i in range(s_count):
        if kappa[i] < 1e-12:
            continue
        n_c = kvec[i] / kappa[i]  # points toward the center of curvature
        cos_a = d_all[i] @ n_c
        risky = cos_a > 1e-9
        if not risky.any():
            continue
        fold_number = max(
            fold_number, float((inner_radii[i, risky] * kappa[i] * cos_a[risky]).max())
        )
        rho_max = curvature_cap / (kappa[i] * cos_a[risky])
        allowed = np.maximum(rho_max - inner_radii[i, risky], 0.0)
        capped_here = t_field[i, risky] > allowed
        n_capped += int(capped_here.sum())
        t_field[i, risky] = np.minimum(t_field[i, risky], allowed)

    floor = floor_frac * nominal
    floored = t_field < floor
    n_floored = int(floored.sum())
    t_field = np.maximum(t_field, floor)

    notes: list[str] = list(spec_notes)
    if n_capped:
        # Reduce arrays to scalars for the message — never format an ndarray.
        notes.append(
            f"curvature cap limited {n_capped} rays "
            f"(min applied thickness {float(t_field.min()):.4g})"
        )
    if n_floored:
        notes.append(
            f"floor raised {n_floored} rays to {floor:.4g}; "
            "wall is thinner than nominal on the concave side of the bend"
        )

    # Geometric layer grading: layer k thickness ∝ layer_ratio^k.
    k = np.arange(n_layers + 1, dtype=float)
    if abs(layer_ratio - 1.0) < 1e-12:
        fracs = k / n_layers
    else:
        fracs = (layer_ratio**k - 1.0) / (layer_ratio**n_layers - 1.0)

    radial = inner_radii[:, :, None] + t_field[:, :, None] * fracs[None, None, :]
    nodes = stations[:, None, None, :] + radial[:, :, :, None] * d_all[:, :, None, :]

    if closed_s:
        notes.append(
            f"closed-loop lattice: {len(stations)} periodic stations, "
            "holonomy-corrected frames, no inlet/outlet"
        )
    return WallLattice(
        nodes=nodes,
        inner_radii=inner_radii,
        thickness=nominal,
        thickness_field=t_field,
        n_capped=n_capped,
        n_floored=n_floored,
        notes=notes,
        closed_s=closed_s,
        fold_number=fold_number,
    )
