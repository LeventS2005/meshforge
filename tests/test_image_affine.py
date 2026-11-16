"""D83 mask-ingest fixes: affine-mapped marching cubes (90-degree permutation,
oblique ~30 degrees, LPS-style negative directions, a single reflection),
the binary-mask front end (largest component, bounding-box crop under
MAX_VOXELS, blur, empty/thin refusals) and NIfTI length units."""
from __future__ import annotations

import numpy as np
import pytest

from meshforge.io.image import (
    check_affine,
    load_volume_affine,
    mask_to_surface,
    nifti_mm_scale,
    volume_to_surface,
)

nib = pytest.importorskip("nibabel")


def _rot(axis: str, deg: float) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


_AFFINES = {
    # 90-degree axis permutation (i -> -y, j -> +x), det > 0
    "permutation_90": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]]) * 0.7,
    # oblique: 30 deg about z then 20 deg about x, anisotropic voxels
    "oblique_30": _rot("z", 30) @ _rot("x", 20) @ np.diag([0.6, 0.8, 1.1]),
    # RAS view of an LPS volume with positive LPS directions: x, y negative
    "lps_like": np.diag([-0.7, -0.7, 1.0]),
    # a single reflection: det < 0 reverses the triangle winding
    "reflect_x": np.diag([-0.5, 0.6, 0.9]),
}


def _affine(lin: np.ndarray, t=(10.0, -20.0, 30.0)) -> np.ndarray:
    a = np.eye(4)
    a[:3, :3] = lin
    a[:3, 3] = t
    return a


def _sphere_sdf(n=40, center=(18.0, 21.0, 19.5), r=8.0) -> np.ndarray:
    idx = np.indices((n, n, n)).astype(float)
    c = np.asarray(center)[:, None, None, None]
    return np.sqrt(((idx - c) ** 2).sum(axis=0)) - r


@pytest.mark.parametrize("name", sorted(_AFFINES))
def test_affine_mapped_surface_lands_at_the_analytic_world_position(name):
    lin = _AFFINES[name]
    a = _affine(lin)
    center = np.array([18.0, 21.0, 19.5])
    surface, prov = volume_to_surface(_sphere_sdf(center=center), 0.0, affine=a)
    expected_center = lin @ center + a[:3, 3]
    pts = np.asarray(surface.points)
    np.testing.assert_allclose(pts.mean(axis=0), expected_center, atol=0.02)
    # every vertex sits on the mapped (ellipsoidal) sphere: back in index
    # space its distance to the centre is r within marching-cubes error
    back = (pts - a[:3, 3]) @ np.linalg.inv(lin).T
    rad = np.linalg.norm(back - center, axis=1)
    assert np.abs(rad - 8.0).max() < 0.15
    assert prov["watertight"] is True
    # outward: positive signed volume == mapped analytic volume (MC -1..2 %)
    truth = 4.0 / 3.0 * np.pi * 8.0**3 * abs(np.linalg.det(lin))
    assert prov["enclosed_volume"] == pytest.approx(truth, rel=0.025)
    assert float(surface.volume) == pytest.approx(prov["enclosed_volume"], rel=1e-6)
    normals = surface.compute_normals(
        cell_normals=False, point_normals=True, auto_orient_normals=False
    ).point_data["Normals"]
    outward = np.einsum("ij,ij->i", pts - expected_center, np.asarray(normals))
    assert (outward > 0).mean() > 0.999
    np.testing.assert_allclose(prov["affine"], a)
    np.testing.assert_allclose(prov["spacing"], np.linalg.norm(lin, axis=0))


def test_affine_refusals():
    vol = _sphere_sdf(n=16, center=(8, 8, 8), r=4.0)
    sing = np.eye(4)
    sing[:3, :3] = [[1, 2, 0], [2, 4, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="singular"):
        volume_to_surface(vol, 0.0, affine=sing)
    zero = np.diag([1.0, 0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="zero length"):
        volume_to_surface(vol, 0.0, affine=zero)
    bad_row = np.eye(4)
    bad_row[3, 0] = 1.0
    with pytest.raises(ValueError, match="bottom row"):
        check_affine(bad_row)
    nan = np.eye(4)
    nan[0, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        check_affine(nan)
    with pytest.raises(ValueError, match="4x4"):
        check_affine(np.eye(3))
    with pytest.raises(ValueError, match="either affine or spacing"):
        volume_to_surface(vol, 0.0, spacing=(2, 2, 2), affine=np.eye(4))


# ---------------------------------------------------------------------------
# mask_to_surface
# ---------------------------------------------------------------------------
def _tube_mask(shape=(60, 50, 30), center=(30, 25), r=8, axis_len=None):
    ii, jj = np.indices(shape[:2])
    disk = (ii - center[0]) ** 2 + (jj - center[1]) ** 2 <= r**2
    m = np.zeros(shape, dtype=bool)
    m[:, :, : axis_len or shape[2]] = disk[:, :, None]
    return m


def test_mask_tube_cut_by_the_field_of_view_closes_with_flat_caps():
    m = _tube_mask()
    m[5, 5, 5] = True  # a stray voxel: a second component
    a = _affine(np.diag([0.8, 0.8, 1.5]), t=(1.0, 2.0, 3.0))
    surface, prov = mask_to_surface(m, a, sigma=1.0)
    assert prov["watertight"] is True and surface.n_open_edges == 0
    assert prov["components"] == 2 and prov["dropped_components"] == 1
    assert prov["dropped_voxels"] == 1 and prov["kept_voxels"] == m.sum() - 1
    assert any("dropped 1 other component" in n for n in prov["notes"])
    assert any("field-of-view boundary" in n for n in prov["notes"])
    # caps half a voxel past the first/last slice centres
    zmin, zmax = surface.bounds[4], surface.bounds[5]
    assert zmin == pytest.approx(3.0 - 0.75, abs=0.05)
    assert zmax == pytest.approx(3.0 + 29 * 1.5 + 0.75, abs=0.05)
    # axis at the disk centre, world frame
    pts = np.asarray(surface.points)
    mid = pts[np.abs(pts[:, 2] - 25.0) < 5.0]
    np.testing.assert_allclose(mid[:, :2].mean(axis=0), [1 + 30 * 0.8, 2 + 25 * 0.8], atol=0.05)
    radius = np.linalg.norm(mid[:, :2] - [1 + 30 * 0.8, 2 + 25 * 0.8], axis=1)
    assert radius.mean() == pytest.approx(8.0 * 0.8, rel=0.04)
    assert float(surface.volume) > 0


def test_mask_crop_keeps_a_big_grid_small_object_under_max_voxels(monkeypatch):
    """A full-CT-size mask (here 200^3 = 8 M voxels against a ceiling lowered
    to 200 k) still extracts: only the object's bounding box is meshed."""
    from meshforge.geometry import centerline as clmod

    monkeypatch.setattr(clmod, "MAX_VOXELS", 200_000, raising=True)
    m = np.zeros((200, 200, 200), dtype=bool)
    ii, jj, kk = np.indices((30, 30, 30))
    m[100:130, 40:70, 150:180] = (ii - 15) ** 2 + (jj - 15) ** 2 + (kk - 15) ** 2 <= 12**2
    a = _affine(np.diag([-0.5, -0.5, 0.5]))  # LPS-like flips on top
    surface, prov = mask_to_surface(m, a, sigma=1.0)
    assert prov["watertight"] is True
    assert int(np.prod(np.array(prov["crop_hi"]) - prov["crop_lo"] + 1 + 2 * prov["margin"])) < 200_000
    assert float(surface.volume) == pytest.approx(4 / 3 * np.pi * 6.0**3, rel=0.06)
    # over the ceiling: block-averaged (loud note), still the same sphere ...
    small, sprov = mask_to_surface(m, a, sigma=1.0, max_voxels=10_000)
    assert sprov["block_average"] != [1, 1, 1]
    assert any("block-averaged" in n for n in sprov["notes"])
    assert float(small.volume) == pytest.approx(4 / 3 * np.pi * 6.0**3, rel=0.12)
    # ... and refused only when even 8x averaging cannot fit it
    with pytest.raises(ValueError, match="voxel ceiling"):
        mask_to_surface(m, a, sigma=1.0, max_voxels=100)


def test_mask_refusals_and_smoothing_off():
    a = np.eye(4)
    with pytest.raises(ValueError, match="mask is empty"):
        mask_to_surface(np.zeros((10, 10, 10), bool), a)
    single = np.zeros((10, 10, 10), bool)
    single[5, 5, 5] = True
    with pytest.raises(ValueError, match="too thin"):
        mask_to_surface(single, a, sigma=1.0)
    with pytest.raises(ValueError, match="sigma"):
        mask_to_surface(single, a, sigma=-1.0)
    with pytest.raises(ValueError, match="3-D"):
        mask_to_surface(np.ones((4, 4), bool), a)
    surface, prov = mask_to_surface(single, a, sigma=0.0)  # no blur: one voxel
    assert prov["watertight"] is True and prov["sigma"] == 0.0


# ---------------------------------------------------------------------------
# units + the NIfTI loaders
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "unit, factor, noted",
    [("mm", 1.0, False), ("meter", 1000.0, True), ("micron", 1e-3, True), ("unknown", 1.0, True)],
)
def test_nifti_unit_scale(unit, factor, noted):
    hdr = nib.Nifti1Header()
    hdr.set_xyzt_units(unit, "sec")
    got, note = nifti_mm_scale(hdr)
    assert got == factor
    assert (note is not None) is noted
    if unit == "unknown":
        assert "ASSUMING millimetres" in note


def _save_nifti(path, vol, affine, unit="mm"):
    img = nib.Nifti1Image(vol.astype(np.float32), affine)
    img.header.set_xyzt_units(unit, "sec")
    nib.save(img, str(path))
    return path


def test_load_volume_affine_units_and_orientation(tmp_path):
    vol = _sphere_sdf(n=24, center=(12, 12, 12), r=6.0)
    a = _affine(np.diag([-0.5, -0.5, 0.5]), t=(4.0, 5.0, 6.0))
    _, got, info = load_volume_affine(_save_nifti(tmp_path / "m.nii", vol, a, "meter"))
    np.testing.assert_allclose(got[:3, :], a[:3, :] * 1000.0)
    assert info["length_unit"] == "mm" and any("converted" in n for n in info["notes"])
    _, got, info = load_volume_affine(_save_nifti(tmp_path / "u.nii", vol, a, "unknown"))
    np.testing.assert_allclose(got, a, atol=1e-6)
    assert any("ASSUMING" in n for n in info["notes"])
    _, got, info = load_volume_affine(_save_nifti(tmp_path / "mm.nii.gz", vol, a))
    assert info == {"length_unit": "mm", "notes": []}
    npy = tmp_path / "v.npy"
    np.save(npy, vol)
    _, got, info = load_volume_affine(npy)
    assert info["length_unit"] is None and np.array_equal(got, np.eye(4))


def test_oversized_mask_is_block_averaged_not_refused():
    # A fine-grid aorta (0.48 mm in-plane, bifurcation to arch) overflows the
    # 40 M geometry ceiling; the object is block-averaged to fit, with a note,
    # and must land at the same place with the same volume.
    import numpy as np
    import pytest

    from meshforge.io.image import mask_to_surface

    sp = np.array([0.5, 0.5, 1.0])
    shape = (80, 80, 60)
    i, j, kk = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    # LPS-like flipped in-plane axes, as in a real CT
    affine = np.diag([-sp[0], -sp[1], sp[2], 1.0])
    affine[:3, 3] = (30.0, 20.0, -100.0)
    world = np.stack([i, j, kk], -1) @ affine[:3, :3].T + affine[:3, 3]
    cx, cy, r = 10.0, 0.0, 12.0  # world-space cylinder axis, radius (mm)
    mask = ((world[..., 0] - cx) ** 2 + (world[..., 1] - cy) ** 2 <= r**2) & (
        (kk >= 10) & (kk < 50)
    )
    full, _ = mask_to_surface(mask, affine, sigma=1.0)
    coarse, prov = mask_to_surface(mask, affine, sigma=1.0, max_voxels=60_000)
    assert prov["block_average"] != [1, 1, 1]
    assert any("block-averaged" in n for n in prov["notes"])
    assert coarse.n_open_edges == 0
    expected = np.pi * r**2 * 40.0  # 40 slices of 1 mm
    assert prov["enclosed_volume"] == pytest.approx(expected, rel=0.05)
    assert prov["enclosed_volume"] == pytest.approx(
        abs(full.volume), rel=0.05
    )
    # same place: the block-centre index mapping is exact, not off by half a block
    assert np.allclose(coarse.center, full.center, atol=0.3)
    assert coarse.center[0] == pytest.approx(cx, abs=0.3)


def test_merged_slivers_are_deleted_not_demoted():
    # A triangle whose two vertices coincide must vanish on clean, not become
    # a line cell: on a real CTA aorta such leftovers made a watertight
    # surface fail mesh_ready (all-triangles).
    import numpy as np
    import pyvista as pv

    from meshforge.io.image import _clean_triangles

    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=float
    )  # point 4 duplicates point 3
    faces = np.array([3, 0, 1, 2, 3, 1, 3, 2, 3, 1, 3, 4])  # last one: a sliver
    out = _clean_triangles(pv.PolyData(pts, faces=faces))
    assert out.n_lines == 0 and out.n_verts == 0
    assert out.n_cells == 2
