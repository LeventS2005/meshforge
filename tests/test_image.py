"""V18 image ingest: numpy core with analytic sphere truths, physical-
coordinate correctness under anisotropic spacing, loud branches, loader
install-hint errors (deps absent HERE — full loader tests skip unless
pydicom/nibabel exist), session + CLI e2e."""
from __future__ import annotations


import numpy as np
import pytest

from meshforge.io.image import (
    dicom_grid,
    load_volume,
    volume_input_sha256,
    volume_to_surface,
)

try:
    import nibabel  # noqa: F401

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
try:
    import pydicom  # noqa: F401

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def _sphere_volume(
    n: int = 48, r: float = 8.0, center=None, spacing=(1.0, 1.0, 1.0)
) -> np.ndarray:
    """Signed distance to a sphere, sampled on an n^3 grid (negative inside)."""
    center = np.array([n // 2] * 3, dtype=float) if center is None else np.asarray(center, float)
    idx = np.indices((n, n, n)).astype(float)
    phys = idx * np.asarray(spacing, float)[:, None, None, None]
    c_phys = center * np.asarray(spacing, float)
    d = np.sqrt(((phys - c_phys[:, None, None, None]) ** 2).sum(axis=0))
    return d - r


def test_sphere_volume_and_orientation():
    vol = _sphere_volume()
    surface, prov = volume_to_surface(vol, 0.0)
    assert prov["watertight"] is True and surface.n_open_edges == 0
    truth = 4.0 / 3.0 * np.pi * 8.0**3
    assert prov["enclosed_volume"] == pytest.approx(truth, rel=0.02)
    # Outward orientation: pyvista's signed volume must be positive too.
    assert float(surface.volume) == pytest.approx(prov["enclosed_volume"], rel=1e-6)


def test_physical_coordinates_spacing_and_origin():
    """Off-center sphere + anisotropic spacing + origin: the surface centroid
    must land at the right PHYSICAL position (catches index-order bugs)."""
    spacing = (0.5, 1.0, 2.0)
    center_idx = (30.0, 22.0, 12.0)
    vol = _sphere_volume(n=48, r=6.0, center=center_idx, spacing=spacing)
    origin = (5.0, -3.0, 10.0)
    surface, prov = volume_to_surface(vol, 0.0, spacing=spacing, origin=origin)
    expected = np.asarray(origin) + np.asarray(center_idx) * np.asarray(spacing)
    got = np.asarray(surface.points).mean(axis=0)
    np.testing.assert_allclose(got, expected, atol=0.05)
    # 4% here, not 2%: spacing 2.0 on z leaves only ~6 voxels per radius, so
    # the marching-cubes faceting deficit is larger on that axis (measured
    # -2.7%) — coarse-axis discretization, not an orientation bug (the
    # centroid assert above would catch that).
    assert prov["enclosed_volume"] == pytest.approx(4 / 3 * np.pi * 6.0**3, rel=0.04)


def test_leaky_surface_is_a_note_not_an_error():
    vol = _sphere_volume(n=24, r=14.0)  # sphere clips the grid bounds
    surface, prov = volume_to_surface(vol, 0.0)
    assert prov["watertight"] is False
    assert prov["enclosed_volume"] is None
    assert any("not watertight" in n for n in prov["notes"])
    assert surface.n_cells > 0


def test_core_loud_branches():
    vol = _sphere_volume(n=16, r=5.0)
    with pytest.raises(ValueError, match="3-D"):
        volume_to_surface(vol[0], 0.0)
    with pytest.raises(ValueError, match="outside the data range"):
        volume_to_surface(vol, 1e9)
    with pytest.raises(ValueError, match="spacing must be positive"):
        volume_to_surface(vol, 0.0, spacing=(1.0, 0.0, 1.0))
    bad = np.array(vol)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        volume_to_surface(bad, 0.0)


def test_load_volume_npy_and_unsupported(tmp_path):
    vol = _sphere_volume(n=16, r=5.0)
    npy = tmp_path / "vol.npy"
    np.save(npy, vol)
    array, spacing, origin = load_volume(npy)
    np.testing.assert_array_equal(array, vol)
    assert spacing == (1.0, 1.0, 1.0)
    with pytest.raises(FileNotFoundError):
        load_volume(tmp_path / "missing.npy")
    weird = tmp_path / "x.raw"
    weird.write_bytes(b"12")
    with pytest.raises(ValueError, match="unsupported volume input"):
        load_volume(weird)


@pytest.mark.skipif(HAS_NIBABEL, reason="install-hint branch needs nibabel ABSENT")
def test_nifti_install_hint_when_missing(tmp_path):
    nii = tmp_path / "x.nii"
    nii.write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match="pip install nibabel"):
        load_volume(nii)


@pytest.mark.skipif(HAS_PYDICOM, reason="install-hint branch needs pydicom ABSENT")
def test_dicom_install_hint_when_missing(tmp_path):
    d = tmp_path / "series"
    d.mkdir()
    with pytest.raises(RuntimeError, match="pip install pydicom"):
        load_volume(d)


@pytest.mark.skipif(not HAS_NIBABEL, reason="nibabel not installed")
def test_nifti_roundtrip_and_rotated_affine_refused(tmp_path):
    import nibabel as nib

    vol = _sphere_volume(n=24, r=7.0)
    affine = np.diag([0.5, 1.0, 2.0, 1.0])
    affine[:3, 3] = [1.0, 2.0, 3.0]
    nii = tmp_path / "s.nii"
    nib.save(nib.Nifti1Image(vol, affine), str(nii))
    array, spacing, origin = load_volume(nii)
    np.testing.assert_allclose(array, vol, atol=1e-6)
    assert spacing == (0.5, 1.0, 2.0) and origin == (1.0, 2.0, 3.0)

    rot = np.eye(4)
    rot[0, 1] = 0.3
    bad = tmp_path / "rot.nii"
    nib.save(nib.Nifti1Image(vol, rot), str(bad))
    # D83: load_volume's (spacing, origin) API still cannot express a
    # rotated/sheared grid, so it refuses — but it now points at the
    # affine-aware loader, which import_image uses and which maps it exactly.
    with pytest.raises(ValueError, match="re-orient it first, or load it with load_volume_affine"):
        load_volume(bad)
    from meshforge.io.image import load_volume_affine

    _, affine, _ = load_volume_affine(bad)
    np.testing.assert_allclose(affine, rot)


def test_volume_input_sha256_file_and_dir(tmp_path):
    f = tmp_path / "a.npy"
    np.save(f, np.zeros((2, 2, 2)))
    h1 = volume_input_sha256(f)
    assert len(h1) == 64
    d = tmp_path / "series"
    d.mkdir()
    (d / "s1").write_bytes(b"one")
    (d / "s2").write_bytes(b"two")
    h2 = volume_input_sha256(d)
    (d / "s2").write_bytes(b"TWO")
    assert volume_input_sha256(d) != h2  # content change is detected
    with pytest.raises(FileNotFoundError):
        volume_input_sha256(tmp_path / "ghost")


# ---------------------------------------------------------------------------
# V39: single-file multi-frame DICOM (the real GE cardiac-CT layout)
# ---------------------------------------------------------------------------
def _write_multiframe_dcm(path, hu_zyx, *, spacing=(0.75, 0.75, 0.9),
                          origin=(-10.0, -20.0, -30.0),
                          iop=(1, 0, 0, 0, 1, 0), slice_spacing=True):
    """Write a classic multi-frame CT DICOM (NumberOfFrames, no functional
    groups) from an (frames, rows, cols) HU array, slope 1 / intercept -1024."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    nf, R, C = hu_zyx.shape
    stored = np.clip(hu_zyx + 1024, 0, 65535).astype(np.uint16)
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = CTImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = fm
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    ds.Modality = "CT"
    ds.Rows, ds.Columns = R, C
    ds.NumberOfFrames = nf
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [spacing[1], spacing[0]]  # (row=y, col=x)
    if slice_spacing:
        ds.SpacingBetweenSlices = spacing[2]
    ds.SliceThickness = 0.625
    ds.ImagePositionPatient = list(origin)
    ds.ImageOrientationPatient = list(iop)
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.PixelData = stored.tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return path


@pytest.mark.skipif(not HAS_PYDICOM, reason="pydicom not installed")
def test_multiframe_dicom_load_geometry_and_hu(tmp_path):
    # a ramp in HU so we can check frame order (+z) and the rescale intercept
    nf, R, C = 8, 6, 5
    hu = np.zeros((nf, R, C), dtype=np.int16)
    for k in range(nf):
        hu[k] = 100 * k - 500  # frame k -> known HU
    p = _write_multiframe_dcm(tmp_path / "vol.dcm", hu,
                              spacing=(0.75, 0.75, 0.9), origin=(-10.0, -20.0, -30.0))
    array, spacing, origin = load_volume(p)
    assert array.shape == (C, R, nf)          # (x, y, z)
    assert spacing == pytest.approx((0.75, 0.75, 0.9))   # z = SpacingBetweenSlices
    assert origin == pytest.approx((-10.0, -20.0, -30.0))
    # frame k (z index) carries HU 100k-500, and the intercept is applied
    assert array[0, 0, 0] == pytest.approx(-500.0)
    assert array[0, 0, nf - 1] == pytest.approx(100 * (nf - 1) - 500.0)
    # geometry-only grid matches, with no ceiling and nothing loaded
    dims, gsp, gorg = dicom_grid(p)
    assert dims == (C, R, nf) and gsp == pytest.approx(spacing) and gorg == pytest.approx(origin)


@pytest.mark.skipif(not HAS_PYDICOM, reason="pydicom not installed")
def test_multiframe_dicom_loud_branches(tmp_path):
    hu = np.zeros((4, 5, 5), dtype=np.int16)
    # oblique orientation refused (orientation safety)
    obl = _write_multiframe_dcm(tmp_path / "obl.dcm", hu,
                                iop=(0.9, 0.1, 0, 0, 1, 0))
    with pytest.raises(ValueError, match="non-axial"):
        load_volume(obl)
    # a single-frame .dcm is not a volume
    one = _write_multiframe_dcm(tmp_path / "one.dcm", hu[:1])
    with pytest.raises(ValueError, match="single-frame|frame"):
        load_volume(one)


@pytest.mark.skipif(not HAS_PYDICOM, reason="pydicom not installed")
def test_multiframe_dicom_falls_back_to_slice_thickness(tmp_path):
    hu = np.zeros((5, 4, 4), dtype=np.int16)
    p = _write_multiframe_dcm(tmp_path / "nosbs.dcm", hu, slice_spacing=False)
    _, spacing, _ = load_volume(p)
    assert spacing[2] == pytest.approx(0.625)  # SliceThickness when SBS absent
