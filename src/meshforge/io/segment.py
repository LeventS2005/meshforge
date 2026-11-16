"""AI segmentation (M1, D83): CT -> TotalSegmentator aorta mask -> surface.

TotalSegmentator (Wasserthal et al., Apache-2.0; ``total`` task only, never
a licensed task) runs as a SUBPROCESS from its own conda env, so torch never
enters meshforge's environment. It is found like ccx
(``solve/runner.find_ccx``): ``MESHFORGE_TOTALSEG`` -> ``TotalSegmentator``
on PATH -> ``~/miniconda3/envs/totalseg/bin/TotalSegmentator``. An explicit
``MESHFORGE_TOTALSEG`` that does not point at an executable is an error, not
a silent fall-through to another copy.

The CT may be NIfTI or NRRD (``io/nrrd.py``; converted to a RAS NIfTI). The
input is staged — copied or converted — into a temp dir whose path has no
spaces, TotalSegmentator writes there, and the temp dir is deleted: the
user's CT is only ever read (rule 1, checksummed before and after) and
nothing lands in the repo. The mask becomes a surface through
``io/image.mask_to_surface`` (largest component, bounding-box crop,
Gaussian blur, level 0.5, affine-mapped marching cubes).

Output frame: RAS world coordinates in millimetres (the NIfTI convention
nibabel reports), so the surface overlays the reference segmentation of the
Zenodo AAA dataset without any transform.

This module never touches TotalSegmentator's own configuration (usage
statistics, licence); it only reports what that config says.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from meshforge.textguard import scrub

TOTALSEG_ENV_VAR = "MESHFORGE_TOTALSEG"
_TOTALSEG_FALLBACK = Path.home() / "miniconda3/envs/totalseg/bin/TotalSegmentator"
INSTALL_HINT = (
    "conda create -n totalseg --solver=libmamba python=3.11, then "
    "conda run -n totalseg pip install TotalSegmentator"
)
#: ``total``-task classes this step may request (no licensed task, ever).
#: The vessel classes are unioned into the one source mask; ``heart`` (the
#: whole heart — never the licensed ``heartchambers_highres`` task) is
#: CONTEXT only: its own surface, drawn as a ghost, never part of the vessel.
VESSEL_ROIS = ("aorta", "iliac_artery_left", "iliac_artery_right")
CONTEXT_ROIS = ("heart",)
ALLOWED_ROIS = VESSEL_ROIS + CONTEXT_ROIS
DEVICES = ("mps", "cpu")
#: Defaults from a benchmark on an Apple-silicon laptop.
DEFAULT_DEVICE = "mps"
DEFAULT_FAST = False
DEFAULT_TIMEOUT_S = 1800.0
#: CT ceiling, checked from the header before any data is read: a 512 x 512
#: x 2048-slice scan (1 GB of int16). A chest-to-pelvis CTA at 0.6 mm runs
#: 1000-1200 slices (AortaSeg-60 Anev_04: 1136), so 1024 refused real scans.
#: The geometry ceiling (MAX_VOXELS, 40 M) applies later, to the mask cropped
#: to its object.
MAX_CT_VOXELS = 512 * 512 * 2048
CT_SUFFIXES = (".nrrd", ".nii", ".nii.gz")
EMPTY_MASK_MESSAGE = (
    "TotalSegmentator returned an empty aorta mask — is the aorta in the "
    "field of view? The field of view may also be a tight crop without body "
    "context (e.g. a box around the aneurysm), which TotalSegmentator cannot "
    "read as anatomy; give it the full-field-of-view CT"
)
#: Below this the mask is a fragment, not an aorta (even 2 cm of a 1 cm-wide
#: aorta is ~6 mL). Seen for real in M0: an 84-voxel, 0.0046 mL speck inside
#: the thrombus of a tight crop around an aneurysm.
MIN_AORTA_ML = 1.0
_TAIL_LINES = 25
#: D85 (M2): contrast blood lumen = AI aorta mask AND CT >= this many HU.
DEFAULT_LUMEN_HU = 150.0
LUMEN_METHOD_NOTE = (
    "blood lumen = TotalSegmentator aorta mask AND CT >= {hu:g} HU, largest "
    "connected piece: a CT THRESHOLD inside the AI label, not an AI "
    "segmentation; bright calcification touching the lumen stays in it"
)


def find_totalseg() -> str:
    """Path of the TotalSegmentator executable, or a loud RuntimeError."""
    explicit = os.environ.get(TOTALSEG_ENV_VAR, "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        raise RuntimeError(
            f"{TOTALSEG_ENV_VAR}={explicit!r} is not an executable file — point "
            f"it at the TotalSegmentator executable or unset it. Install: "
            f"{INSTALL_HINT}"
        )
    for candidate in (shutil.which("TotalSegmentator"), str(_TOTALSEG_FALLBACK)):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        f"TotalSegmentator not found: set {TOTALSEG_ENV_VAR} to its executable, "
        f"put it on PATH, or install it in its own env: {INSTALL_HINT}"
    )


def totalseg_version(executable: str) -> str | None:
    """``TotalSegmentator --version`` (None when it cannot be read)."""
    try:
        cp = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=300,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = scrub((cp.stdout or cp.stderr or "").strip()).splitlines()
    if cp.returncode != 0 or not lines:
        return None
    return lines[-1].strip()[:64]


def _usage_stats_note() -> str | None:
    """Report (never change) TotalSegmentator's usage-statistics setting."""
    home = os.environ.get("TOTALSEG_HOME_DIR")
    cfg = (Path(home) if home else Path.home() / ".totalsegmentator") / "config.json"
    if not cfg.is_file():
        return (
            f"no TotalSegmentator config at {cfg}: its first run creates one "
            f"with usage statistics ON (send_usage_stats)"
        )
    try:
        on = bool(json.loads(cfg.read_text()).get("send_usage_stats", True))
    except (OSError, ValueError):
        return f"could not read {cfg} to check TotalSegmentator's usage statistics"
    if on:
        return f"TotalSegmentator usage statistics are ON in {cfg} (send_usage_stats)"
    return None


def _tail(text: str, n: int = _TAIL_LINES) -> str:
    # tqdm progress bars redraw with \r: keep only the final state of a line.
    lines = [ln.rsplit("\r", 1)[-1].rstrip() for ln in text.splitlines()]
    lines = [ln[:300] for ln in lines if ln.strip()]
    return "\n".join(lines[-n:]) if lines else "(no output)"


@contextmanager
def _staging_dir():
    """A fresh temp dir whose path contains no spaces (removed afterwards)."""
    path = None
    for base in (None, "/tmp"):
        try:
            candidate = tempfile.mkdtemp(prefix="meshforge_ts_", dir=base)
        except OSError:
            continue
        if " " not in candidate:
            path = Path(candidate)
            break
        shutil.rmtree(candidate, ignore_errors=True)
    if path is None:
        raise RuntimeError(
            "could not create a temporary directory without spaces in its path "
            "for TotalSegmentator — set TMPDIR to a space-free directory"
        )
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)  # its worker processes too
        else:  # pragma: no cover - not exercised on macOS
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


@dataclass
class TotalSegRun:
    executable: str
    version: str | None
    flags: list[str]
    device: str
    fast: bool
    rois: tuple[str, ...]
    wall_time_s: float
    log: str
    masks: dict[str, Path]
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "executable": self.executable,
            "version": self.version,
            "task": "total",
            "rois": list(self.rois),
            "fast": self.fast,
            "device": self.device,
            "flags": list(self.flags),
            "wall_time_s": round(self.wall_time_s, 2),
        }


def run_totalseg(
    ct_nifti: str | Path,
    out_dir: str | Path,
    *,
    device: str = DEFAULT_DEVICE,
    fast: bool = DEFAULT_FAST,
    timeout: float = DEFAULT_TIMEOUT_S,
    rois: tuple[str, ...] = ("aorta",),
    executable: str | None = None,
    version: str | None = None,
) -> TotalSegRun:
    """Run TotalSegmentator (``total`` task, ``--roi_subset``) on a NIfTI.

    ``ct_nifti`` and ``out_dir`` must be space-free paths (``segment_ct``
    stages them). Argument list, never a shell. The combined stdout/stderr
    log is captured; a non-zero exit, a timeout (the whole process group is
    killed) or a missing mask raises RuntimeError with a scrubbed log tail.
    ``version`` skips the ``--version`` probe when a previous call read it.
    """
    device, rois, timeout = _check_run_args(device, rois, timeout)
    ct_nifti, out_dir = Path(ct_nifti), Path(out_dir)
    for p in (ct_nifti, out_dir):
        if " " in str(p):
            raise ValueError(f"stage TotalSegmentator paths without spaces: {p}")
    exe = executable or find_totalseg()
    notes = [n for n in (_usage_stats_note(),) if n]
    if version is None:
        version = totalseg_version(exe)
    if version is None:
        notes.append("could not read the TotalSegmentator version (--version)")
    flags = ["--roi_subset", *rois, *(["--fast"] if fast else []), "--device", device]
    cmd = [exe, "-i", str(ct_nifti), "-o", str(out_dir), *flags]
    log_path = out_dir.parent / f"{out_dir.name}.log"
    t0 = time.monotonic()
    with log_path.open("wb") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=(os.name == "posix"),
        )
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill(proc)
            log = scrub(log_path.read_bytes().decode("utf-8", "replace"))
            raise RuntimeError(
                f"TotalSegmentator exceeded the {timeout:g} s timeout and was "
                f"stopped — raise the timeout or use fast mode. Last log lines:\n"
                f"{_tail(log)}"
            ) from None
        finally:
            _kill(proc)  # no-op once it exited; never leave workers behind
    wall = time.monotonic() - t0
    log = scrub(log_path.read_bytes().decode("utf-8", "replace"))
    if rc != 0:
        raise RuntimeError(
            f"TotalSegmentator failed (exit code {rc}). Last log lines:\n{_tail(log)}"
        )
    masks = {roi: out_dir / f"{roi}.nii.gz" for roi in rois}
    missing = [roi for roi, p in masks.items() if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"TotalSegmentator exited 0 but wrote no mask for {missing}. Last log "
            f"lines:\n{_tail(log)}"
        )
    return TotalSegRun(
        executable=exe,
        version=version,
        flags=flags,
        device=device,
        fast=bool(fast),
        rois=rois,
        wall_time_s=wall,
        log=log,
        masks=masks,
        notes=notes,
    )


def _check_run_args(device, rois, timeout):
    device = str(device).strip().lower()
    if device not in DEVICES:
        raise ValueError(f"device must be one of {'/'.join(DEVICES)}, got {device!r}")
    rois = tuple(str(r) for r in rois)
    bad = [r for r in rois if r not in ALLOWED_ROIS]
    if not rois or bad:
        raise ValueError(
            f"roi(s) {bad or '(none)'} not allowed; this step requests only "
            f"{', '.join(ALLOWED_ROIS)} from the 'total' task"
        )
    timeout = float(timeout)
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"timeout must be a positive number of seconds, got {timeout}")
    return device, rois, timeout


def _check_context_rois(rois: tuple[str, ...], context_rois) -> tuple[str, ...]:
    """The vessel classes stay vessel classes and the context classes stay
    context: ``heart`` may never be unioned into the source, and only the
    whole heart of the free ``total`` task may be asked for as context."""
    overlap = [r for r in rois if r in CONTEXT_ROIS]
    if overlap:
        raise ValueError(
            f"{overlap} is context (its own ghost surface), not part of the vessel "
            f"— pass it as context_rois, not rois"
        )
    context_rois = tuple(str(r) for r in (context_rois or ()))
    bad = [r for r in context_rois if r not in CONTEXT_ROIS]
    if bad or len(set(context_rois)) != len(context_rois):
        raise ValueError(
            f"context roi(s) {bad or '(duplicate)'} not allowed; context is only "
            f"{', '.join(CONTEXT_ROIS)} (the whole heart from the 'total' task — no "
            f"licensed task, no chambers)"
        )
    return context_rois


# ---------------------------------------------------------------------------
# CT staging (NRRD -> NIfTI) and mask loading
# ---------------------------------------------------------------------------
def _nib():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "segmentation needs nibabel — install with: pip install nibabel"
        ) from exc
    return nib


def _refuse_shear(affine: np.ndarray, name: str) -> None:
    cols = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    gram = cols.T @ cols
    if float(np.abs(gram - np.eye(3)).max()) > 1e-4:
        raise ValueError(
            f"{name}: the voxel axes are not orthogonal (a sheared grid, e.g. "
            f"gantry tilt) — a NIfTI qform cannot carry that faithfully; "
            f"resample the CT to an orthogonal grid first"
        )


def write_ct_nifti(array: np.ndarray, affine: np.ndarray, out: Path) -> None:
    """Write a RAS-mm NIfTI with qform = sform = ``affine`` (code scanner)."""
    nib = _nib()
    img = nib.Nifti1Image(array, affine)
    img.header.set_xyzt_units("mm", "sec")
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, str(out))


def stage_ct(path: Path, dest_dir: Path) -> tuple[Path, dict[str, Any], list[str]]:
    """Copy (NIfTI) or convert (NRRD) the CT into ``dest_dir``.

    Returns ``(staged_nifti, grid, notes)`` with ``grid`` = ``{"shape",
    "affine", "format"}`` in RAS millimetres.
    """
    from meshforge.io.image import check_affine, nifti_affine_mm, nifti_mm_scale

    name = path.name.lower()
    if name.endswith(".nrrd"):
        from meshforge.io.nrrd import read_nrrd

        vol = read_nrrd(path, max_voxels=MAX_CT_VOXELS)
        _refuse_shear(vol.affine, path.name)
        staged = dest_dir / "ct.nii.gz"
        write_ct_nifti(vol.array, vol.affine, staged)
        grid = {"shape": tuple(vol.array.shape), "affine": vol.affine, "format": "nrrd"}
        return staged, grid, list(vol.notes)
    if name.endswith((".nii", ".nii.gz")):
        nib = _nib()
        img = nib.load(str(path))
        shape = tuple(int(s) for s in img.header.get_data_shape())
        if len(shape) != 3:
            raise ValueError(f"{path.name}: a 3-D CT is required, got shape {shape}")
        if int(np.prod(shape)) > MAX_CT_VOXELS:
            raise ValueError(
                f"{path.name}: {shape} = {int(np.prod(shape))} voxels is over the "
                f"{MAX_CT_VOXELS} voxel CT ceiling — crop or downsample first"
            )
        factor, _ = nifti_mm_scale(img.header)
        if factor != 1.0:
            raise ValueError(
                f"{path.name} declares spatial unit "
                f"{img.header.get_xyzt_units()[0]!r}; TotalSegmentator reads voxel "
                f"spacing as millimetres — convert the CT to mm first"
            )
        affine, notes = nifti_affine_mm(img)
        _refuse_shear(check_affine(affine), path.name)
        staged = dest_dir / ("ct.nii.gz" if name.endswith(".gz") else "ct.nii")
        shutil.copyfile(path, staged)
        return staged, {"shape": shape, "affine": affine, "format": "nifti"}, notes
    raise ValueError(
        f"unsupported CT input {path.name!r}; supported: {', '.join(CT_SUFFIXES)}"
    )


def load_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """A TotalSegmentator mask as ``(bool array, RAS-mm affine)``.

    The header shape is bounded before any voxel is read; the data is read
    as stored (uint8), never as float64.
    """
    from meshforge.io.image import check_affine

    nib = _nib()
    img = nib.load(str(path))
    shape = tuple(int(s) for s in img.header.get_data_shape())
    if len(shape) != 3 or int(np.prod(shape)) > MAX_CT_VOXELS:
        raise ValueError(
            f"mask {path.name} has shape {shape}: expected a 3-D grid of at most "
            f"{MAX_CT_VOXELS} voxels"
        )
    data = np.asanyarray(img.dataobj)
    return data > 0.5, check_affine(img.affine)


def _refuse_fragment(n_voxels: int, voxel_ml: float, what: str) -> None:
    ml = n_voxels * voxel_ml
    if ml < MIN_AORTA_ML:
        raise ValueError(
            f"TotalSegmentator's aorta mask {what}holds only {ml:.4f} mL "
            f"({n_voxels} voxels) — implausibly small for an aorta (< "
            f"{MIN_AORTA_ML:g} mL). It needs more anatomy than a tight crop "
            f"around the aneurysm; refusing to pass a fragment on as the aorta"
        )


# ---------------------------------------------------------------------------
# contrast lumen (D85): a CT threshold inside the AI mask
# ---------------------------------------------------------------------------
def _ct_box(path: Path, shape: tuple[int, int, int], lo, hi) -> np.ndarray:
    """CT intensities in the index box [lo, hi) — read-only, the stored
    values with the NIfTI scaling applied (Hounsfield units for a CT)."""
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    name = path.name.lower()
    if name.endswith(".nrrd"):
        from meshforge.io.nrrd import read_nrrd

        vol = read_nrrd(path, max_voxels=MAX_CT_VOXELS)
        if tuple(vol.array.shape) != tuple(shape):
            raise ValueError(f"{path.name}: CT shape {vol.array.shape} != grid {shape}")
        return np.asarray(vol.array[sl])
    img = _nib().load(str(path))
    if tuple(int(v) for v in img.header.get_data_shape()) != tuple(shape):
        raise ValueError(f"{path.name}: CT shape differs from the segmented grid")
    return np.asanyarray(img.dataobj[sl])


def ct_lumen_surface(
    ct_path: str | Path,
    mask: np.ndarray,
    mask_affine: np.ndarray,
    ct_shape: tuple[int, int, int],
    ct_affine: np.ndarray,
    *,
    lumen_hu: float = DEFAULT_LUMEN_HU,
    smoothing: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """The contrast-filled blood lumen inside an AI aorta mask.

    lumen = mask AND (CT >= ``lumen_hu``), then the same ``mask_to_surface``
    path as the aorta (largest 6-connected component, blur ``smoothing``
    voxels, level 0.5). Bright calcified plaque is also >= 150 HU: plaque
    detached from the lumen is dropped with the small components (counted);
    plaque touching the lumen stays in it. Refused loudly: a mask not on the
    CT grid, no voxel over the threshold, a lumen under ``MIN_AORTA_ML``.
    Returns ``(surface, summary)``; the summary carries the Dice between the
    lumen and the whole AI mask (the lumen is a subset, so Dice =
    2|L| / (|M| + |L|)).
    """
    from meshforge.io.image import mask_to_surface

    ct_path = Path(ct_path)
    lumen_hu = float(lumen_hu)
    if not np.isfinite(lumen_hu):
        raise ValueError(f"lumen_hu must be a finite HU value, got {lumen_hu}")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != tuple(ct_shape) or not np.allclose(mask_affine, ct_affine, atol=1e-3):
        raise ValueError(
            "the aorta mask is not on the CT grid — the CT cannot be thresholded "
            "inside it voxel for voxel"
        )
    n_mask = int(np.count_nonzero(mask))
    if n_mask == 0:
        raise ValueError("the aorta mask is empty — no lumen to threshold")
    lo, hi = [], []
    for ax in range(3):
        other = tuple(i for i in range(3) if i != ax)
        idx = np.flatnonzero(mask.any(axis=other))
        lo.append(max(int(idx[0]) - 1, 0))
        hi.append(min(int(idx[-1]) + 2, mask.shape[ax]))
    ct = _ct_box(ct_path, tuple(ct_shape), lo, hi)
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] & (ct >= lumen_hu)
    del ct
    n_thr = int(np.count_nonzero(sub))
    voxel_ml = abs(float(np.linalg.det(np.asarray(mask_affine)[:3, :3]))) / 1000.0
    if n_thr == 0:
        raise ValueError(
            f"no voxel inside the aorta mask reaches {lumen_hu:g} HU — not a contrast "
            f"CT (CTA), or the threshold is too high"
        )
    shift = np.eye(4)
    shift[:3, 3] = lo
    surface, prov = mask_to_surface(sub, np.asarray(mask_affine) @ shift, sigma=smoothing)
    kept = int(prov["kept_voxels"])
    if kept * voxel_ml < MIN_AORTA_ML:
        raise ValueError(
            f"the contrast lumen's largest piece holds only {kept * voxel_ml:.4f} mL "
            f"(< {MIN_AORTA_ML:g} mL) — not a lumen; refusing to pass a fragment on"
        )
    summary = {
        "method": LUMEN_METHOD_NOTE.format(hu=lumen_hu),
        "lumen_hu": lumen_hu,
        "mask_voxels": n_mask,
        "above_threshold_voxels": n_thr,
        "kept_voxels": kept,
        "dropped_components": prov["dropped_components"],
        "dropped_voxels": prov["dropped_voxels"],
        "lumen_volume_ml": round(kept * voxel_ml, 3),
        "fraction_of_mask": round(kept / n_mask, 4),
        "dice_vs_ai_mask": round(2.0 * kept / (n_mask + kept), 4),
        "surface": {
            "n_points": prov["n_points"],
            "n_triangles": prov["n_triangles"],
            "watertight": prov["watertight"],
            "enclosed_volume_ml": (
                None if prov["enclosed_volume"] is None
                else round(prov["enclosed_volume"] / 1000.0, 3)
            ),
        },
        "notes": [LUMEN_METHOD_NOTE.format(hu=lumen_hu)] + list(prov["notes"]),
    }
    return surface, summary


# ---------------------------------------------------------------------------
# the whole step
# ---------------------------------------------------------------------------
@dataclass
class SegmentResult:
    surface: Any  # pv.PolyData, RAS world frame, millimetres
    summary: dict[str, Any]
    ct_shape: tuple[int, int, int]
    ct_affine: np.ndarray  # voxel index -> RAS mm
    input_sha256: str
    log: str
    notes: list[str]
    lumen_surface: Any = None  # D85: contrast lumen (CT threshold), when asked
    #: D108 addendum 7: context organs (``heart``) as their own surfaces, same
    #: smoothing, never unioned into the vessel; an organ outside the field is
    #: left out (a note in ``notes``), not an error.
    context_surfaces: dict[str, Any] = field(default_factory=dict)


def segment_ct(
    path: str | Path,
    *,
    device: str = DEFAULT_DEVICE,
    fast: bool = DEFAULT_FAST,
    smoothing: float = 1.0,
    timeout: float = DEFAULT_TIMEOUT_S,
    rois: tuple[str, ...] = ("aorta",),
    context_rois: tuple[str, ...] = ("heart",),
    executable: str | None = None,
    keep_mask: str | Path | None = None,
    lumen_hu: float | None = None,
) -> SegmentResult:
    """CT (.nrrd/.nii/.nii.gz) -> TotalSegmentator -> watertight surface (mm, RAS).

    ``smoothing`` is the Gaussian sigma in voxels applied to the binary mask
    before the level-0.5 surface (0 = off). ``keep_mask`` optionally saves
    the (union) mask NIfTI there before the staging dir is deleted. The CT is
    checksummed before and after: a change on disk is a loud error (rule 1).
    ``lumen_hu`` (D85) also extracts the contrast blood lumen inside the mask
    (``ct_lumen_surface``) as ``result.lumen_surface``, summary key ``lumen``.

    ``context_rois`` (D108 addendum 7) are asked of TotalSegmentator in a
    SECOND call of their own, and each becomes its own surface in
    ``result.context_surfaces`` (summary key ``context``), never unioned
    into the vessel mask: the patient's own heart, drawn as a ghost above
    the arch. A second call, not one call for both: with ``--roi_subset``
    TotalSegmentator first finds the requested classes with a 6 mm model
    and runs the 1.5 mm model on a crop around them, so adding the heart to
    the vessel's call enlarges that crop and CHANGES the vessel's own
    outline (seen on the demo CT: the sac volume, the centreline length and
    the flare landmark all moved, and the run's wall gate failed). The
    vessel call stays exactly what it was without context. Only
    ``CONTEXT_ROIS`` (the whole heart from the free ``total`` task) is
    accepted; an empty context mask — the organ is outside the field of
    view — is skipped with a note, not an error.
    """
    from meshforge.io.image import mask_to_surface
    from meshforge.io.load import file_sha256

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CT volume not found: {path}")
    if not path.name.lower().endswith(CT_SUFFIXES):
        raise ValueError(
            f"unsupported CT input {path.name!r}; supported: {', '.join(CT_SUFFIXES)}"
        )
    device, rois, timeout = _check_run_args(device, rois, timeout)
    context_rois = _check_context_rois(rois, context_rois)
    smoothing = float(smoothing)
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError(f"smoothing must be >= 0 voxels, got {smoothing}")
    exe = executable or find_totalseg()  # fail fast, before staging the CT
    sha_before = file_sha256(path)

    with _staging_dir() as stage:
        staged, grid, notes = stage_ct(path, stage)
        run = run_totalseg(
            staged,
            stage / "ts",
            device=device,
            fast=fast,
            timeout=timeout,
            rois=rois,
            executable=exe,
        )
        notes = notes + run.notes
        crun = None
        if context_rois:  # its own call: the vessel's crop must not change
            crun = run_totalseg(
                staged,
                stage / "ts_context",
                device=device,
                fast=fast,
                timeout=timeout,
                rois=context_rois,
                executable=exe,
                version=run.version,
            )
            notes = notes + [n for n in crun.notes if n not in notes]
        mask = mask_affine = None
        for roi in rois:  # the vessel: one union mask
            m, a = load_mask(run.masks[roi])
            if mask is None:
                mask, mask_affine = m, a
            elif m.shape != mask.shape or not np.allclose(a, mask_affine, atol=1e-4):
                raise RuntimeError(
                    f"TotalSegmentator masks {list(run.masks)} are on different grids"
                )
            else:
                mask |= m
        if mask.shape != tuple(grid["shape"]) or not np.allclose(
            mask_affine, grid["affine"], atol=1e-3
        ):
            notes.append(
                "the mask grid differs from the CT grid; the mask's own affine "
                "is used for the surface"
            )
        if not mask.any():
            raise ValueError(EMPTY_MASK_MESSAGE)
        voxel_ml = abs(float(np.linalg.det(mask_affine[:3, :3]))) / 1000.0
        _refuse_fragment(int(np.count_nonzero(mask)), voxel_ml, "")
        # context organs: each its own mask, loaded while the staging dir lives
        context_masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for roi in context_rois:
            m, a = load_mask(crun.masks[roi])
            if not m.any():
                notes.append(f"{roi} not in field")  # outside the scan: context only
                continue
            context_masks[roi] = (m, a)
        if keep_mask is not None:
            nib = _nib()
            keep_mask = Path(keep_mask)
            keep_mask.parent.mkdir(parents=True, exist_ok=True)
            out = nib.Nifti1Image(mask.astype(np.uint8), mask_affine)
            out.header.set_xyzt_units("mm", "sec")
            out.set_qform(mask_affine, code=1)
            out.set_sform(mask_affine, code=1)
            nib.save(out, str(keep_mask))

    surface, mprov = mask_to_surface(mask, mask_affine, sigma=smoothing)
    _refuse_fragment(mprov["kept_voxels"], voxel_ml, "its largest connected piece ")
    lumen_surface = lumen_summary = None
    if lumen_hu is not None:
        lumen_surface, lumen_summary = ct_lumen_surface(
            path, mask, mask_affine, tuple(grid["shape"]), np.asarray(grid["affine"]),
            lumen_hu=lumen_hu, smoothing=smoothing,
        )
    del mask
    notes = notes + list(mprov["notes"])
    context_surfaces: dict[str, Any] = {}
    context_summary: dict[str, Any] = {}
    for roi, (m, a) in context_masks.items():
        csurf, cprov = mask_to_surface(m, a, sigma=smoothing)
        context_surfaces[roi] = csurf
        context_summary[roi] = {
            "n_points": cprov["n_points"],
            "n_triangles": cprov["n_triangles"],
            "watertight": cprov["watertight"],
            "mask_volume_ml": round(cprov["mask_volume"] / 1000.0, 3),
            "role": "context only (viewport ghost), not part of the vessel",
        }
        notes = notes + [f"{roi}: {n}" for n in cprov["notes"]]
    del context_masks
    if file_sha256(path) != sha_before:
        raise RuntimeError(f"input changed on disk during segmentation: {path}")

    affine = np.asarray(grid["affine"], dtype=float)
    summary = {
        "input": str(path),
        "input_sha256": sha_before,
        "ct": {
            "format": grid["format"],
            "shape": [int(s) for s in grid["shape"]],
            "spacing_mm": [round(float(s), 6) for s in np.linalg.norm(affine[:3, :3], axis=0)],
            "affine_ras_mm": affine.round(6).tolist(),
        },
        # the vessel call (``rois`` unioned into the source) and, apart, the
        # context call with its own flags and time
        "totalseg": {
            **run.summary(),
            "context_rois": list(context_rois),
            "context_flags": list(crun.flags) if crun else [],
            "context_wall_time_s": round(crun.wall_time_s, 2) if crun else None,
        },
        "mask": {
            key: mprov[key]
            for key in (
                "mask_voxels",
                "kept_voxels",
                "components",
                "dropped_components",
                "dropped_voxels",
                "crop_lo",
                "crop_hi",
                "sigma",
            )
        },
        "surface": {
            "n_points": mprov["n_points"],
            "n_triangles": mprov["n_triangles"],
            "watertight": mprov["watertight"],
            "pieces_before_cleanup": mprov["surface_pieces"],
            "enclosed_volume_ml": (
                None
                if mprov["enclosed_volume"] is None
                else round(mprov["enclosed_volume"] / 1000.0, 3)
            ),
            "frame": "RAS world (NIfTI convention), millimetres",
        },
        "mask_volume_ml": round(mprov["mask_volume"] / 1000.0, 3),
        "length_unit": "mm",
        "notes": notes,
    }
    if lumen_summary is not None:
        summary["lumen"] = lumen_summary
    if context_rois:
        summary["context"] = context_summary  # {} when every organ was out of field
    return SegmentResult(
        surface=surface,
        summary=summary,
        ct_shape=tuple(int(s) for s in grid["shape"]),
        ct_affine=affine,
        input_sha256=sha_before,
        log=run.log if crun is None else f"{run.log}\n--- context {'+'.join(context_rois)} ---\n{crun.log}",
        notes=notes,
        lumen_surface=lumen_surface,
        context_surfaces=context_surfaces,
    )
