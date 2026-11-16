"""CalculiX runner: stage a deck copy in a temp dir, run ccx, persist results.

The deck is COPIED into a fresh temporary directory and ccx runs there, so
solver scratch files never land next to the input deck or inside the repo. Only the artifacts we keep (.dat, .frd) are written to
``out_dir`` afterwards. Every failure mode is loud: missing solver, nonzero
exit, ``*ERROR`` in the solver output, missing .frd, timeout.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from meshforge.textguard import scrub

from meshforge.export.neutral import guard_output_path
from meshforge.solve.frd import FrdModel, read_frd

_CCX_FALLBACK = Path.home() / "miniconda3/envs/ccxsolve/bin/ccx"


def find_ccx() -> str | None:
    """MESHFORGE_CCX env var, then PATH, then the known conda env."""
    for candidate in (
        os.environ.get("MESHFORGE_CCX"),
        shutil.which("ccx"),
        str(_CCX_FALLBACK),
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


@dataclass
class SolveResult:
    deck: Path
    ccx: str
    elapsed: float
    frd: FrdModel
    dat_path: Path | None
    frd_path: Path
    n_nodes: int
    u_max: float | None
    mises_max: float | None
    eigenfrequencies: np.ndarray | None = None  # *FREQUENCY steps (V24)
    temp_range: tuple[float, float] | None = None  # NDTEMP present (V24)
    time_steps: np.ndarray | None = None  # transient NDTEMP history (V31)
    buckling_factors: np.ndarray | None = None  # *BUCKLE steps (V32)
    riks_lpf: np.ndarray | None = None  # *STATIC, RIKS arc-length series (V42)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "deck": str(self.deck),
            "ccx": self.ccx,
            "elapsed_s": round(self.elapsed, 3),
            "n_nodes": self.n_nodes,
            "fields": sorted(self.frd.fields),
            "u_max": self.u_max,
            "mises_max": self.mises_max,
            "eigenfrequencies": (
                None
                if self.eigenfrequencies is None
                else [float(f) for f in self.eigenfrequencies]
            ),
            "temp_range": (
                None if self.temp_range is None else list(self.temp_range)
            ),
            "time_steps": (
                None
                if self.time_steps is None
                else [float(t) for t in self.time_steps]
            ),
            "buckling_factors": (
                None
                if self.buckling_factors is None
                else [float(f) for f in self.buckling_factors]
            ),
            "riks_lpf": (
                None
                if self.riks_lpf is None
                else [float(t) for t in self.riks_lpf]
            ),
            "dat": str(self.dat_path) if self.dat_path else None,
            "frd": str(self.frd_path),
            "notes": self.notes,
        }

    def to_text(self) -> str:
        text = (
            f"solve[ccx]: {self.n_nodes} nodes, fields {', '.join(sorted(self.frd.fields)) or '(none)'}, "
            f"{self.elapsed:.1f}s\n"
        )
        if self.u_max is not None:
            text += f"  |U| max     : {self.u_max:.6g}\n"
        if self.mises_max is not None:
            text += f"  von Mises max: {self.mises_max:.6g}\n"
        if self.eigenfrequencies is not None:
            shown = ", ".join(f"{f:.6g}" for f in self.eigenfrequencies[:6])
            more = len(self.eigenfrequencies) - 6
            text += f"  frequencies : {shown}{f' (+{more} more)' if more > 0 else ''}\n"
        if self.buckling_factors is not None:
            shown = ", ".join(f"{f:.6g}" for f in self.buckling_factors[:6])
            more = len(self.buckling_factors) - 6
            text += f"  buckling    : {shown}{f' (+{more} more)' if more > 0 else ''}\n"
        if self.temp_range is not None:
            text += (
                f"  T range     : {self.temp_range[0]:.6g} .. "
                f"{self.temp_range[1]:.6g}\n"
            )
        if self.time_steps is not None:
            text += (
                f"  time steps  : {len(self.time_steps)} "
                f"(t={self.time_steps[0]:.6g} .. {self.time_steps[-1]:.6g})\n"
            )
        if self.riks_lpf is not None:
            text += (
                f"  riks incs   : {len(self.riks_lpf)} arc-length increments "
                f"(load from RF, not this parameter)\n"
            )
        for note in self.notes:
            text += f"  note: {note}\n"
        return text


def solve_deck(
    deck: str | Path,
    out_dir: str | Path,
    *,
    ccx: str | None = None,
    timeout: float = 600.0,
    read_elements: bool = True,
) -> SolveResult:
    """Run CalculiX on ``deck`` and read the results back. ``read_elements``
    False reads the .frd's nodal fields only (``read_frd(elements=False)``)."""
    deck = Path(deck)
    if not deck.is_file():
        raise FileNotFoundError(f"deck not found: {deck}")
    exe = ccx or find_ccx()
    if exe is None:
        raise RuntimeError(
            "CalculiX (ccx) not found: set MESHFORGE_CCX, add ccx to PATH, or "
            "create the conda env (conda create -n ccxsolve -c conda-forge calculix)"
        )
    if not Path(exe).is_file():
        raise RuntimeError(f"CalculiX executable not found: {exe}")

    out_dir = Path(out_dir)
    guard_output_path(out_dir / f"{deck.stem}.frd", input_paths=[deck])

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="meshforge_ccx_") as tmp:
        work = Path(tmp)
        staged = work / deck.name
        shutil.copyfile(deck, staged)
        try:
            proc = subprocess.run(
                [exe, "-i", deck.stem],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ccx exceeded the {timeout:.0f}s timeout on {deck.name}"
            ) from exc
        elapsed = time.perf_counter() - t0
        output = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            raise RuntimeError(
                f"ccx failed (exit {proc.returncode}) on {deck.name}:\n{scrub(output[-2000:])}"
            )
        if "*ERROR" in output:
            raise RuntimeError(
                f"ccx reported an error on {deck.name} (exit 0 — CalculiX "
                f"fails quietly):\n{scrub(output[-2000:])}"
            )
        frd_src = work / f"{deck.stem}.frd"
        if not frd_src.is_file():
            raise RuntimeError(
                f"ccx produced no .frd for {deck.name} — solver output:\n{scrub(output[-2000:])}"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        frd_path = out_dir / frd_src.name
        shutil.copyfile(frd_src, frd_path)
        dat_src = work / f"{deck.stem}.dat"
        dat_path = None
        if dat_src.is_file():
            dat_path = out_dir / dat_src.name
            shutil.copyfile(dat_src, dat_path)

    model = read_frd(frd_path, elements=read_elements)
    notes = list(model.notes)
    u_max = mises_max = None
    disp = model.fields.get("DISP")
    if disp is not None:
        u_max = float(np.linalg.norm(disp.values[:, :3], axis=1).max())
    stress = model.fields.get("STRESS")
    if stress is not None and stress.values.shape[1] == 6:
        from meshforge.solve.frd import von_mises

        mises_max = float(von_mises(stress.values, stress.components).max())
    if not model.fields:
        notes.append("frd contains no nodal result fields (deck requested none?)")

    eigenfrequencies = None
    buckling_factors = None
    if dat_path is not None:
        from meshforge.solve.dat import (
            parse_buckling_factors,
            parse_eigenfrequencies,
        )

        dat_text = dat_path.read_text()
        freqs = parse_eigenfrequencies(dat_text)
        if freqs.size:
            eigenfrequencies = freqs
            notes.append(
                f"{freqs.size} eigenfrequency(ies) parsed from the .dat; the "
                f"frd DISP block holds the LAST mode shape only"
            )
        facs = parse_buckling_factors(dat_text)
        if facs.size:
            buckling_factors = facs
            notes.append(
                f"{facs.size} buckling factor(s) parsed from the .dat "
                f"(factor × load = critical load); the frd DISP block holds "
                f"the LAST mode shape only"
            )
    temp_range = None
    time_steps = None
    ndtemp = model.fields.get("NDTEMP")
    if ndtemp is not None:
        col = ndtemp.values[:, 0]
        temp_range = (float(col.min()), float(col.max()))
        nd_hist = model.history.get("NDTEMP", [])
        if len(nd_hist) > 1:  # transient heat: one block per increment (V31)
            time_steps = np.asarray([t for t, _ in nd_hist], dtype=float)

    # RIKS arc-length step (V42): the frd carries one DISP+FORC (reaction)
    # block per increment. The per-increment "time" is the arc-length/LPF
    # parameter, NOT the load — the load comes from FORC summed over the
    # constrained set. Signature: multi-increment DISP with FORC present, and
    # not a frequency/buckle step (those set eigenfrequencies/buckling_factors).
    riks_lpf = None
    disp_hist = model.history.get("DISP", [])
    if (
        len(disp_hist) > 1
        and "FORC" in model.history
        and eigenfrequencies is None
        and buckling_factors is None
    ):
        riks_lpf = np.asarray([t for t, _ in disp_hist], dtype=float)
        notes.append(
            "RIKS arc-length step: the frd 'time' is the arc-length/LPF "
            "parameter, NOT the applied load — read the true load from the "
            "reaction force (FORC) summed over the constrained set. riks_lpf "
            "holds the per-increment parameter; frd.history['DISP'] and "
            "['FORC'] give the load–displacement curve."
        )

    return SolveResult(
        deck=deck,
        ccx=exe,
        elapsed=elapsed,
        frd=model,
        dat_path=dat_path,
        frd_path=frd_path,
        n_nodes=len(model.node_ids),
        u_max=u_max,
        mises_max=mises_max,
        eigenfrequencies=eigenfrequencies,
        temp_range=temp_range,
        time_steps=time_steps,
        buckling_factors=buckling_factors,
        riks_lpf=riks_lpf,
        notes=notes,
    )
