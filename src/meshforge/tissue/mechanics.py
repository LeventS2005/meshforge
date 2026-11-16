"""Unit and stress-measure conversions for uniaxial strip tests (stdlib only).

Conventions (the USP AAA strip tests, and anything else entering the library):

* lengths in mm, loads in N, so a stress F / (mm * mm) is N/mm^2 = MPa;
* strain is engineering strain (displacement / gauge length), stretch
  lambda = 1 + strain;
* Cauchy stress under incompressibility: both transverse dimensions shrink by
  lambda^-1/2, so the current area is A0 / lambda and sigma = F * lambda / A0;
* engineering (first Piola-Kirchhoff) stress is F / A0 = sigma / lambda;
* membrane tension (force per unit current width) is F / (w0 * lambda^-1/2)
  = F * sqrt(lambda) / w0, equal to sigma * t, with t = t0 / sqrt(lambda).

Every function refuses non-finite or non-physical input (rule 6).
"""
from __future__ import annotations

import math

#: 1 MPa = 1 N/mm^2 = 100 N/cm^2.
N_PER_CM2_PER_MPA = 100.0
#: 1 N/mm = 10 N/cm.
N_PER_CM_PER_N_PER_MM = 10.0


def _finite(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return v


def _positive(name: str, value: float) -> float:
    v = _finite(name, value)
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return v


def _stretch(strain: float) -> float:
    lam = 1.0 + _finite("strain", strain)
    if lam <= 0.0:
        raise ValueError(f"strain {strain!r} gives a non-positive stretch 1 + strain")
    return lam


def n_per_cm2_to_mpa(value: float) -> float:
    """Stress in N/cm^2 -> MPa (divide by 100)."""
    return _finite("stress (N/cm^2)", value) / N_PER_CM2_PER_MPA


def mpa_to_n_per_cm2(value: float) -> float:
    """Stress in MPa -> N/cm^2 (multiply by 100)."""
    return _finite("stress (MPa)", value) * N_PER_CM2_PER_MPA


def n_per_cm_to_n_per_mm(value: float) -> float:
    """Tension in N/cm -> N/mm (divide by 10)."""
    return _finite("tension (N/cm)", value) / N_PER_CM_PER_N_PER_MM


def engineering_from_cauchy(cauchy: float, strain: float) -> float:
    """Engineering stress = Cauchy / (1 + strain) (incompressible uniaxial)."""
    return _finite("Cauchy stress", cauchy) / _stretch(strain)


def cauchy_from_engineering(engineering: float, strain: float) -> float:
    """Cauchy stress = engineering * (1 + strain) (incompressible uniaxial)."""
    return _finite("engineering stress", engineering) * _stretch(strain)


def cauchy_incompressible_mpa(
    load_n: float, stretch: float, width_mm: float, thickness_mm: float
) -> float:
    """sigma = F * lambda / (w0 * t0): N and mm in, MPa out."""
    lam = _positive("stretch", stretch)
    a0 = _positive("width (mm)", width_mm) * _positive("thickness (mm)", thickness_mm)
    return _finite("load (N)", load_n) * lam / a0


def tension_n_per_mm(load_n: float, stretch: float, width_mm: float) -> float:
    """Membrane tension F * sqrt(lambda) / w0: N and mm in, N/mm out."""
    lam = _positive("stretch", stretch)
    return _finite("load (N)", load_n) * math.sqrt(lam) / _positive("width (mm)", width_mm)


def thickness_from_tension_mm(
    tension_n_per_mm_: float, cauchy_mpa: float, strain: float
) -> float:
    """Initial thickness t0 = T * sqrt(lambda) / sigma (T = sigma * t0 / sqrt(lambda)).

    Used where a test reports stress and tension but not the thickness (the
    chart-only strips); N/mm and MPa in, mm out.
    """
    return (
        _positive("tension (N/mm)", tension_n_per_mm_)
        * math.sqrt(_stretch(strain))
        / _positive("Cauchy stress (MPa)", cauchy_mpa)
    )
