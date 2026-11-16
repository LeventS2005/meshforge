"""Solver material record (Abaqus-keyword blocks) used by the AAA stress run.

``Material.elastic(...)`` builds a validated material and ``.lines()`` emits
its ``*MATERIAL`` block for a CalculiX/Abaqus-style input file.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    """Solver material; build via the classmethod constructors."""

    name: str
    kind: str  # "elastic" | "neo_hooke" | "mooney_rivlin" | "thermal"
    params: tuple[float, ...]
    density: float | None = None
    specific_heat: float | None = None  # thermal only (V31 transient)
    # thermal only (V41): temperature-dependent conductivity as ascending
    # (k, T) rows; when set, params[0] is unused and lines() emits the table.
    conductivity_table: tuple[tuple[float, float], ...] | None = None
    # elastic only (V45): constant thermal-expansion coefficient (*EXPANSION).
    expansion: float | None = None

    @classmethod
    def elastic(
        cls,
        E: float,
        nu: float,
        *,
        name: str = "MAT",
        density: float | None = None,
        expansion: float | None = None,
    ) -> "Material":
        if not E > 0:
            raise ValueError(f"Young's modulus must be positive, got E={E}")
        if not -1.0 < nu < 0.5:
            raise ValueError(f"Poisson's ratio must satisfy -1 < nu < 0.5, got nu={nu}")
        if expansion is not None:
            if not isinstance(expansion, (int, float)):
                raise ValueError(
                    "temperature-dependent expansion tables are not supported: "
                    "ccx evaluates an *EXPANSION table as a secant anchored at "
                    "numeric temperature ZERO — not at the *INITIAL CONDITIONS "
                    "reference — so the identical physics yields different "
                    "strains in Celsius vs Kelvin with no warning (V45 probe); "
                    "pass a constant alpha"
                )
            if not (np.isfinite(expansion) and expansion > 0):
                raise ValueError(
                    f"thermal-expansion coefficient must be a positive finite "
                    f"number, got {expansion}"
                )
            expansion = float(expansion)
        return cls(
            name=name, kind="elastic", params=(float(E), float(nu)),
            density=density, expansion=expansion,
        )

    @classmethod
    def neo_hookean(
        cls, c10: float, d1: float, *, name: str = "MAT", density: float | None = None
    ) -> "Material":
        if not c10 > 0:
            raise ValueError(f"neo-Hookean C10 must be positive, got {c10}")
        if d1 < 0:
            raise ValueError(f"neo-Hookean D1 must be >= 0, got {d1}")
        return cls(name=name, kind="neo_hooke", params=(float(c10), float(d1)), density=density)

    @classmethod
    def mooney_rivlin(
        cls,
        c10: float,
        c01: float,
        d1: float,
        *,
        name: str = "MAT",
        density: float | None = None,
    ) -> "Material":
        if not c10 + c01 > 0:
            raise ValueError(f"Mooney-Rivlin needs C10+C01 > 0, got {c10}+{c01}")
        if d1 < 0:
            raise ValueError(f"Mooney-Rivlin D1 must be >= 0, got {d1}")
        return cls(
            name=name,
            kind="mooney_rivlin",
            params=(float(c10), float(c01), float(d1)),
            density=density,
        )

    @classmethod
    def thermal(
        cls,
        k: float | Sequence[tuple[float, float]],
        *,
        name: str = "MAT",
        density: float | None = None,
        specific_heat: float | None = None,
    ) -> "Material":
        """Heat-transfer material: isotropic conductivity (V24); transient
        steps additionally need density + specific_heat (V31).

        ``k`` is a positive scalar, or a temperature-dependent table of
        ascending ``(k, T)`` rows (V41 — *CONDUCTIVITY with a temperature
        column; every k>0, temperatures strictly increasing, >= 2 rows)."""
        if specific_heat is not None and not specific_heat > 0:
            raise ValueError(
                f"specific heat must be positive, got {specific_heat}"
            )
        sh = None if specific_heat is None else float(specific_heat)
        if isinstance(k, (int, float)):
            if not k > 0:
                raise ValueError(f"conductivity must be positive, got k={k}")
            return cls(
                name=name, kind="thermal", params=(float(k),),
                density=density, specific_heat=sh,
            )
        table = tuple((float(kk), float(tt)) for kk, tt in k)
        if len(table) < 2:
            raise ValueError(
                "a conductivity table needs >= 2 (k, T) rows — pass a scalar "
                f"for constant conductivity, got {len(table)} row(s)"
            )
        for kk, _t in table:
            if not kk > 0:
                raise ValueError(
                    f"every conductivity-table k must be positive, got k={kk}"
                )
        temps = [t for _k, t in table]
        if any(b <= a for a, b in zip(temps, temps[1:])):
            raise ValueError(
                f"conductivity-table temperatures must strictly increase, "
                f"got {temps}"
            )
        return cls(
            name=name, kind="thermal", params=(table[0][0],),
            density=density, specific_heat=sh, conductivity_table=table,
        )

    @property
    def is_hyperelastic(self) -> bool:
        return self.kind in ("neo_hooke", "mooney_rivlin")

    def lines(self) -> list[str]:
        out = [f"*MATERIAL, NAME={self.name}"]
        vals = ", ".join(f"{v:.10g}" for v in self.params)
        if self.kind == "elastic":
            out += ["*ELASTIC", vals]
            if self.expansion is not None:
                out += ["*EXPANSION", f"{self.expansion:.10g}"]
        elif self.kind == "neo_hooke":
            out += ["*HYPERELASTIC, NEO HOOKE", vals]
        elif self.kind == "mooney_rivlin":
            out += ["*HYPERELASTIC, MOONEY-RIVLIN", vals]
        elif self.kind == "thermal":
            if self.conductivity_table is not None:
                out.append("*CONDUCTIVITY")
                out += [f"{k:.10g}, {t:.10g}" for k, t in self.conductivity_table]
            else:
                out += ["*CONDUCTIVITY", vals]
        else:  # pragma: no cover - constructors are the only entry points
            raise ValueError(f"unknown material kind {self.kind!r}")
        if self.density is not None:
            out += ["*DENSITY", f"{self.density:.10g}"]
        if self.specific_heat is not None:
            out += ["*SPECIFIC HEAT", f"{self.specific_heat:.10g}"]
        return out

