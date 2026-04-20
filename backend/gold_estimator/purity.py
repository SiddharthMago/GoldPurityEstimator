from __future__ import annotations

from dataclasses import dataclass


KARAT_DENSITY_RANGES: dict[int, tuple[float, float]] = {
    24: (19.31, 19.51),
    22: (17.45, 18.24),
    20: (16.03, 17.11),
    18: (14.84, 16.12),
    14: (12.91, 14.44),
    10: (11.42, 13.09),
}


@dataclass(frozen=True)
class PurityResult:
    density_g_cm3: float
    karat_formula: float
    karat_formula_clamped: float
    range_match_karat: int | None
    closest_karat: int | None
    classification: str


def density_to_karat_formula(density_g_cm3: float) -> float:
    """Convert density to estimated karat using the notebook polynomial."""
    return (
        0.0089 * density_g_cm3**3
        - 0.550 * density_g_cm3**2
        + 12.5299 * density_g_cm3
        - 77.06
    )


def classify_density(density_g_cm3: float) -> tuple[int | None, int | None, str]:
    for karat, (min_density, max_density) in sorted(
        KARAT_DENSITY_RANGES.items(), reverse=True
    ):
        if min_density <= density_g_cm3 <= max_density:
            return (
                karat,
                karat,
                f"Matches the {karat}K density range ({min_density:.2f}-{max_density:.2f} g/cm3).",
            )

    closest_karat = min(
        KARAT_DENSITY_RANGES,
        key=lambda karat: abs(
            density_g_cm3
            - (KARAT_DENSITY_RANGES[karat][0] + KARAT_DENSITY_RANGES[karat][1])
            / 2
        ),
    )
    min_density, max_density = KARAT_DENSITY_RANGES[closest_karat]
    direction = "below" if density_g_cm3 < min_density else "above"
    return (
        None,
        closest_karat,
        f"No exact range match. Density is {direction} the nearest {closest_karat}K range ({min_density:.2f}-{max_density:.2f} g/cm3).",
    )


def estimate_purity(weight_g: float, volume_cm3: float) -> PurityResult:
    if weight_g <= 0:
        raise ValueError("Weight must be greater than zero.")
    if volume_cm3 <= 0:
        raise ValueError("Volume must be greater than zero.")

    density = weight_g / volume_cm3
    karat_formula = density_to_karat_formula(density)
    range_match, closest, classification = classify_density(density)

    return PurityResult(
        density_g_cm3=density,
        karat_formula=karat_formula,
        karat_formula_clamped=max(0.0, min(24.0, karat_formula)),
        range_match_karat=range_match,
        closest_karat=closest,
        classification=classification,
    )

