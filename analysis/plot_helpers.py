"""Small formatting and axis-limit helpers shared by the wind plots."""
import numpy as np


def padded_log_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    values = arrays.ravel() if isinstance(arrays, np.ndarray) else np.concatenate(arrays)
    values = values[np.isfinite(values) & (values > 0.0)]
    return float(values.min() / 1.35), float(values.max() * 1.35)


def padded_linear_limits(
    arrays: list[np.ndarray],
    include_zero: bool = False,
) -> tuple[float, float]:
    values = arrays.ravel() if isinstance(arrays, np.ndarray) else np.concatenate(arrays)
    values = values[np.isfinite(values)]
    low, high = float(values.min()), float(values.max())
    if include_zero:
        low, high = min(low, 0.0), max(high, 0.0)
    span = high - low or max(abs(low), 1.0)
    return low - 0.12 * span, high + 0.12 * span


def signed_log_limits(arrays: list[np.ndarray], *, min_fraction: float = 1.0e-10) -> tuple[float, float, float]:
    """Return symmetric-log limits and a useful linear threshold."""
    values = np.concatenate([np.ravel(array) for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot determine limits from empty data")

    maximum = float(np.max(np.abs(values)))
    nonzero = np.abs(values[values != 0.0])
    if maximum == 0.0 or nonzero.size == 0:
        return -1.0, 1.0, 1.0e-3

    linthresh = max(float(np.percentile(nonzero, 1.0)), maximum * min_fraction)
    low = min(float(values.min()) * 1.25, -linthresh)
    high = max(float(values.max()) * 1.25, linthresh)
    return low, high, linthresh


def scientific_latex(value: float) -> str:
    """Format a positive value compactly for a Matplotlib math label."""
    mantissa_text, exponent_text = f"{value:.2e}".split("e")
    mantissa = float(mantissa_text)
    exponent = int(exponent_text)
    if np.isclose(mantissa, 1.0):
        return rf"10^{{{exponent}}}"
    return rf"{mantissa:g}\times10^{{{exponent}}}"


def sample_histories(
    profiles: list[dict[str, np.ndarray]],
    radius_km: np.ndarray,
    key: str,
) -> list[tuple[str, int, np.ndarray]]:
    """Nearest-cell presentation histories; numerical exports interpolate exactly."""
    requested = (
        (r"$20\,\mathrm{km}$", 20.0),
        (r"$100\,\mathrm{km}$", 100.0),
        (r"$1000\,\mathrm{km}$", 1000.0),
    )
    samples: list[tuple[str, int, np.ndarray]] = []
    for label, requested_radius in requested:
        index = int(np.argmin(np.abs(radius_km - requested_radius)))
        history = np.asarray([profile[key][index] for profile in profiles])
        samples.append((label, index, history))
    samples.append(
        (
            "outer",
            radius_km.size - 1,
            np.asarray([profile[key][-1] for profile in profiles]),
        )
    )
    return samples
