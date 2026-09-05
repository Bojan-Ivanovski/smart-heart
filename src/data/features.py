import math

import numpy as np
from scipy.signal import welch

from ..domain.recording import FrequencyBand, SignalRepresentation
from ..domain.recording.aggregate_features import TemporalVariability
from ..domain.segment import WindowAnalysis


_FREQUENCY_BANDS = {
    FrequencyBand.DELTA: (1.0, 4.0),
    FrequencyBand.THETA: (4.0, 8.0),
    FrequencyBand.ALPHA: (8.0, 13.0),
    FrequencyBand.BETA: (13.0, 30.0),
    FrequencyBand.GAMMA: (30.0, 45.0),
}


def analyze_window(
    signal: np.ndarray,
    *,
    representation: SignalRepresentation,
    sampling_rate_hz: float | None,
    channel_names: list[str],
) -> WindowAnalysis:
    cleaned = np.nan_to_num(np.asarray(signal, dtype=np.float64))
    if cleaned.ndim != 2 or cleaned.shape[1] < 2:
        raise ValueError("Window analysis requires channels-first two-dimensional data.")

    if representation is SignalRepresentation.RAW_EEG:
        if sampling_rate_hz is None:
            raise ValueError("Raw EEG window analysis requires a sampling rate.")
        relative_power, spectral_entropy = _raw_eeg_spectrum(
            cleaned,
            sampling_rate_hz,
        )
    else:
        relative_power = _band_power_channels(cleaned, channel_names)
        spectral_entropy = _normalized_entropy(np.array(list(relative_power.values())))

    temporal_variability = _temporal_variability(cleaned)
    artifact_fraction, normalized_rms, zero_crossing_rate = _time_domain_features(
        cleaned
    )
    theta = relative_power[FrequencyBand.THETA]
    alpha = relative_power[FrequencyBand.ALPHA]
    beta = relative_power[FrequencyBand.BETA]
    return WindowAnalysis(
        relative_band_power={
            band.value: _rounded(power) for band, power in relative_power.items()
        },
        dominant_band=max(relative_power, key=relative_power.get),
        theta_beta_ratio=_rounded(theta / max(beta, 1e-12)),
        theta_alpha_ratio=_rounded(theta / max(alpha, 1e-12)),
        spectral_entropy=_rounded(float(np.clip(spectral_entropy, 0.0, 1.0))),
        channel_correlation_mean=_rounded(_channel_correlation(cleaned)),
        temporal_variability=temporal_variability,
        artifact_fraction=_rounded(artifact_fraction),
        rms_amplitude_normalized=_rounded(normalized_rms),
        zero_crossing_rate=_rounded(zero_crossing_rate),
    )


def _raw_eeg_spectrum(
    signal: np.ndarray,
    sampling_rate_hz: float,
) -> tuple[dict[FrequencyBand, float], float]:
    frequencies, spectrum = welch(
        signal,
        fs=sampling_rate_hz,
        axis=1,
        nperseg=min(512, signal.shape[1]),
    )
    mean_spectrum = spectrum.mean(axis=0)
    powers = {
        band: float(
            np.trapezoid(
                mean_spectrum[(frequencies >= lower) & (frequencies < upper)],
                frequencies[(frequencies >= lower) & (frequencies < upper)],
            )
        )
        for band, (lower, upper) in _FREQUENCY_BANDS.items()
    }
    relative_power = _normalize_power(powers)
    analysis_spectrum = mean_spectrum[
        (frequencies >= 1.0) & (frequencies < 45.0)
    ]
    return relative_power, _normalized_entropy(analysis_spectrum)


def _band_power_channels(
    signal: np.ndarray,
    channel_names: list[str],
) -> dict[FrequencyBand, float]:
    if signal.shape[0] != len(channel_names):
        raise ValueError("Channel metadata does not match the time series.")
    powers = {band: 0.0 for band in FrequencyBand}
    for channel_name, power in zip(channel_names, np.abs(signal).mean(axis=1)):
        normalized_name = channel_name.lower().replace("-", "_")
        try:
            band = FrequencyBand(normalized_name)
        except ValueError:
            if "beta" not in normalized_name:
                continue
            band = FrequencyBand.BETA
        powers[band] += float(power)
    return _normalize_power(powers)


def _normalize_power(
    powers: dict[FrequencyBand, float],
) -> dict[FrequencyBand, float]:
    total = sum(powers.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Cannot calculate relative power from an empty spectrum.")
    relative = {band: power / total for band, power in powers.items()}
    rounded = {band: _rounded(power) for band, power in relative.items()}
    dominant = max(rounded, key=rounded.get)
    rounded[dominant] = _rounded(rounded[dominant] + 1.0 - sum(rounded.values()))
    return rounded


def _normalized_entropy(values: np.ndarray) -> float:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size <= 1:
        return 0.0
    probability = positive / positive.sum()
    return float(
        -np.sum(probability * np.log(probability + 1e-15))
        / np.log(probability.size)
    )


def _channel_correlation(signal: np.ndarray) -> float:
    if signal.shape[0] <= 1:
        return 1.0
    correlations = np.corrcoef(signal)
    values = correlations[np.triu_indices(signal.shape[0], k=1)]
    correlation = float(np.nanmean(values))
    if not math.isfinite(correlation):
        correlation = 0.0
    return float(np.clip(correlation, -1.0, 1.0))


def _temporal_variability(signal: np.ndarray) -> TemporalVariability:
    chunk_count = min(8, max(2, signal.shape[1] // 256))
    chunk_rms = np.array(
        [
            np.sqrt(np.mean(chunk * chunk))
            for chunk in np.array_split(signal, chunk_count, axis=1)
        ]
    )
    score = float(np.std(chunk_rms) / (np.mean(chunk_rms) + 1e-12))
    if score < 0.08:
        return TemporalVariability.STABLE
    if score < 0.2:
        return TemporalVariability.MODERATELY_VARIABLE
    return TemporalVariability.HIGHLY_VARIABLE


def _time_domain_features(signal: np.ndarray) -> tuple[float, float, float]:
    medians = np.median(signal, axis=1, keepdims=True)
    deviations = np.median(np.abs(signal - medians), axis=1, keepdims=True)
    robust_z = np.abs(signal - medians) / (1.4826 * deviations + 1e-12)
    artifact_fraction = float(np.mean(np.any(robust_z > 8.0, axis=0)))
    zero_crossing_rate = float(
        np.mean(np.diff(np.signbit(signal), axis=1) != 0)
    )
    rms = float(np.sqrt(np.mean(signal * signal)))
    robust_scale = float(np.median(deviations) * 1.4826)
    normalized_rms = rms / (robust_scale + 1e-12)
    return (
        float(np.clip(artifact_fraction, 0.0, 1.0)),
        max(normalized_rms, 0.0),
        float(np.clip(zero_crossing_rate, 0.0, 1.0)),
    )


def _rounded(value: float) -> float:
    return round(float(value), 6)
