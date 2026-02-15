from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
from scipy.signal import medfilt

EPS = 1e-12


def load_audio_mono(file_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load audio as mono float32 waveform."""
    y, loaded_sr = librosa.load(file_path, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Audio file has no samples: {file_path}")
    return y.astype(np.float32), int(loaded_sr)


def compute_basic_metrics(file_path: str) -> Dict[str, float]:
    y, sr = load_audio_mono(file_path, sr=None)
    duration_sec = float(len(y) / sr)

    rms = float(np.sqrt(np.mean(np.square(y)) + EPS))
    peak = float(np.max(np.abs(y)) + EPS)
    crest_factor = float(peak / rms)

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=2048, hop_length=512
    )
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)

    return {
        "sr": sr,
        "num_samples": int(len(y)),
        "duration_sec": duration_sec,
        "rms": rms,
        "peak": peak,
        "crest_factor": crest_factor,
        "spectral_centroid_mean_hz": float(np.mean(spectral_centroid)),
        "zcr_mean": float(np.mean(zcr)),
    }


def compute_spectrogram_db(
    file_path: str,
    n_fft: int = 2048,
    hop_length: int = 512,
    eps: float = EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    y, sr = load_audio_mono(file_path, sr=None)

    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
    )
    power = np.abs(stft) ** 2
    spec_db = librosa.power_to_db(power + eps, ref=np.max)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.times_like(power, sr=sr, hop_length=hop_length)

    meta = {
        "sr": sr,
        "duration_sec": float(len(y) / sr),
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "num_frames": int(power.shape[1]),
        "num_bins": int(power.shape[0]),
    }
    return spec_db.astype(np.float32), freqs.astype(np.float32), times.astype(np.float32), meta


def compute_power_spectrum(
    file_path: str,
    n_fft: int = 4096,
    hop_length: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Average STFT power spectrum over time."""
    y, sr = load_audio_mono(file_path, sr=None)

    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
    )
    power = np.abs(stft) ** 2
    avg_power = np.mean(power, axis=1)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    meta = {
        "sr": sr,
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "duration_sec": float(len(y) / sr),
    }
    return freqs.astype(np.float64), avg_power.astype(np.float64), meta


def make_octave_bands(
    band: str = "third",
    fmin: float = 20.0,
    fmax: float = 20000.0,
) -> List[Dict[str, float]]:
    if band not in {"octave", "third"}:
        raise ValueError("band must be 'octave' or 'third'")
    if fmin <= 0 or fmax <= fmin:
        raise ValueError("fmin/fmax must satisfy 0 < fmin < fmax")

    bands_per_octave = 1 if band == "octave" else 3
    half_step = 1.0 / (2.0 * bands_per_octave)
    edge_ratio = 2.0**half_step

    k_min = math.floor(bands_per_octave * math.log2(fmin / 1000.0)) - 2
    k_max = math.ceil(bands_per_octave * math.log2(fmax / 1000.0)) + 2

    bands: List[Dict[str, float]] = []
    for k in range(k_min, k_max + 1):
        fc = 1000.0 * (2.0 ** (k / bands_per_octave))
        f1 = fc / edge_ratio
        f2 = fc * edge_ratio

        if f2 < fmin or f1 > fmax:
            continue

        bands.append(
            {
                "fc_hz": float(fc),
                "f1_hz": float(max(f1, fmin)),
                "f2_hz": float(min(f2, fmax)),
            }
        )

    bands.sort(key=lambda b: b["fc_hz"])
    return bands


def band_levels_db(
    freqs: np.ndarray,
    power: np.ndarray,
    bands: List[Dict[str, float]],
    eps: float = EPS,
) -> np.ndarray:
    if len(freqs) < 2:
        raise ValueError("Need at least two frequency bins")

    df = float(np.mean(np.diff(freqs)))
    levels_db = []

    for b in bands:
        mask = (freqs >= b["f1_hz"]) & (freqs < b["f2_hz"])
        if not np.any(mask):
            band_power = eps
        else:
            band_power = float(np.sum(power[mask]) * df)
        levels_db.append(10.0 * np.log10(band_power + eps))

    return np.asarray(levels_db, dtype=np.float64)


def align_reference_spectrum(
    target_freqs: np.ndarray,
    ref_freqs: np.ndarray,
    ref_power: np.ndarray,
) -> np.ndarray:
    """Interpolate reference spectrum onto target frequency bins."""
    return np.interp(
        target_freqs,
        ref_freqs,
        ref_power,
        left=float(ref_power[0]),
        right=float(ref_power[-1]),
    )


def smooth_levels(levels: np.ndarray, smoothing: str = "none") -> np.ndarray:
    if smoothing == "none":
        return levels
    if smoothing == "median":
        if len(levels) < 3:
            return levels
        return medfilt(levels, kernel_size=3)
    raise ValueError("smoothing must be 'none' or 'median'")
