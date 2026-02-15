from __future__ import annotations

import base64
from io import BytesIO
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def fig_to_base64_png(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_spectrogram_png_base64(
    spec_db: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    fmin: float = 20.0,
    fmax: float = 20000.0,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    mesh = ax.pcolormesh(times, freqs, spec_db, shading="auto", cmap="magma")
    ax.set_yscale("log")
    ax.set_ylim(max(fmin, float(freqs[1]) if len(freqs) > 1 else fmin), fmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram (dB)")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Level (dB, rel. max)")
    fig.tight_layout()
    return fig_to_base64_png(fig)


def render_octave_response_png_base64(
    fc_hz: Iterable[float],
    levels_db: Iterable[float],
    band: str,
    has_reference: bool,
) -> str:
    fc_hz = np.asarray(list(fc_hz), dtype=np.float64)
    levels_db = np.asarray(list(levels_db), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(fc_hz, levels_db, marker="o", linewidth=1.8, markersize=4)
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

    ylabel = "Response (dB)" if has_reference else "Relative Level (dB)"
    title = (
        f"{'1/3' if band == 'third' else '1'} octave response"
        + (" (meas - ref)" if has_reference else " (measurement only)")
    )

    ax.set_title(title)
    ax.set_xlabel("Center Frequency (Hz)")
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    return fig_to_base64_png(fig)
