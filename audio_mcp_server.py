from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np

from fastmcp import FastMCP

from src.audio.features import (
    align_reference_spectrum,
    band_levels_db,
    compute_basic_metrics,
    compute_power_spectrum,
    compute_spectrogram_db,
    make_octave_bands,
    smooth_levels,
)
from src.audio.plots import (
    fig_to_base64_png,
    render_octave_response_png_base64,
    render_spectrogram_png_base64,
)
from src.fan_noise import (
    compare_logs,
    generate_sample_logs_pair,
    load_noise_csv,
    render_compare_png_base64,
    summarize_noise_series,
)

mcp = FastMCP("audio-analysis-example")


def _assert_exists(path: str, name: str) -> None:
    if not path:
        raise ValueError(f"{name} is required")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} does not exist: {path}")


@mcp.tool()
def analyze_audio_basic(file_path: str) -> dict:
    """Return basic audio metrics (sr, duration, rms, centroid, zcr...)."""
    _assert_exists(file_path, "file_path")
    return compute_basic_metrics(file_path)


@mcp.tool()
def render_spectrogram_png(
    file_path: str,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: int = 20,
    fmax: int = 20000,
) -> dict:
    """Render spectrogram as PNG base64 and return metadata."""
    _assert_exists(file_path, "file_path")

    spec_db, freqs, times, meta = compute_spectrogram_db(
        file_path=file_path,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    png_base64 = render_spectrogram_png_base64(
        spec_db=spec_db,
        times=times,
        freqs=freqs,
        fmin=float(fmin),
        fmax=float(fmax),
    )

    meta.update({"fmin": int(fmin), "fmax": int(fmax)})
    return {"png_base64": png_base64, "meta": meta}


@mcp.tool()
def render_octave_response_png(
    meas_path: str,
    ref_path: Optional[str] = None,
    band: str = "third",
    fmin: float = 20.0,
    fmax: float = 20000.0,
    smoothing: str = "none",
) -> dict:
    """
    Render octave/third-octave response as PNG base64.

    If ref_path is provided, returns transfer response in dB = L_meas - L_ref.
    Otherwise returns measurement-only relative level.
    """
    _assert_exists(meas_path, "meas_path")
    if ref_path:
        _assert_exists(ref_path, "ref_path")

    if band not in {"octave", "third"}:
        raise ValueError("band must be 'octave' or 'third'")
    if smoothing not in {"none", "median"}:
        raise ValueError("smoothing must be 'none' or 'median'")

    meas_freqs, meas_power, meas_meta = compute_power_spectrum(meas_path)
    bands = make_octave_bands(band=band, fmin=fmin, fmax=fmax)
    if not bands:
        raise ValueError("No octave bands available for selected range")

    meas_levels = band_levels_db(meas_freqs, meas_power, bands)

    ref_meta = None
    has_reference = bool(ref_path)
    if has_reference:
        ref_freqs, ref_power, ref_meta = compute_power_spectrum(ref_path)
        ref_power_on_meas_bins = align_reference_spectrum(meas_freqs, ref_freqs, ref_power)
        ref_levels = band_levels_db(meas_freqs, ref_power_on_meas_bins, bands)
        response_db = meas_levels - ref_levels
    else:
        response_db = meas_levels

    response_db = smooth_levels(response_db, smoothing=smoothing)

    fc = [b["fc_hz"] for b in bands]
    png_base64 = render_octave_response_png_base64(
        fc_hz=fc,
        levels_db=response_db,
        band=band,
        has_reference=has_reference,
    )

    bands_out = [
        {
            "fc_hz": float(fc_i),
            "db": float(db_i),
        }
        for fc_i, db_i in zip(fc, response_db)
    ]

    meta = {
        "band": band,
        "fmin": float(fmin),
        "fmax": float(fmax),
        "smoothing": smoothing,
        "has_reference": has_reference,
        "mode": "transfer_function_db" if has_reference else "measurement_relative_db",
        "num_bands": len(bands_out),
        "meas_sr": meas_meta.get("sr"),
        "meas_duration_sec": meas_meta.get("duration_sec"),
    }
    if ref_meta is not None:
        meta.update(
            {
                "ref_sr": ref_meta.get("sr"),
                "ref_duration_sec": ref_meta.get("duration_sec"),
            }
        )

    return {"png_base64": png_base64, "bands": bands_out, "meta": meta}


WORKSPACE_ROOT = Path(os.environ.get("MCP_WORKSPACE", os.getcwd())).resolve()
print(f"[MCP] WORKSPACE_ROOT = {WORKSPACE_ROOT}")

def _is_within(child: Path, root: Path) -> bool:
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False

def resolve_in_workspace(p: str, *, must_exist: bool = True) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    path = path.resolve()

    if not _is_within(path, WORKSPACE_ROOT):
        raise PermissionError(f"Access denied: {path} is outside workspace {WORKSPACE_ROOT}")
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return path

@mcp.tool
def list_dir(path: str = ".") -> dict:
    """워크스페이스 내 디렉토리 목록"""
    d = resolve_in_workspace(path)
    if not d.is_dir():
        return {"ok": False, "error": "not_a_directory", "path": str(d)}
    items = []
    for p in sorted(d.iterdir()):
        try:
            st = p.stat()
            items.append({
                "name": p.name,
                "path": str(p.relative_to(WORKSPACE_ROOT)),
                "is_dir": p.is_dir(),
                "size_bytes": st.st_size,
            })
        except Exception:
            items.append({
                "name": p.name,
                "path": str(p.relative_to(WORKSPACE_ROOT)),
                "is_dir": p.is_dir(),
                "size_bytes": None,
            })
    return {"ok": True, "root": str(WORKSPACE_ROOT), "items": items}

@mcp.tool
def read_text_file(path: str, max_chars: int = 20000) -> dict:
    """워크스페이스 내 텍스트 파일 읽기(크기 제한)"""
    p = resolve_in_workspace(path)
    if p.is_dir():
        return {"ok": False, "error": "is_directory", "path": str(p)}
    # 안전: 바이너리/대용량 방지
    if p.stat().st_size > 5_000_000:
        return {"ok": False, "error": "file_too_large", "size_bytes": p.stat().st_size, "path": str(p)}

    data = None
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            data = p.read_text(encoding=enc)
            used = enc
            break
        except Exception:
            continue
    if data is None:
        return {"ok": False, "error": "cannot_decode_text", "path": str(p)}

    if len(data) > max_chars:
        data = data[:max_chars] + "\n\n[TRUNCATED]"
    return {"ok": True, "path": str(p.relative_to(WORKSPACE_ROOT)), "encoding": used, "content": data}

@mcp.tool
def glob_files(pattern: str) -> dict:
    """워크스페이스 내 glob 검색 (예: **/*.py)"""
    # pattern은 워크스페이스 기준으로만 허용
    base = str(WORKSPACE_ROOT)
    full_pattern = str((WORKSPACE_ROOT / pattern).resolve())
    # 워크스페이스 밖으로 튀는 패턴 방지(대충 차단)
    if not full_pattern.startswith(base):
        return {"ok": False, "error": "pattern_outside_workspace"}
    matches = glob.glob(full_pattern, recursive=True)
    rel = []
    for m in matches[:500]:
        try:
            rel.append(str(Path(m).resolve().relative_to(WORKSPACE_ROOT)))
        except Exception:
            continue
    return {"ok": True, "count": len(matches), "matches": rel}

@mcp.tool
def render_fft_png(
    file_path: str,
    fmin: float = 20.0,
    fmax: float = 20000.0,
    window_sec: float = 2.0,
    scale: str = "db",  # "db" or "linear"
) -> dict:
    """
    오디오 파일의 FFT(단일 채널) 스펙트럼을 PNG(base64)로 반환.
    - window_sec: 분석에 사용할 구간(초). 파일이 짧으면 전체 사용.
    - scale: "db"면 dB, "linear"면 선형 진폭.
    """
    # workspace guard가 있으면 사용
    try:
        safe = resolve_in_workspace(file_path)  # 너의 서버에 있다면
        path = str(safe)
    except NameError:
        path = file_path

    y, sr = librosa.load(path, sr=None, mono=True)

    # 중앙 구간을 window_sec만큼 선택
    n = len(y)
    win_n = int(max(1, min(n, window_sec * sr)))
    start = max(0, (n - win_n) // 2)
    seg = y[start:start + win_n]

    # Hann window
    w = np.hanning(len(seg))
    seg_w = seg * w

    # rFFT
    X = np.fft.rfft(seg_w)
    mag = np.abs(X)

    freqs = np.fft.rfftfreq(len(seg_w), d=1.0 / sr)

    # 범위 필터
    mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    freqs_plot = freqs[mask]
    mag_plot = mag[mask]

    eps = 1e-12
    if str(scale).lower() == "db":
        y_plot = 20.0 * np.log10(mag_plot + eps)  # amplitude dB
        y_label = "Magnitude (dB)"
    else:
        y_plot = mag_plot
        y_label = "Magnitude (linear)"

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(freqs_plot, y_plot)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(y_label)
    ax.set_title("FFT Spectrum")
    ax.grid(True, which="both", ls=":", lw=0.5)

    png_b64 = fig_to_base64_png(fig)
    return {
        "png_base64": png_b64,
        "meta": {
            "sample_rate": int(sr),
            "window_sec_used": float(len(seg) / sr),
            "fmin": float(fmin),
            "fmax": float(fmax),
            "scale": str(scale).lower(),
        },
    }

def _safe_path(p: str) -> str:
    """서버에 resolve_in_workspace()가 있으면 사용하고, 없으면 그대로"""
    try:
        safe = resolve_in_workspace(p)  # type: ignore[name-defined]
        return str(safe)
    except NameError:
        return p


def _fft_spectrum(
    file_path: str,
    fmin: float,
    fmax: float,
    window_sec: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """중앙 구간 FFT 스펙트럼 계산(단일 채널)"""
    path = _safe_path(file_path)
    y, sr = librosa.load(path, sr=None, mono=True)

    n = len(y)
    win_n = int(max(1, min(n, window_sec * sr)))
    start = max(0, (n - win_n) // 2)
    seg = y[start : start + win_n]

    w = np.hanning(len(seg))
    seg_w = seg * w

    X = np.fft.rfft(seg_w)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(len(seg_w), d=1.0 / sr)

    mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    return freqs[mask], mag[mask], int(sr)


@mcp.tool
def render_fft_compare_png(
    paths: List[str],
    fmin: float = 20.0,
    fmax: float = 20000.0,
    window_sec: float = 2.0,
    scale: str = "db",  # "db" or "linear"
) -> dict:
    """
    여러 파일 FFT 스펙트럼을 하나의 plot에 겹쳐서 비교(overlay) PNG(base64)로 반환.
    """
    if not paths or len(paths) < 2:
        return {"error": "paths must contain at least 2 items"}

    scale = str(scale).lower()
    if scale not in {"db", "linear"}:
        scale = "db"

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)

    used_sr = None
    for p in paths:
        freqs, mag, sr = _fft_spectrum(p, fmin=fmin, fmax=fmax, window_sec=window_sec)
        used_sr = used_sr or sr

        eps = 1e-12
        if scale == "db":
            y_plot = 20.0 * np.log10(mag + eps)
            y_label = "Magnitude (dB)"
        else:
            y_plot = mag
            y_label = "Magnitude (linear)"

        label = Path(p).name
        ax.plot(freqs, y_plot, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(y_label)
    ax.set_title("FFT Compare (Overlay)")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(fontsize=8)

    png_b64 = fig_to_base64_png(fig)
    return {
        "png_base64": png_b64,
        "meta": {
            "paths": [Path(p).name for p in paths],
            "sample_rate_first": int(used_sr) if used_sr is not None else None,
            "window_sec": float(window_sec),
            "fmin": float(fmin),
            "fmax": float(fmax),
            "scale": scale,
        },
    }


# (선택) 옥타브 비교도 overlay로 만들고 싶으면: 서버에 옥타브 band 계산 함수가 이미 있을 때 쉽게 붙일 수 있음.
# 기존 서버에 make_octave_bands / band_levels_db 같은 로직이 있다면, 그걸 호출해서 plot만 overlay 하면 된다.

@mcp.tool
def render_octave_compare_png(
    paths: List[str],
    band: str = "third",   # "octave" or "third"
    fmin: float = 20.0,
    fmax: float = 20000.0,
    smoothing: str = "none",
) -> dict:
    """
    여러 파일의 옥타브(또는 1/3옥타브) band 레벨(dB)을 overlay 비교 plot으로 반환.
    전제: 서버에 기존 단일 render_octave_response_png에서 쓰는 내부 계산 함수가 있어야 함.
    없다면 이 tool은 사용하지 말고(=app에서 자동 fallback), 필요 시 네 기존 로직을 여기로 옮겨 붙이면 됨.
    """
    if not paths or len(paths) < 2:
        return {"error": "paths must contain at least 2 items"}

    if band not in {"octave", "third"}:
        raise ValueError("band must be 'octave' or 'third'")
    if smoothing not in {"none", "median"}:
        raise ValueError("smoothing must be 'none' or 'median'")

    bands = make_octave_bands(band=band, fmin=fmin, fmax=fmax)
    if not bands:
        raise ValueError("No octave bands available for selected range")
    centers = [b["fc_hz"] for b in bands]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)

    for p in paths:
        freqs, power, _meta = compute_power_spectrum(file_path=_safe_path(p))
        levels = band_levels_db(freqs=freqs, power=power, bands=bands)
        levels = smooth_levels(levels, smoothing=smoothing)
        ax.plot(centers, levels.tolist(), label=Path(p).name)

    ax.set_xscale("log")
    ax.set_xlabel("Center Frequency (Hz)")
    ax.set_ylabel("Level (dB)")
    ax.set_title("Octave Compare (Overlay)")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(fontsize=8)

    png_b64 = fig_to_base64_png(fig)
    return {
        "png_base64": png_b64,
        "meta": {
            "paths": [Path(p).name for p in paths],
            "band": band,
            "fmin": float(fmin),
            "fmax": float(fmax),
            "smoothing": smoothing,
            "num_bands": len(centers),
        },
    }


@mcp.tool()
def generate_sample_fan_noise_logs(
    output_dir: str = "./fan_noise_samples",
    points: int = 240,
    step_sec: float = 1.0,
) -> dict:
    """팬 노이즈 샘플 로그 2개(csv)를 생성한다."""
    out_dir = resolve_in_workspace(output_dir, must_exist=False)
    paths = generate_sample_logs_pair(out_dir, points=points, step_sec=step_sec)
    logs = [load_noise_csv(p) for p in paths]
    return {
        "ok": True,
        "output_dir": str(out_dir.relative_to(WORKSPACE_ROOT)),
        "paths": [str(p.relative_to(WORKSPACE_ROOT)) for p in paths],
        "summaries": [summarize_noise_series(x) for x in logs],
    }


@mcp.tool()
def list_fan_noise_logs(pattern: str = "./fan_noise_samples/*.csv") -> dict:
    """워크스페이스 내 팬 노이즈 csv 로그 목록 조회."""
    full_pattern = str(resolve_in_workspace(pattern, must_exist=False))
    matches = sorted(glob.glob(full_pattern))
    logs = []
    for m in matches[:500]:
        p = Path(m).resolve()
        if not _is_within(p, WORKSPACE_ROOT):
            continue
        logs.append(str(p.relative_to(WORKSPACE_ROOT)))
    return {"ok": True, "count": len(logs), "matches": logs}


@mcp.tool()
def read_fan_noise_log(log_path: str, smoothing_window: int = 1, max_points: int = 3000) -> dict:
    """팬 노이즈 로그 단일 파일 조회 + 요약."""
    p = resolve_in_workspace(log_path)
    if not p.exists():
        return {"ok": False, "error": "not_found", "path": str(p)}

    log = load_noise_csv(p)
    times = log["time_sec"]
    noise = log["noise_db"]
    if len(times) > max_points:
        step = max(1, len(times) // max_points)
        times = times[::step]
        noise = noise[::step]

    if smoothing_window > 1:
        from src.fan_noise.core import moving_average

        noise = moving_average(noise, smoothing_window)

    return {
        "ok": True,
        "path": str(p.relative_to(WORKSPACE_ROOT)),
        "summary": summarize_noise_series(log),
        "series": {"time_sec": times, "noise_db": noise},
    }


@mcp.tool()
def compare_fan_noise_logs(
    log_a_path: str,
    log_b_path: str,
    smoothing_window: int = 1,
    max_points: int = 3000,
) -> dict:
    """
    x축=시간(sec), y축=소음 레벨(dB) 기준으로 두 로그를 비교하고 플롯 PNG(base64)를 반환한다.
    """
    pa = resolve_in_workspace(log_a_path)
    pb = resolve_in_workspace(log_b_path)
    if not pa.exists() or not pb.exists():
        return {
            "ok": False,
            "error": "log_not_found",
            "log_a_path": str(pa),
            "log_b_path": str(pb),
        }

    log_a = load_noise_csv(pa)
    log_b = load_noise_csv(pb)
    compared = compare_logs(log_a, log_b, smoothing_window=smoothing_window, max_points=max_points)
    series = compared["series"]
    png = render_compare_png_base64(
        times=series["time_sec"],
        noise_a=series["noise_db_a"],
        noise_b=series["noise_db_b"],
        delta=series["delta_db_b_minus_a"],
        label_a=pa.name,
        label_b=pb.name,
    )

    return {
        "ok": True,
        "log_a": {
            "path": str(pa.relative_to(WORKSPACE_ROOT)),
            "summary": summarize_noise_series(log_a),
        },
        "log_b": {
            "path": str(pb.relative_to(WORKSPACE_ROOT)),
            "summary": summarize_noise_series(log_b),
        },
        "comparison": compared["metrics"],
        "series": series,
        "plot_png_base64": png,
        "meta": {
            "aligned_points": compared["aligned_points"],
            "smoothing_window": compared["smoothing_window"],
            "x_axis": "time_sec",
            "y_axis": "noise_db",
        },
    }


if __name__ == "__main__":
    mcp.run()
