from __future__ import annotations

import base64
import csv
import io
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _build_profile(profile: str, idx: int) -> tuple[float, float, float]:
    t = float(idx)
    if profile == "a":
        # 상대적으로 조용한 패턴
        noise_db = 35.0 + 5.0 * math.sin(t / 16.0) + 1.2 * math.sin(t / 5.5)
        rpm = 1750.0 + 280.0 * math.sin(t / 20.0) + 65.0 * math.sin(t / 6.0)
        temp_c = 44.0 + 4.0 * math.sin(t / 24.0) + 1.0 * math.sin(t / 9.0)
    else:
        # 부하가 큰/noisy 패턴
        burst = 3.0 if (idx % 45) in (8, 9, 10, 11, 12) else 0.0
        noise_db = 40.0 + 6.6 * math.sin(t / 15.0) + 1.7 * math.sin(t / 4.8) + burst
        rpm = 2280.0 + 420.0 * math.sin(t / 18.0) + 95.0 * math.sin(t / 5.0) + burst * 85.0
        temp_c = 49.0 + 5.4 * math.sin(t / 22.0) + 1.2 * math.sin(t / 7.0) + burst * 0.35

    noise_db = _clip(noise_db, 25.0, 80.0)
    rpm = _clip(rpm, 900.0, 6200.0)
    temp_c = _clip(temp_c, 25.0, 100.0)
    return (round(noise_db, 3), round(rpm, 2), round(temp_c, 3))


def generate_sample_logs_pair(
    output_dir: Path,
    points: int = 240,
    step_sec: float = 1.0,
) -> list[Path]:
    if points < 10 or points > 20000:
        raise ValueError("points must be in range [10, 20000]")
    if step_sec <= 0:
        raise ValueError("step_sec must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "fan_noise_sample_a.csv",
        output_dir / "fan_noise_sample_b.csv",
    ]

    for path, profile in zip(paths, ("a", "b")):
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["time_sec", "noise_db", "fan_rpm", "temperature_c"])
            for i in range(points):
                time_sec = round(i * step_sec, 6)
                noise_db, rpm, temp_c = _build_profile(profile, i)
                writer.writerow([time_sec, noise_db, rpm, temp_c])
    return paths


def load_noise_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"log file not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"log path is directory: {path}")

    times: list[float] = []
    noise: list[float] = []
    rpm: list[float] = []
    temp: list[float] = []

    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        required = {"time_sec", "noise_db"}
        cols = set(reader.fieldnames or [])
        if not required.issubset(cols):
            raise ValueError("CSV must include columns: time_sec, noise_db")

        for row in reader:
            try:
                t = float(row.get("time_sec", ""))
                n = float(row.get("noise_db", ""))
            except ValueError:
                continue
            times.append(t)
            noise.append(n)

            rpm_raw = row.get("fan_rpm", "")
            temp_raw = row.get("temperature_c", "")
            try:
                rpm.append(float(rpm_raw))
            except (TypeError, ValueError):
                pass
            try:
                temp.append(float(temp_raw))
            except (TypeError, ValueError):
                pass

    if not times:
        raise ValueError(f"no valid rows in {path}")

    return {
        "path": str(path),
        "time_sec": times,
        "noise_db": noise,
        "fan_rpm": rpm,
        "temperature_c": temp,
    }


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    if window > len(values):
        return [float(mean(values))] * len(values)

    kernel = np.ones(window, dtype=float) / float(window)
    arr = np.array(values, dtype=float)
    smoothed = np.convolve(arr, kernel, mode="same")
    return smoothed.tolist()


def summarize_noise_series(log: dict[str, Any]) -> dict[str, Any]:
    noise = log.get("noise_db") or []
    if not noise:
        return {"points": 0}
    arr = np.array(noise, dtype=float)
    return {
        "points": int(arr.size),
        "duration_sec": round(float(log["time_sec"][-1] - log["time_sec"][0]), 4) if arr.size > 1 else 0.0,
        "noise_db_avg": round(float(arr.mean()), 4),
        "noise_db_min": round(float(arr.min()), 4),
        "noise_db_max": round(float(arr.max()), 4),
        "noise_db_p95": round(float(np.percentile(arr, 95)), 4),
    }


def _align_for_compare(log_a: dict[str, Any], log_b: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    time_a = [float(x) for x in log_a["time_sec"]]
    time_b = [float(x) for x in log_b["time_sec"]]
    noise_a = [float(x) for x in log_a["noise_db"]]
    noise_b = [float(x) for x in log_b["noise_db"]]

    map_a = {round(t, 6): v for t, v in zip(time_a, noise_a)}
    map_b = {round(t, 6): v for t, v in zip(time_b, noise_b)}
    common = sorted(set(map_a.keys()).intersection(map_b.keys()))
    if common:
        times = [float(t) for t in common]
        a_vals = [float(map_a[t]) for t in common]
        b_vals = [float(map_b[t]) for t in common]
        return times, a_vals, b_vals

    n = min(len(time_a), len(time_b))
    if n <= 0:
        return [], [], []
    return time_a[:n], noise_a[:n], noise_b[:n]


def compare_logs(
    log_a: dict[str, Any],
    log_b: dict[str, Any],
    smoothing_window: int = 1,
    max_points: int = 5000,
) -> dict[str, Any]:
    if smoothing_window < 1 or smoothing_window > 999:
        raise ValueError("smoothing_window must be in range [1, 999]")
    if max_points < 50 or max_points > 20000:
        raise ValueError("max_points must be in range [50, 20000]")

    times, a_vals, b_vals = _align_for_compare(log_a, log_b)
    if not times:
        raise ValueError("no overlapping samples between two logs")

    if len(times) > max_points:
        step = max(1, len(times) // max_points)
        times = times[::step]
        a_vals = a_vals[::step]
        b_vals = b_vals[::step]

    a_s = moving_average(a_vals, smoothing_window)
    b_s = moving_average(b_vals, smoothing_window)
    delta = [b - a for a, b in zip(a_s, b_s)]

    arr_d = np.array(delta, dtype=float)
    avg_delta = float(arr_d.mean())
    loud_a = float(np.mean(a_s))
    loud_b = float(np.mean(b_s))

    return {
        "aligned_points": len(times),
        "smoothing_window": smoothing_window,
        "series": {
            "time_sec": times,
            "noise_db_a": a_s,
            "noise_db_b": b_s,
            "delta_db_b_minus_a": delta,
        },
        "metrics": {
            "avg_delta_db_b_minus_a": round(avg_delta, 4),
            "max_delta_db_b_minus_a": round(float(arr_d.max()), 4),
            "min_delta_db_b_minus_a": round(float(arr_d.min()), 4),
            "rmse_delta_db": round(float(math.sqrt(np.mean(arr_d ** 2))), 4),
            "louder_log": "b" if loud_b > loud_a else "a",
            "loudness_gap_db": round(abs(loud_b - loud_a), 4),
        },
    }


def render_compare_png_base64(
    times: list[float],
    noise_a: list[float],
    noise_b: list[float],
    delta: list[float],
    label_a: str,
    label_b: str,
) -> str:
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    ax1.plot(times, noise_a, label=label_a, linewidth=1.3)
    ax1.plot(times, noise_b, label=label_b, linewidth=1.3)
    ax1.set_ylabel("Noise Level (dB)")
    ax1.set_title("Fan Noise Log Compare")
    ax1.grid(True, ls=":", lw=0.6)
    ax1.legend(fontsize=8)

    ax2.plot(times, delta, color="tab:red", linewidth=1.1)
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax2.set_xlabel("Time (sec)")
    ax2.set_ylabel("Delta dB\n(B - A)")
    ax2.grid(True, ls=":", lw=0.6)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
