from __future__ import annotations

import glob
import os
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fastmcp import FastMCP

from src.fan_noise import (
    compare_logs,
    generate_sample_logs_pair,
    load_noise_csv,
    render_compare_png_base64,
    summarize_noise_series,
)
from src.fan_noise.core import moving_average


mcp = FastMCP("thermal-fan-noise-server")
WORKSPACE_ROOT = Path(os.environ.get("MCP_WORKSPACE", os.getcwd())).resolve()
print(f"[THERMAL_MCP] WORKSPACE_ROOT = {WORKSPACE_ROOT}")


def _is_within(child: Path, root: Path) -> bool:
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_in_workspace(p: str) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    path = path.resolve()

    if not _is_within(path, WORKSPACE_ROOT):
        raise PermissionError(f"Access denied: {path} is outside workspace {WORKSPACE_ROOT}")
    return path


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _summarize_temperature(log: dict[str, Any]) -> dict[str, Any]:
    temps = log.get("temperature_c") or []
    if not temps:
        return {"points": 0}
    arr = np.array(temps, dtype=float)
    times = log.get("time_sec") or []
    duration = 0.0
    if isinstance(times, list) and len(times) >= 2:
        duration = round(float(times[-1] - times[0]), 4)
    return {
        "points": int(arr.size),
        "duration_sec": duration,
        "temperature_c_avg": round(float(arr.mean()), 4),
        "temperature_c_min": round(float(arr.min()), 4),
        "temperature_c_max": round(float(arr.max()), 4),
        "temperature_c_p95": round(float(np.percentile(arr, 95)), 4),
    }


def _align_series_for_compare(
    times_a: list[float],
    values_a: list[float],
    times_b: list[float],
    values_b: list[float],
) -> tuple[list[float], list[float], list[float]]:
    map_a = {round(float(t), 6): float(v) for t, v in zip(times_a, values_a)}
    map_b = {round(float(t), 6): float(v) for t, v in zip(times_b, values_b)}
    common = sorted(set(map_a.keys()).intersection(map_b.keys()))
    if common:
        times = [float(t) for t in common]
        a_vals = [float(map_a[t]) for t in common]
        b_vals = [float(map_b[t]) for t in common]
        return times, a_vals, b_vals

    n = min(len(times_a), len(times_b), len(values_a), len(values_b))
    if n <= 0:
        return [], [], []
    return [float(x) for x in times_a[:n]], [float(x) for x in values_a[:n]], [float(x) for x in values_b[:n]]


def _compare_generic_series(
    times_a: list[float],
    values_a: list[float],
    times_b: list[float],
    values_b: list[float],
    smoothing_window: int,
    max_points: int,
) -> dict[str, Any]:
    if smoothing_window < 1 or smoothing_window > 999:
        raise ValueError("smoothing_window must be in range [1, 999]")
    if max_points < 50 or max_points > 20000:
        raise ValueError("max_points must be in range [50, 20000]")

    times, a_vals, b_vals = _align_series_for_compare(times_a, values_a, times_b, values_b)
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
    avg_a = float(np.mean(a_s))
    avg_b = float(np.mean(b_s))

    return {
        "aligned_points": len(times),
        "smoothing_window": smoothing_window,
        "series": {
            "time_sec": times,
            "value_a": a_s,
            "value_b": b_s,
            "delta_b_minus_a": delta,
        },
        "metrics": {
            "avg_delta_b_minus_a": round(avg_delta, 4),
            "max_delta_b_minus_a": round(float(arr_d.max()), 4),
            "min_delta_b_minus_a": round(float(arr_d.min()), 4),
            "rmse_delta": round(float(math.sqrt(np.mean(arr_d ** 2))), 4),
            "higher_avg_log": "b" if avg_b > avg_a else "a",
            "avg_gap": round(abs(avg_b - avg_a), 4),
        },
    }


def _render_generic_compare_png_base64(
    times: list[float],
    values_a: list[float],
    values_b: list[float],
    delta: list[float],
    label_a: str,
    label_b: str,
    y_label: str,
    title: str,
    delta_label: str,
) -> str:
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    ax1.plot(times, values_a, label=label_a, linewidth=1.3)
    ax1.plot(times, values_b, label=label_b, linewidth=1.3)
    ax1.set_ylabel(y_label)
    ax1.set_title(title)
    ax1.grid(True, ls=":", lw=0.6)
    ax1.legend(fontsize=8)

    ax2.plot(times, delta, color="tab:red", linewidth=1.1)
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax2.set_xlabel("Time (sec)")
    ax2.set_ylabel(delta_label)
    ax2.grid(True, ls=":", lw=0.6)

    from io import BytesIO
    import base64

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@mcp.tool()
def ping() -> dict[str, Any]:
    """서버 상태 체크."""
    return {"ok": True, "server": "thermal-fan-noise-server", "workspace_root": str(WORKSPACE_ROOT)}


@mcp.tool()
def thermal_snapshot() -> dict[str, Any]:
    """시스템 thermal zone 온도 스냅샷을 반환한다."""
    root = Path("/sys/class/thermal")
    zones = []
    for zone in sorted(root.glob("thermal_zone*")):
        raw = _read_text(zone / "temp")
        t_c = None
        if raw:
            try:
                v = float(raw)
                t_c = round(v / 1000.0, 3) if abs(v) > 200 else round(v, 3)
            except ValueError:
                t_c = None
        zones.append(
            {
                "zone": zone.name,
                "type": _read_text(zone / "type"),
                "temp_c": t_c,
            }
        )

    valid = [z["temp_c"] for z in zones if isinstance(z.get("temp_c"), (int, float))]
    return {
        "ok": True,
        "count": len(zones),
        "zones": zones,
        "max_temp_c": max(valid) if valid else None,
        "min_temp_c": min(valid) if valid else None,
        "avg_temp_c": round(sum(valid) / len(valid), 3) if valid else None,
    }


@mcp.tool()
def generate_sample_fan_noise_logs(
    output_dir: str = "./fan_noise_samples",
    points: int = 240,
    step_sec: float = 1.0,
) -> dict[str, Any]:
    """팬 노이즈 샘플 로그 2개(csv)를 생성한다."""
    out_dir = resolve_in_workspace(output_dir)
    paths = generate_sample_logs_pair(out_dir, points=points, step_sec=step_sec)

    logs = [load_noise_csv(p) for p in paths]
    return {
        "ok": True,
        "output_dir": str(out_dir.relative_to(WORKSPACE_ROOT)),
        "paths": [str(p.relative_to(WORKSPACE_ROOT)) for p in paths],
        "summaries": [summarize_noise_series(x) for x in logs],
    }


@mcp.tool()
def list_fan_noise_logs(pattern: str = "./fan_noise_samples/*.csv") -> dict[str, Any]:
    """워크스페이스 내 팬 노이즈 csv 로그 목록 조회."""
    full_pattern = str(resolve_in_workspace(pattern))
    matches = sorted(glob.glob(full_pattern))

    logs = []
    for m in matches[:500]:
        p = Path(m).resolve()
        if not _is_within(p, WORKSPACE_ROOT):
            continue
        logs.append(str(p.relative_to(WORKSPACE_ROOT)))
    return {"ok": True, "count": len(logs), "matches": logs}


@mcp.tool()
def read_fan_noise_log(
    log_path: str,
    signal: str = "noise",  # noise | temperature | both
    smoothing_window: int = 1,
    max_points: int = 3000,
) -> dict[str, Any]:
    """팬/온도 로그 단일 파일 조회 + 요약."""
    p = resolve_in_workspace(log_path)
    if not p.exists():
        return {"ok": False, "error": "not_found", "path": str(p)}
    if signal not in {"noise", "temperature", "both"}:
        return {"ok": False, "error": "invalid_signal", "signal": signal}

    log = load_noise_csv(p)
    times = log["time_sec"]
    noise = log.get("noise_db", [])
    temp = log.get("temperature_c", [])

    if signal in {"temperature", "both"} and (not temp or len(temp) != len(times)):
        return {"ok": False, "error": "temperature_series_not_available", "path": str(p.relative_to(WORKSPACE_ROOT))}

    step = 1
    if len(times) > max_points:
        step = max(1, len(times) // max_points)
    times_ds = times[::step]
    noise_ds = noise[::step] if noise else []
    temp_ds = temp[::step] if temp else []

    if smoothing_window > 1:
        if noise_ds:
            noise_ds = moving_average(noise_ds, smoothing_window)
        if temp_ds:
            temp_ds = moving_average(temp_ds, smoothing_window)

    series: dict[str, Any] = {"time_sec": times_ds}
    summary: dict[str, Any] = {}
    if signal in {"noise", "both"}:
        series["noise_db"] = noise_ds
        summary["noise"] = summarize_noise_series(log)
    if signal in {"temperature", "both"}:
        series["temperature_c"] = temp_ds
        summary["temperature"] = _summarize_temperature(log)

    return {
        "ok": True,
        "path": str(p.relative_to(WORKSPACE_ROOT)),
        "signal": signal,
        "summary": summary,
        "series": series,
        "meta": {
            "x_axis": "time_sec",
            "available_signals": [k for k in ("noise", "temperature") if k in summary],
            "smoothing_window": int(smoothing_window),
        },
    }


@mcp.tool()
def compare_fan_noise_logs(
    log_a_path: str,
    log_b_path: str,
    signal: str = "noise",  # noise | temperature | both
    smoothing_window: int = 1,
    max_points: int = 3000,
) -> dict[str, Any]:
    """
    x축=시간(sec), y축=소음/온도 기준으로 두 로그를 비교하고 플롯 PNG(base64)를 반환한다.
    """
    if signal not in {"noise", "temperature", "both"}:
        return {"ok": False, "error": "invalid_signal", "signal": signal}

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

    def _build_noise() -> dict[str, Any]:
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
            "comparison": compared["metrics"],
            "series": series,
            "plot_png_base64": png,
            "meta": {
                "aligned_points": compared["aligned_points"],
                "smoothing_window": compared["smoothing_window"],
                "x_axis": "time_sec",
                "y_axis": "noise_db",
                "signal": "noise",
            },
        }

    def _build_temperature() -> dict[str, Any]:
        t_a = log_a.get("temperature_c") or []
        t_b = log_b.get("temperature_c") or []
        if not t_a or not t_b:
            return {"ok": False, "error": "temperature_series_not_available"}
        if len(t_a) != len(log_a["time_sec"]) or len(t_b) != len(log_b["time_sec"]):
            return {"ok": False, "error": "temperature_series_length_mismatch"}

        compared = _compare_generic_series(
            times_a=log_a["time_sec"],
            values_a=t_a,
            times_b=log_b["time_sec"],
            values_b=t_b,
            smoothing_window=smoothing_window,
            max_points=max_points,
        )
        series = compared["series"]
        png = _render_generic_compare_png_base64(
            times=series["time_sec"],
            values_a=series["value_a"],
            values_b=series["value_b"],
            delta=series["delta_b_minus_a"],
            label_a=pa.name,
            label_b=pb.name,
            y_label="Temperature (C)",
            title="Temperature Log Compare",
            delta_label="Delta C\n(B - A)",
        )
        return {
            "comparison": compared["metrics"],
            "series": {
                "time_sec": series["time_sec"],
                "temperature_c_a": series["value_a"],
                "temperature_c_b": series["value_b"],
                "delta_c_b_minus_a": series["delta_b_minus_a"],
            },
            "plot_png_base64": png,
            "meta": {
                "aligned_points": compared["aligned_points"],
                "smoothing_window": compared["smoothing_window"],
                "x_axis": "time_sec",
                "y_axis": "temperature_c",
                "signal": "temperature",
            },
        }

    log_a_summary = {
        "noise": summarize_noise_series(log_a),
        "temperature": _summarize_temperature(log_a),
    }
    log_b_summary = {
        "noise": summarize_noise_series(log_b),
        "temperature": _summarize_temperature(log_b),
    }

    if signal == "noise":
        noise_res = _build_noise()
        return {
            "ok": True,
            "signal": "noise",
            "log_a": {"path": str(pa.relative_to(WORKSPACE_ROOT)), "summary": log_a_summary["noise"]},
            "log_b": {"path": str(pb.relative_to(WORKSPACE_ROOT)), "summary": log_b_summary["noise"]},
            **noise_res,
        }

    if signal == "temperature":
        temp_res = _build_temperature()
        if not temp_res.get("comparison"):
            return {
                "ok": False,
                "signal": "temperature",
                "error": temp_res.get("error", "temperature_compare_failed"),
            }
        return {
            "ok": True,
            "signal": "temperature",
            "log_a": {"path": str(pa.relative_to(WORKSPACE_ROOT)), "summary": log_a_summary["temperature"]},
            "log_b": {"path": str(pb.relative_to(WORKSPACE_ROOT)), "summary": log_b_summary["temperature"]},
            **temp_res,
        }

    noise_res = _build_noise()
    temp_res = _build_temperature()
    if not temp_res.get("comparison"):
        return {
            "ok": False,
            "signal": "both",
            "error": temp_res.get("error", "temperature_compare_failed"),
        }
    return {
        "ok": True,
        "signal": "both",
        "log_a": {"path": str(pa.relative_to(WORKSPACE_ROOT)), "summary": log_a_summary},
        "log_b": {"path": str(pb.relative_to(WORKSPACE_ROOT)), "summary": log_b_summary},
        "results": {
            "noise": noise_res,
            "temperature": temp_res,
        },
        "meta": {
            "x_axis": "time_sec",
            "available_signals": ["noise", "temperature"],
            "smoothing_window": int(smoothing_window),
        },
    }


if __name__ == "__main__":
    mcp.run()
