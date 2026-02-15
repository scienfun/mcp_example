from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import ollama
import streamlit as st

from src.fan_noise import generate_sample_logs_pair

DEFAULT_MCP_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
DEFAULT_MCP_SCRIPT = os.environ.get("MCP_SERVER_SCRIPT", "audio_mcp_server.py")
_DEFAULT_MCP_TRANSPORT_MODE_RAW = os.environ.get("MCP_TRANSPORT_MODE", "stdio").strip().lower()
DEFAULT_MCP_TRANSPORT_MODE = _DEFAULT_MCP_TRANSPORT_MODE_RAW if _DEFAULT_MCP_TRANSPORT_MODE_RAW in {"http", "stdio"} else "stdio"
DEFAULT_THERMAL_MCP_URL = os.environ.get("THERMAL_MCP_SERVER_URL", "http://127.0.0.1:8101/mcp")
DEFAULT_THERMAL_MCP_SCRIPT = os.environ.get("THERMAL_MCP_SCRIPT", "thermal_mcp_server.py")
_DEFAULT_THERMAL_MCP_TRANSPORT_MODE_RAW = os.environ.get("THERMAL_MCP_TRANSPORT_MODE", "stdio").strip().lower()
DEFAULT_THERMAL_MCP_TRANSPORT_MODE = (
    _DEFAULT_THERMAL_MCP_TRANSPORT_MODE_RAW if _DEFAULT_THERMAL_MCP_TRANSPORT_MODE_RAW in {"http", "stdio"} else "stdio"
)
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "glm-4.7:cloud")
APP_DIR = Path(__file__).resolve().parent
DEFAULT_FAN_SAMPLE_DIR = APP_DIR / "fan_noise_samples"
DEFAULT_FAN_SAMPLE_A = DEFAULT_FAN_SAMPLE_DIR / "fan_noise_sample_a.csv"
DEFAULT_FAN_SAMPLE_B = DEFAULT_FAN_SAMPLE_DIR / "fan_noise_sample_b.csv"
DEFAULT_FAN_SOURCE = os.environ.get("FAN_MCP_SOURCE", "auto").strip().lower()
if DEFAULT_FAN_SOURCE not in {"auto", "audio", "thermal", "both"}:
    DEFAULT_FAN_SOURCE = "auto"


# -----------------------------
# Helpers: parsing MCP payloads
# -----------------------------
def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def extract_tool_payload(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result

    if isinstance(result, str):
        parsed = _try_parse_json(result)
        if parsed is not None:
            return parsed

    if hasattr(result, "model_dump"):
        dumped = result.model_dump()
        if isinstance(dumped, dict):
            for key in ("data", "structured_content", "structuredContent"):
                if isinstance(dumped.get(key), dict):
                    return dumped[key]
            content = dumped.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("json"), dict):
                        return item["json"]
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parsed = _try_parse_json(item["text"])
                        if parsed is not None:
                            return parsed
            return dumped

    for key in ("data", "structured_content", "structuredContent"):
        value = getattr(result, key, None)
        if isinstance(value, dict):
            return value

    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("json"), dict):
                return item["json"]
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parsed = _try_parse_json(item["text"])
                if parsed is not None:
                    return parsed

    raise ValueError(f"Could not parse MCP tool result: {type(result)}")


def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "asyncio.run() cannot be called" not in str(exc):
            raise
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def render_base64_png(png_base64: str, caption: str, use_container_width: bool = False, width: Optional[int] = None) -> None:
    image_bytes = base64.b64decode(png_base64.encode("utf-8"))
    st.image(image_bytes, caption=caption, use_container_width=use_container_width, width=width)


def resolve_mcp_transport(mode: str, mcp_url: str, mcp_script: str) -> str:
    transport_mode = (mode or "").strip().lower()
    if transport_mode == "stdio":
        script_path = Path(mcp_script).expanduser()
        if not script_path.is_absolute():
            script_path = (APP_DIR / script_path).resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"stdio script not found: {script_path}")
        return str(script_path)

    return str(mcp_url).strip()


def ensure_default_fan_samples() -> tuple[bool, str]:
    try:
        if DEFAULT_FAN_SAMPLE_A.exists() and DEFAULT_FAN_SAMPLE_B.exists():
            return True, "ready"
        generate_sample_logs_pair(DEFAULT_FAN_SAMPLE_DIR, points=240, step_sec=1.0)
        return True, "generated"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def call_tool_once(server_transport: str, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    from fastmcp import Client

    async with Client(server_transport) as client:
        raw = await client.call_tool(tool_name, args)
    payload = extract_tool_payload(raw)
    if isinstance(payload, dict):
        payload.setdefault("ok", True)
    return payload


def safe_call_tool(server_transport: str, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return run_async(call_tool_once(server_transport, tool_name, args))
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "tool": tool_name}


async def call_tool_dual(
    audio_transport: str,
    thermal_transport: str,
    tool_name: str,
    args: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    audio_task = call_tool_once(audio_transport, tool_name, args)
    thermal_task = call_tool_once(thermal_transport, tool_name, args)
    audio_res, thermal_res = await asyncio.gather(audio_task, thermal_task, return_exceptions=True)

    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(audio_res, Exception):
        out["audio"] = {"ok": False, "error": f"{type(audio_res).__name__}: {audio_res}", "tool": tool_name}
    else:
        out["audio"] = audio_res

    if isinstance(thermal_res, Exception):
        out["thermal"] = {"ok": False, "error": f"{type(thermal_res).__name__}: {thermal_res}", "tool": tool_name}
    else:
        out["thermal"] = thermal_res
    return out


# -----------------------------
# Robust JSON extraction (code fences ok)
# -----------------------------
def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    cleaned = strip_code_fences(text)
    parsed = _try_parse_json(cleaned)
    if parsed is not None:
        return parsed
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        return None
    return _try_parse_json(m.group(0))


# -----------------------------
# Natural language -> plan JSON
# -----------------------------
def llm_parse_command(nl_text: str, model: str) -> Dict[str, Any]:
    """
    핵심 변경:
      - meas_path 단일 -> inputs: [..] 다중 파일
      - compare_mode: "side_by_side" | "overlay"
    """
    schema_hint = {
        "action": "analyze",
        "inputs": ["./test.wav", "./test2.wav"],
        "ref_path": None,               # (선택) 옥타브/응답 기준 참조파일 1개만 둘 때
        "compare_mode": "side_by_side", # "side_by_side" | "overlay"

        "run_spectrogram": False,       # 비교는 보통 spectrogram은 side_by_side만 권장
        "run_fft": True,
        "run_octave": False,

        "n_fft": 2048,
        "hop_length": 512,
        "fmin": 20,
        "fmax": 20000,

        "band": "third",                # octave/third
        "smoothing": "none",

        "fft_window_sec": 1.0,
        "fft_scale": "db",              # db|linear
    }

    system = (
        "너는 오디오 분석 앱의 명령 해석기다.\n"
        "사용자의 자연어 요청을 아래 JSON 스키마로만 변환해라.\n"
        "다른 텍스트/코드펜스(```)/주석은 절대 출력하지 마라.\n"
        "\n"
        "규칙:\n"
        "- 사용자가 언급한 오디오 파일 경로들을 inputs 배열에 모두 넣어라.\n"
        "- 파일이 2개 이상이면 기본 compare_mode는 'overlay'로 해라(사용자가 '나란히'라고 하면 side_by_side).\n"
        "- 파일이 1개면 compare_mode는 'side_by_side'로 두어도 무방.\n"
        "- 사용자가 '스펙트로그램'을 말하면 run_spectrogram=true.\n"
        "- 사용자가 'fft'/'스펙트럼'을 말하면 run_fft=true.\n"
        "- 사용자가 '주파수 응답'/'옥타브'를 말하면 run_octave=true.\n"
        "- band: '1옥타브'면 octave, '1/3옥타브'면 third(기본 third).\n"
        "- 주파수 범위를 말하면 fmin/fmax 반영.\n"
        "- fft 스케일: 'dB'면 db, '선형'이면 linear(기본 db).\n"
    )

    user = (
        f"요청: {nl_text}\n\n"
        f"출력 JSON 예시(형태만 참고):\n{json.dumps(schema_hint, ensure_ascii=False, indent=2)}"
    )

    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0},
    )

    content = resp.get("message", {}).get("content", "").strip()
    parsed = parse_llm_json(content)
    if not parsed:
        raise ValueError(f"LLM이 JSON을 반환하지 않았습니다. 응답 일부: {content[:400]}")
    return parsed


def infer_domain_heuristic(nl_text: str) -> str:
    text = (nl_text or "").lower()
    fan_keywords = ("fan", "팬", "소음", "노이즈", "thermal", "rpm", "log", "로그", ".csv", "db")
    audio_keywords = ("audio", "오디오", "fft", "스펙트로", "옥타브", ".wav", ".mp3", ".m4a")
    fan_score = sum(1 for k in fan_keywords if k in text)
    audio_score = sum(1 for k in audio_keywords if k in text)
    return "fan_noise" if fan_score > audio_score else "audio"


def _to_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    return max(lo, min(hi, out))


def _to_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    return max(lo, min(hi, out))


def _to_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return []
        if "," in v:
            return [x.strip() for x in v.split(",") if x.strip()]
        return [v]
    return []


def extract_csv_paths(nl_text: str) -> List[str]:
    text = nl_text or ""

    def _clean_path(p: str) -> str:
        p2 = p.strip().strip(",")
        p2 = re.sub(r"^(?:와|과|및|and)\s+", "", p2, flags=re.IGNORECASE)
        return p2

    # 따옴표로 감싼 경로 우선
    quoted = re.findall(r"[\"']([^\"']+?\.csv)[\"']", text, flags=re.IGNORECASE)
    out: List[str] = []
    for p in quoted:
        p2 = _clean_path(p)
        if p2 and p2 not in out:
            out.append(p2)
    # 일반 패턴
    plain = re.findall(r"([\w./~\-]+\.csv)", text, flags=re.IGNORECASE)
    for p in plain:
        p2 = _clean_path(p)
        if p2 and p2 not in out:
            out.append(p2)
    return out


def resolve_fan_signal(requested: str, nl_text: str) -> str:
    req = (requested or "").strip().lower()
    if req in {"noise", "temperature", "both"}:
        return req

    text = (nl_text or "").lower()
    if re.search(r"(온도만|temperature만|temp만|temperature\s*only|temp\s*only)", text):
        return "temperature"
    if re.search(r"(소음만|노이즈만|noise만|noise\s*only)", text):
        return "noise"

    metric_both_kw = ("온도와 소음", "소음과 온도", "temperature and noise", "noise and temperature")
    has_temp = bool(re.search(r"(temperature|temp|온도)", text))
    has_noise = bool(re.search(r"(\bnoise\b|소음|노이즈|\bdb\b)", text))

    if any(k in text for k in metric_both_kw):
        return "both"
    if has_temp and has_noise:
        return "both"
    if has_temp:
        return "temperature"
    if has_noise:
        return "noise"
    return "noise"


def resolve_fan_source(requested: str, nl_text: str, signal: str) -> str:
    req = (requested or "").strip().lower()
    if req in {"audio", "thermal", "both"}:
        if req == "audio" and signal in {"temperature", "both"}:
            # audio MCP는 noise 중심이라 temperature 요청을 수용할 수 없다.
            return "thermal"
        if req == "both" and signal in {"temperature", "both"}:
            # temperature가 포함되면 thermal 단독으로 응답해 강제 dual 출력을 피한다.
            return "thermal"
        return req

    text = (nl_text or "").lower()
    both_kw = ("both", "둘다", "둘 다", "동시", "같이", "audio와 thermal")
    if any(k in text for k in both_kw) and signal == "noise":
        return "both"

    has_audio = ("audio" in text) or ("오디오" in text)
    has_thermal = ("thermal" in text) or ("써멀" in text) or ("온도" in text) or ("열" in text)
    if has_audio and not has_thermal:
        return "audio"
    if has_thermal and not has_audio:
        return "thermal"

    # fan noise 기본은 thermal 단독으로 둬서 결과를 강제 병합하지 않는다.
    return "thermal"


def normalize_fan_plan(raw: Dict[str, Any], nl_text: str = "") -> Dict[str, Any]:
    task = str(raw.get("task", "auto")).strip().lower()
    if task not in {"auto", "generate", "list", "read", "compare"}:
        task = "auto"

    source_requested = str(raw.get("source", DEFAULT_FAN_SOURCE)).strip().lower()
    if source_requested not in {"auto", "audio", "thermal", "both"}:
        source_requested = DEFAULT_FAN_SOURCE
    signal_requested = str(raw.get("signal", "auto")).strip().lower()
    if signal_requested not in {"auto", "noise", "temperature", "both"}:
        signal_requested = "auto"

    sample_dir = str(raw.get("sample_dir", "./fan_noise_samples")).strip() or "./fan_noise_samples"
    pattern = str(raw.get("pattern", f"{sample_dir.rstrip('/')}" + "/*.csv")).strip()

    log_paths = _to_str_list(raw.get("log_paths"))
    legacy_a = str(raw.get("log_a_path", "")).strip()
    legacy_b = str(raw.get("log_b_path", "")).strip()
    if legacy_a and legacy_a not in log_paths:
        log_paths.append(legacy_a)
    if legacy_b and legacy_b not in log_paths:
        log_paths.append(legacy_b)

    # 자연어의 CSV 경로를 LLM 출력과 합친다.
    for p in extract_csv_paths(nl_text):
        if p not in log_paths:
            log_paths.append(p)

    # 경로 미지정 시 기본 샘플 2개
    if not log_paths:
        log_paths = [
            "./fan_noise_samples/fan_noise_sample_a.csv",
            "./fan_noise_samples/fan_noise_sample_b.csv",
        ]
    else:
        normalized_paths: List[str] = []
        for p in log_paths:
            pp = p.strip()
            if "/" in pp or pp.startswith(".") or pp.startswith("~"):
                normalized_paths.append(pp)
            else:
                normalized_paths.append(f"{sample_dir.rstrip('/')}/{pp}")
        log_paths = normalized_paths

    points = _to_int(raw.get("points", 240), default=240, lo=60, hi=5000)
    step_sec = _to_float(raw.get("step_sec", 1.0), default=1.0, lo=0.1, hi=60.0)
    smoothing_window = _to_int(raw.get("smoothing_window", 3), default=3, lo=1, hi=61)
    if smoothing_window % 2 == 0:
        smoothing_window = max(1, smoothing_window - 1)
    max_points = _to_int(raw.get("max_points", 3000), default=3000, lo=200, hi=20000)

    # task 자동 추론
    if task == "auto":
        text = (nl_text or "").lower()
        if any(k in text for k in ("생성", "create", "generate", "만들어")):
            task = "generate"
        elif any(k in text for k in ("목록", "리스트", "list", "조회")):
            task = "list"
        else:
            task = "compare" if len(log_paths) >= 2 else "read"

    # compare 요청인데 파일이 1개면 단일 조회로 강등
    if task == "compare" and len(log_paths) < 2:
        task = "read"

    signal_effective = resolve_fan_signal(signal_requested, nl_text=nl_text)
    source_effective = resolve_fan_source(source_requested, nl_text=nl_text, signal=signal_effective)

    return {
        "task": task,
        "source_requested": source_requested,
        "signal_requested": signal_requested,
        "source": source_effective,
        "signal": signal_effective,
        "sample_dir": sample_dir,
        "pattern": pattern,
        "log_paths": log_paths[:2],
        "log_a_path": log_paths[0] if log_paths else "",
        "log_b_path": log_paths[1] if len(log_paths) > 1 else "",
        "points": points,
        "step_sec": step_sec,
        "smoothing_window": smoothing_window,
        "max_points": max_points,
    }


def llm_route_request(nl_text: str, model: str) -> Dict[str, Any]:
    schema_hint = {
        "domain": "audio",  # audio | fan_noise
        "audio_plan": {
            "inputs": ["./test.wav", "./test2.wav"],
            "compare_mode": "overlay",
            "run_spectrogram": False,
            "run_fft": True,
            "run_octave": False,
            "n_fft": 2048,
            "hop_length": 512,
            "fmin": 20,
            "fmax": 20000,
            "band": "third",
            "smoothing": "none",
            "fft_window_sec": 1.0,
            "fft_scale": "db",
        },
        "fan_plan": {
            "task": "auto",     # auto | read | compare | generate | list
            "source": "auto",   # auto | audio | thermal | both
            "signal": "auto",   # auto | noise | temperature | both
            "sample_dir": "./fan_noise_samples",
            "pattern": "./fan_noise_samples/*.csv",
            "log_paths": [
                "./fan_noise_samples/fan_noise_sample_a.csv",
                "./fan_noise_samples/fan_noise_sample_b.csv",
            ],
            "points": 240,
            "step_sec": 1.0,
            "smoothing_window": 3,
            "max_points": 3000,
        },
    }

    system = (
        "너는 audio/fan_noise 통합 앱의 명령 라우터다.\n"
        "반드시 JSON 하나만 출력해라. 코드펜스/주석/설명 금지.\n"
        "domain은 audio 또는 fan_noise 중 하나다.\n"
        "audio 질문이면 domain=audio, fan/thermal/noise/log/csv 질문이면 domain=fan_noise.\n"
        "fan_noise 규칙:\n"
        "- 파일 1개 분석이면 task=read, 파일 2개 비교면 task=compare\n"
        "- 생성 요청이면 task=generate, 목록/리스트 요청이면 task=list\n"
        "- task를 모르면 auto로 둬라\n"
        "- source는 auto가 기본이며, 사용자가 audio/thermal/both를 명시한 경우만 반영\n"
        "- signal은 noise/temperature/both/auto 중 하나로 설정\n"
        "- 온도 요청이면 signal=temperature, 소음 요청이면 signal=noise, 둘 다면 signal=both\n"
        "- 비교/조회에 사용할 파일은 log_paths 배열에 담아라(최대 2개)\n"
        "- 값이 없으면 기본 경로/기본 수치 사용\n"
        "audio 규칙:\n"
        "- 기존 오디오 플랜 키(inputs, run_fft, run_spectrogram, run_octave...)를 채워라.\n"
    )

    user = (
        f"요청: {nl_text}\n\n"
        f"출력 JSON 예시(형태만 참고):\n{json.dumps(schema_hint, ensure_ascii=False, indent=2)}"
    )

    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0},
    )

    content = resp.get("message", {}).get("content", "").strip()
    parsed = parse_llm_json(content)
    if not isinstance(parsed, dict):
        domain = infer_domain_heuristic(nl_text)
        if domain == "audio":
            return {"domain": "audio", "audio_plan": llm_parse_command(nl_text, model)}
        return {"domain": "fan_noise", "fan_plan": normalize_fan_plan({}, nl_text=nl_text)}

    domain = str(parsed.get("domain", "")).strip().lower()
    if domain not in {"audio", "fan_noise"}:
        domain = infer_domain_heuristic(nl_text)

    if domain == "audio":
        audio_plan = parsed.get("audio_plan")
        if not isinstance(audio_plan, dict):
            audio_plan = llm_parse_command(nl_text, model)
        return {"domain": "audio", "audio_plan": audio_plan}

    fan_raw = parsed.get("fan_plan")
    if not isinstance(fan_raw, dict):
        fan_raw = parsed
    return {"domain": "fan_noise", "fan_plan": normalize_fan_plan(fan_raw, nl_text=nl_text)}


# -----------------------------
# MCP Calls
# -----------------------------
async def run_mcp_from_plan(server_transport: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    지원:
      - 단일 분석(파일별) : analyze_audio_basic, render_spectrogram_png, render_fft_png, render_octave_response_png
      - 비교 플롯(overlay) : render_fft_compare_png (필수), render_octave_compare_png (선택)
    """
    from fastmcp import Client

    inputs = plan.get("inputs")
    if isinstance(inputs, str):
        inputs = [inputs]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("inputs가 비어 있습니다. 자연어에 파일 경로 1개 이상 포함해 주세요.")
    paths = [str(p).strip() for p in inputs if str(p).strip()]
    if not paths:
        raise ValueError("inputs에 유효한 파일 경로가 없습니다.")

    compare_mode = str(plan.get("compare_mode", "overlay")).strip().lower()
    if compare_mode not in {"overlay", "side_by_side"}:
        compare_mode = "overlay" if len(paths) >= 2 else "side_by_side"

    run_spectrogram = bool(plan.get("run_spectrogram", False))
    run_fft = bool(plan.get("run_fft", True))
    run_octave = bool(plan.get("run_octave", False))

    n_fft = int(plan.get("n_fft", 2048))
    hop_length = int(plan.get("hop_length", 512))
    fmin = float(plan.get("fmin", 20))
    fmax = float(plan.get("fmax", 20000))

    band = str(plan.get("band", "third"))
    smoothing = str(plan.get("smoothing", "none"))

    fft_window_sec = float(plan.get("fft_window_sec", 1.0))
    fft_scale = str(plan.get("fft_scale", "db")).lower()
    if fft_scale not in {"db", "linear"}:
        fft_scale = "db"

    outputs: Dict[str, Any] = {
        "plan_used": {
            "inputs": paths,
            "compare_mode": compare_mode,
            "run_spectrogram": run_spectrogram,
            "run_fft": run_fft,
            "run_octave": run_octave,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "fmin": fmin,
            "fmax": fmax,
            "band": band,
            "smoothing": smoothing,
            "fft_window_sec": fft_window_sec,
            "fft_scale": fft_scale,
        }
    }

    async with Client(server_transport) as client:
        # overlay 모드: 비교 플롯은 MCP 비교 tool을 우선 사용
        if compare_mode == "overlay" and len(paths) >= 2 and run_fft:
            fftc_raw = await client.call_tool(
                "render_fft_compare_png",
                {
                    "paths": paths,
                    "fmin": float(fmin),
                    "fmax": float(fmax),
                    "window_sec": float(fft_window_sec),
                    "scale": fft_scale,
                },
            )
            outputs["fft_compare"] = extract_tool_payload(fftc_raw)

        if compare_mode == "overlay" and len(paths) >= 2 and run_octave:
            # (선택) MCP 서버에 구현되어 있으면 사용
            try:
                octc_raw = await client.call_tool(
                    "render_octave_compare_png",
                    {
                        "paths": paths,
                        "band": band,
                        "fmin": float(fmin),
                        "fmax": float(fmax),
                        "smoothing": smoothing,
                    },
                )
                outputs["octave_compare"] = extract_tool_payload(octc_raw)
            except Exception:
                # 없으면 side-by-side로 대체
                compare_mode = "side_by_side"
                outputs["plan_used"]["compare_mode"] = compare_mode

        # side_by_side 또는 overlay라도 개별 결과를 함께 보고 싶으면 file-wise 결과 생성
        per_file: Dict[str, Any] = {}
        for p in paths:
            item: Dict[str, Any] = {}

            basic_raw = await client.call_tool("analyze_audio_basic", {"file_path": p})
            item["basic"] = extract_tool_payload(basic_raw)

            if run_spectrogram:
                spec_raw = await client.call_tool(
                    "render_spectrogram_png",
                    {"file_path": p, "n_fft": n_fft, "hop_length": hop_length, "fmin": int(fmin), "fmax": int(fmax)},
                )
                item["spectrogram"] = extract_tool_payload(spec_raw)

            if run_fft and not (compare_mode == "overlay" and len(paths) >= 2):
                fft_raw = await client.call_tool(
                    "render_fft_png",
                    {"file_path": p, "fmin": float(fmin), "fmax": float(fmax), "window_sec": float(fft_window_sec), "scale": fft_scale},
                )
                item["fft"] = extract_tool_payload(fft_raw)

            if run_octave and not (compare_mode == "overlay" and len(paths) >= 2 and "octave_compare" in outputs):
                # ref_path는 여기서는 사용하지 않고 상대값으로 표시(필요하면 확장 가능)
                oct_raw = await client.call_tool(
                    "render_octave_response_png",
                    {"meas_path": p, "band": band, "fmin": float(fmin), "fmax": float(fmax), "smoothing": smoothing},
                )
                item["octave"] = extract_tool_payload(oct_raw)

            per_file[p] = item

        outputs["per_file"] = per_file

    return outputs


# -----------------------------
# Rendering helpers (layout)
# -----------------------------
def _short_name(path: str) -> str:
    try:
        return Path(path).name
    except Exception:
        return path


def render_compare_blocks(results: Dict[str, Any], plot_width: int) -> None:
    # overlay 결과(있으면 상단에)
    if isinstance(results.get("fft_compare"), dict) and isinstance(results["fft_compare"].get("png_base64"), str):
        st.markdown("### FFT 비교(Overlay)")
        render_base64_png(results["fft_compare"]["png_base64"], "FFT Compare (Overlay)", use_container_width=False, width=plot_width)

    if isinstance(results.get("octave_compare"), dict) and isinstance(results["octave_compare"].get("png_base64"), str):
        st.markdown("### 옥타브 비교(Overlay)")
        render_base64_png(results["octave_compare"]["png_base64"], "Octave Compare (Overlay)", use_container_width=False, width=plot_width)


def render_side_by_side(results: Dict[str, Any], plot_width: int) -> None:
    per_file = results.get("per_file", {})
    if not isinstance(per_file, dict) or not per_file:
        st.info("표시할 파일별 결과가 없습니다.")
        return

    paths = list(per_file.keys())
    n = len(paths)

    # 1~3개: 그 수만큼 컬럼, 4개 이상: 2열 그리드
    cols_count = n if n <= 3 else 2
    cols = st.columns(cols_count)

    for i, p in enumerate(paths):
        col = cols[i % cols_count]
        with col:
            st.markdown(f"#### {_short_name(p)}")
            item = per_file[p]

            # 기본 지표는 간단히 (필요시 접기)
            basic = item.get("basic", {})
            if isinstance(basic, dict):
                sr = basic.get("sample_rate") or basic.get("sr")
                dur = basic.get("duration_sec") or basic.get("duration")
                st.caption(f"sr={sr}, duration={dur}s")

            # 이미지들: 파일별로 한 컬럼 안에 쌓기
            for key, title in (("spectrogram", "Spectrogram"), ("fft", "FFT"), ("octave", "Octave")):
                blk = item.get(key)
                if isinstance(blk, dict) and isinstance(blk.get("png_base64"), str):
                    # 개별 파일 결과는 컬럼 너비에 맞추거나, 또는 지정된 너비의 절반 정도로?
                    # 여기서는 그냥 container width를 쓰는게 나을수도 있음.
                    # 하지만 사용자가 "사이즈 작게"를 원하므로 여기서도 width를 적용해봄.
                    # 단, 컬럼을 넘어가지 않도록 use_container_width=True를 함께 쓰면 Streamlit이 알아서 함.
                    # 하지만 Streamlit image는 width가 있으면 use_container_width를 무시할 수 있음.
                    render_base64_png(blk["png_base64"], title, use_container_width=True)

            # 디테일은 접어서
            with st.expander("기본 지표(JSON)", expanded=False):
                st.json(item.get("basic", {}))


def _pick_metric_summary(summary_obj: Any, metric: str) -> Dict[str, Any]:
    if not isinstance(summary_obj, dict):
        return {}
    if metric in summary_obj and isinstance(summary_obj[metric], dict):
        return summary_obj[metric]
    return summary_obj


def _render_fan_compare_metric_block(payload: Dict[str, Any], metric: str, header: str = "") -> None:
    if header:
        st.markdown(f"##### {header}")
    if not payload.get("ok", True):
        st.error(f"{metric} 비교 실패: {payload}")
        return

    if isinstance(payload.get("plot_png_base64"), str):
        caption = "Fan Noise Compare" if metric == "noise" else "Temperature Compare"
        render_base64_png(payload["plot_png_base64"], caption, use_container_width=True)

    comparison = payload.get("comparison", {}) if isinstance(payload.get("comparison"), dict) else {}
    avg_delta = comparison.get("avg_delta_db_b_minus_a", comparison.get("avg_delta_b_minus_a"))
    rmse_delta = comparison.get("rmse_delta_db", comparison.get("rmse_delta"))
    higher_log = comparison.get("louder_log", comparison.get("higher_avg_log"))
    gap = comparison.get("loudness_gap_db", comparison.get("avg_gap"))

    unit = "dB" if metric == "noise" else "C"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Avg Δ{unit} (B-A)", avg_delta)
    c2.metric(f"RMSE Δ{unit}", rmse_delta)
    c3.metric("Higher Avg Log", higher_log)
    c4.metric(f"Gap {unit}", gap)

    log_a = payload.get("log_a", {}) if isinstance(payload.get("log_a"), dict) else {}
    log_b = payload.get("log_b", {}) if isinstance(payload.get("log_b"), dict) else {}
    rows = []
    s_a = _pick_metric_summary(log_a.get("summary"), metric)
    s_b = _pick_metric_summary(log_b.get("summary"), metric)

    if s_a:
        row = {"log": Path(str(log_a.get("path", "A"))).name, "points": s_a.get("points"), "duration_sec": s_a.get("duration_sec")}
        if metric == "noise":
            row.update(
                {
                    "avg": s_a.get("noise_db_avg"),
                    "min": s_a.get("noise_db_min"),
                    "max": s_a.get("noise_db_max"),
                    "p95": s_a.get("noise_db_p95"),
                }
            )
        else:
            row.update(
                {
                    "avg": s_a.get("temperature_c_avg"),
                    "min": s_a.get("temperature_c_min"),
                    "max": s_a.get("temperature_c_max"),
                    "p95": s_a.get("temperature_c_p95"),
                }
            )
        rows.append(row)

    if s_b:
        row = {"log": Path(str(log_b.get("path", "B"))).name, "points": s_b.get("points"), "duration_sec": s_b.get("duration_sec")}
        if metric == "noise":
            row.update(
                {
                    "avg": s_b.get("noise_db_avg"),
                    "min": s_b.get("noise_db_min"),
                    "max": s_b.get("noise_db_max"),
                    "p95": s_b.get("noise_db_p95"),
                }
            )
        else:
            row.update(
                {
                    "avg": s_b.get("temperature_c_avg"),
                    "min": s_b.get("temperature_c_min"),
                    "max": s_b.get("temperature_c_max"),
                    "p95": s_b.get("temperature_c_p95"),
                }
            )
        rows.append(row)

    if rows:
        st.dataframe(rows, use_container_width=True)


def render_fan_compare_payload(payload: Dict[str, Any], title: str = "") -> None:
    if title:
        st.markdown(f"#### {title}")

    if not payload.get("ok"):
        st.error(f"비교 실패: {payload}")
        return

    signal = str(payload.get("signal", "noise")).lower()
    if signal == "both" and isinstance(payload.get("results"), dict):
        results = payload["results"]
        noise_payload = results.get("noise", {})
        temp_payload = results.get("temperature", {})

        if isinstance(noise_payload, dict):
            merged_noise = dict(noise_payload)
            merged_noise["ok"] = True
            merged_noise["log_a"] = payload.get("log_a", {})
            merged_noise["log_b"] = payload.get("log_b", {})
            _render_fan_compare_metric_block(merged_noise, metric="noise", header="Noise")
        if isinstance(temp_payload, dict):
            merged_temp = dict(temp_payload)
            merged_temp["ok"] = True
            merged_temp["log_a"] = payload.get("log_a", {})
            merged_temp["log_b"] = payload.get("log_b", {})
            _render_fan_compare_metric_block(merged_temp, metric="temperature", header="Temperature")
    else:
        metric = "temperature" if signal == "temperature" else "noise"
        _render_fan_compare_metric_block(payload, metric=metric)

    with st.expander("비교 원본(JSON)", expanded=False):
        st.json(payload)


def render_fan_single_payload(payload: Dict[str, Any], title: str = "") -> None:
    if title:
        st.markdown(f"#### {title}")

    if not payload.get("ok"):
        st.error(f"조회 실패: {payload}")
        return

    signal = str(payload.get("signal", "noise")).lower()
    summary_all = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    series = payload.get("series", {}) if isinstance(payload.get("series"), dict) else {}
    times = series.get("time_sec") if isinstance(series.get("time_sec"), list) else []

    if signal in {"noise", "both"}:
        s = _pick_metric_summary(summary_all, "noise")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Noise Avg dB", s.get("noise_db_avg"))
        c2.metric("Noise Min dB", s.get("noise_db_min"))
        c3.metric("Noise Max dB", s.get("noise_db_max"))
        c4.metric("Noise P95 dB", s.get("noise_db_p95"))
        noise = series.get("noise_db") if isinstance(series.get("noise_db"), list) else []
        if times and noise and len(times) == len(noise):
            rows = [{"time_sec": t, "noise_db": n} for t, n in zip(times, noise)]
            st.line_chart(rows, x="time_sec", y="noise_db", use_container_width=True)

    if signal in {"temperature", "both"}:
        s = _pick_metric_summary(summary_all, "temperature")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temp Avg C", s.get("temperature_c_avg"))
        c2.metric("Temp Min C", s.get("temperature_c_min"))
        c3.metric("Temp Max C", s.get("temperature_c_max"))
        c4.metric("Temp P95 C", s.get("temperature_c_p95"))
        temps = series.get("temperature_c") if isinstance(series.get("temperature_c"), list) else []
        if times and temps and len(times) == len(temps):
            rows = [{"time_sec": t, "temperature_c": v} for t, v in zip(times, temps)]
            st.line_chart(rows, x="time_sec", y="temperature_c", use_container_width=True)

    with st.expander("단일 로그 원본(JSON)", expanded=False):
        st.json(payload)


def execute_fan_plan(
    fan_plan: Dict[str, Any],
    audio_transport: Optional[str],
    thermal_transport: Optional[str],
) -> Dict[str, Any]:
    task = fan_plan["task"]
    source = fan_plan["source"]
    signal = str(fan_plan.get("signal", "noise")).lower()
    log_paths = fan_plan.get("log_paths", [])
    if not isinstance(log_paths, list):
        log_paths = []

    if task == "generate":
        tool_name = "generate_sample_fan_noise_logs"
        args = {
            "output_dir": fan_plan["sample_dir"],
            "points": int(fan_plan["points"]),
            "step_sec": float(fan_plan["step_sec"]),
        }
    elif task == "list":
        tool_name = "list_fan_noise_logs"
        args = {"pattern": fan_plan["pattern"]}
    elif task == "read":
        if not log_paths:
            raise ValueError("read task requires at least 1 log path")
        tool_name = "read_fan_noise_log"
        args = {
            "log_path": log_paths[0],
            "smoothing_window": int(fan_plan["smoothing_window"]),
            "max_points": int(fan_plan["max_points"]),
        }
    else:
        tool_name = "compare_fan_noise_logs"
        if len(log_paths) < 2:
            raise ValueError("compare task requires 2 log paths")
        args = {
            "log_a_path": log_paths[0],
            "log_b_path": log_paths[1],
            "smoothing_window": int(fan_plan["smoothing_window"]),
            "max_points": int(fan_plan["max_points"]),
        }

    # temperature/both signal은 thermal에서만 의미가 있으므로 source를 강제 정합시킨다.
    if task in {"read", "compare"} and signal in {"temperature", "both"}:
        if source == "audio":
            raise ValueError("audio source는 temperature/both signal을 지원하지 않습니다. thermal source를 사용하세요.")
        if source == "both":
            source = "thermal"

    if source == "audio":
        if audio_transport is None:
            raise ValueError("Audio MCP transport가 설정되지 않았습니다.")
        payload = safe_call_tool(audio_transport, tool_name, args)
    elif source == "thermal":
        if thermal_transport is None:
            raise ValueError("Thermal MCP transport가 설정되지 않았습니다.")
        if task in {"read", "compare"}:
            args = dict(args)
            args["signal"] = signal
        payload = safe_call_tool(thermal_transport, tool_name, args)
    else:
        if audio_transport is None or thermal_transport is None:
            raise ValueError("Dual 모드 transport 설정이 누락되었습니다.")
        payload = run_async(call_tool_dual(audio_transport, thermal_transport, tool_name, args))

    return {
        "task": task,
        "source": source,
        "signal": signal,
        "tool_name": tool_name,
        "args": args,
        "payload": payload,
    }


def render_fan_execution(execution: Dict[str, Any]) -> None:
    task = execution.get("task")
    source = execution.get("source")
    payload = execution.get("payload")

    if task in {"compare", "read"}:
        render_fn = render_fan_compare_payload if task == "compare" else render_fan_single_payload
        if source == "audio":
            render_fn(payload if isinstance(payload, dict) else {}, title="Audio MCP 결과")
        elif source == "thermal":
            render_fn(payload if isinstance(payload, dict) else {}, title="Thermal MCP 결과")
        else:
            dual = payload if isinstance(payload, dict) else {}
            col_a, col_b = st.columns(2)
            with col_a:
                render_fn(dual.get("audio", {}), title="Audio MCP 결과")
            with col_b:
                render_fn(dual.get("thermal", {}), title="Thermal MCP 결과")
            with st.expander("Dual 결과(JSON)", expanded=False):
                st.json(dual)
        return

    st.json(payload if isinstance(payload, dict) else {"ok": False, "error": "invalid fan payload"})


# -----------------------------
# Short explanation (only results)
# -----------------------------
def build_explain_prompt(results: Dict[str, Any]) -> str:
    plan = results.get("plan_used", {})
    per_file = results.get("per_file", {})

    # 파일별 핵심 값만 추출
    brief = {}
    if isinstance(per_file, dict):
        for p, item in per_file.items():
            if not isinstance(item, dict):
                continue
            b = item.get("basic", {})
            if isinstance(b, dict):
                brief[_short_name(p)] = {
                    "sr": b.get("sample_rate") or b.get("sr"),
                    "duration_sec": b.get("duration_sec") or b.get("duration"),
                    "rms": b.get("rms"),
                    "centroid": b.get("spectral_centroid_mean") or b.get("centroid"),
                    "zcr": b.get("zcr_mean"),
                }

    payload = {
        "compare_mode": plan.get("compare_mode"),
        "inputs": [Path(p).name for p in (plan.get("inputs") or [])],
        "outputs": {
            "fft_compare": "fft_compare" in results,
            "octave_compare": "octave_compare" in results,
        },
        "basic_by_file": brief,
    }

    return (
        "다음 JSON에 기반해 요약해라.\n"
        "규칙:\n"
        "- 한국어\n"
        "- 3~5줄\n"
        "- 결과에 있는 사실만 언급(추측/튜토리얼/실험 제안 금지)\n"
        "- 파일 간 차이는 '관측된 수치' 위주로만 간단히\n\n"
        f"JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def ask_ollama_explanation(results: Dict[str, Any], model: str) -> tuple[Optional[str], Optional[str]]:
    prompt = build_explain_prompt(results)
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "너는 결과를 짧게 정리하는 분석가다."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1},
        )
        content = response.get("message", {}).get("content", "").strip()
        if not content:
            return None, "Ollama 응답이 비어 있습니다."
        return content, None
    except Exception as exc:
        return None, f"Ollama 호출 실패: {exc}"


# -----------------------------
# Streamlit UI (minimal)
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Audio + Thermal Fan Noise MCP", layout="wide")
    st.title("Audio + Thermal Fan Noise MCP")
    st.caption("자연어 요청을 LLM이 해석해 Audio 분석 또는 Fan Noise 분석으로 자동 라우팅")
    fan_samples_ok, fan_samples_state = ensure_default_fan_samples()
    if fan_samples_ok and fan_samples_state == "generated":
        st.info("기본 fan_noise 샘플 2개를 생성했습니다: ./fan_noise_samples/fan_noise_sample_[a|b].csv")
    elif not fan_samples_ok:
        st.warning(f"기본 fan_noise 샘플 준비 실패: {fan_samples_state}")

    with st.expander("서버 설정", expanded=True):
        st.markdown("#### Audio MCP")
        audio_mode_index = 0 if DEFAULT_MCP_TRANSPORT_MODE == "stdio" else 1
        audio_mode = st.selectbox("Audio MCP 연결 방식", options=["stdio", "http"], index=audio_mode_index, key="audio_mode")
        audio_url = DEFAULT_MCP_URL
        audio_script = DEFAULT_MCP_SCRIPT
        if audio_mode == "stdio":
            audio_script = st.text_input("Audio MCP 스크립트 경로", value=DEFAULT_MCP_SCRIPT, key="audio_script")
        else:
            audio_url = st.text_input("Audio MCP URL", value=DEFAULT_MCP_URL, key="audio_url")

        st.markdown("#### Thermal MCP")
        thermal_mode_index = 0 if DEFAULT_THERMAL_MCP_TRANSPORT_MODE == "stdio" else 1
        thermal_mode = st.selectbox("Thermal MCP 연결 방식", options=["stdio", "http"], index=thermal_mode_index, key="thermal_mode")
        thermal_url = DEFAULT_THERMAL_MCP_URL
        thermal_script = DEFAULT_THERMAL_MCP_SCRIPT
        if thermal_mode == "stdio":
            thermal_script = st.text_input("Thermal MCP 스크립트 경로", value=DEFAULT_THERMAL_MCP_SCRIPT, key="thermal_script")
        else:
            thermal_url = st.text_input("Thermal MCP URL", value=DEFAULT_THERMAL_MCP_URL, key="thermal_url")

        ollama_model = st.text_input("Ollama 모델", value=DEFAULT_OLLAMA_MODEL)
        plot_width = st.slider("오디오 플롯 너비(픽셀)", min_value=200, max_value=2000, value=800, step=50)
    st.subheader("명령 입력(자연어)")
    st.caption(
        "예시:\n"
        "- Audio: './test.wav 와 ./test2.wav fft 비교해줘'\n"
        "- Fan(1개): 'fan_noise_sample_a.csv 보여줘'\n"
        "- Fan(2개 비교): 'fan_noise_sample_a.csv와 fan_noise_sample_b.csv 비교해줘'\n"
        "- Fan(signal 지정): 'thermal로 온도만 비교해줘' / 'noise만 보여줘' / '온도와 소음 둘다 비교해줘'\n"
        "- Fan(source 지정): 'thermal로 fan noise 비교해줘' 또는 'audio mcp로 fan noise 봐줘'\n"
    )
    nl = st.text_area("자연어 요청", value="fan noise 로그 비교해줘", height=110, key="unified_nl")
    run_clicked = st.button("실행", type="primary", key="unified_run_btn")

    if not run_clicked:
        return

    try:
        with st.spinner("1) 요청 라우팅(Ollama) 중..."):
            routed = llm_route_request(nl_text=nl, model=ollama_model)

        with st.expander("라우팅 결과(JSON)", expanded=False):
            st.json(routed)

        domain = routed.get("domain")
        if domain == "audio":
            audio_plan = routed.get("audio_plan")
            if not isinstance(audio_plan, dict):
                raise ValueError("audio_plan이 없습니다.")

            audio_transport = resolve_mcp_transport(mode=audio_mode, mcp_url=audio_url, mcp_script=audio_script)
            with st.spinner("2) Audio MCP 분석/비교 실행 중..."):
                results = run_async(run_mcp_from_plan(server_transport=audio_transport, plan=audio_plan))

            st.subheader("Audio 결과")
            render_compare_blocks(results, plot_width=plot_width)
            st.markdown("### 파일별 결과")
            render_side_by_side(results, plot_width=plot_width)

            st.markdown("### 요약")
            with st.spinner("3) 요약 생성 중..."):
                explanation, err = ask_ollama_explanation(results, model=ollama_model)
            if err:
                st.warning(err)
            else:
                st.write(explanation)
            return

        if domain == "fan_noise":
            fan_plan = normalize_fan_plan(routed.get("fan_plan", {}), nl_text=nl)
            with st.expander("Fan 실행 계획(JSON)", expanded=False):
                st.json(fan_plan)

            source = fan_plan["source"]
            audio_transport: Optional[str] = None
            thermal_transport: Optional[str] = None

            if source in {"audio", "both"}:
                audio_transport = resolve_mcp_transport(mode=audio_mode, mcp_url=audio_url, mcp_script=audio_script)
            if source in {"thermal", "both"}:
                thermal_transport = resolve_mcp_transport(mode=thermal_mode, mcp_url=thermal_url, mcp_script=thermal_script)

            with st.spinner("2) Fan Noise MCP 실행 중..."):
                execution = execute_fan_plan(
                    fan_plan=fan_plan,
                    audio_transport=audio_transport,
                    thermal_transport=thermal_transport,
                )

            st.subheader("Fan Noise 결과")
            render_fan_execution(execution)
            return

        raise ValueError(f"알 수 없는 domain: {domain}")

    except Exception as exc:
        st.error(f"실행 오류: {exc}")
        st.info(
            "체크리스트:\n"
            "1) stdio 모드: app의 스크립트 경로(audio_mcp_server.py / thermal_mcp_server.py) 확인\n"
            "2) http 모드: 각 MCP 서버 URL 확인\n"
            "3) MCP_WORKSPACE 설정: export MCP_WORKSPACE=$PWD (서버 실행 전)\n"
            "4) 입력 파일 경로가 워크스페이스 내부인지\n"
            "5) Ollama 모델명이 설치된 모델과 일치하는지\n"
        )


if __name__ == "__main__":
    main()
