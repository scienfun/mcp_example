# MCP Audio Analysis Example (Ollama + Streamlit + FastMCP)

로컬(오프라인) 환경에서 실행 가능한 오디오 분석 예제입니다.

- Streamlit UI에서 오디오 업로드 (`wav/mp3/m4a`)
- FastMCP 도구 호출로
  - 스펙트로그램 PNG 생성
  - 1 octave / 1/3 octave 응답 PNG 생성
- 결과를 Ollama가 초보자 관점으로 설명

## 프로젝트 구조

```text
mcp_example/
├── app.py
├── audio_mcp_server.py
├── requirements.txt
├── README.md
└── src/
    └── audio/
        ├── __init__.py
        ├── features.py
        └── plots.py
```

## 1) 설치 (venv)

```bash
cd /home/scienfun/mcp_example
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Ollama 준비

로컬 Ollama 서버/모델을 준비합니다.

```bash
ollama serve
```

새 터미널에서 모델 설치(최초 1회):

```bash
ollama pull llama3.2
```

## 3) 실행

터미널 A (FastMCP 서버):

```bash
cd /home/scienfun/mcp_example
source .venv/bin/activate
fastmcp run audio_mcp_server.py:mcp --transport http --port 8000
```

터미널 B (Streamlit 앱):

```bash
cd /home/scienfun/mcp_example
source .venv/bin/activate
streamlit run app.py
```

브라우저에서 Streamlit 화면이 열리면:

1. `측정 파일(필수)` 업로드
2. 필요 시 `참조 파일(선택)` 업로드
3. 분석 옵션 설정
4. `분석 실행` 클릭

## 4) UI 사용법

- 업로드:
  - `측정 파일(필수)`
  - `참조 파일(선택)`
- 옵션:
  - 분석 타입: `Spectrogram` / `Octave Response` / `Both`
  - 밴드: `1 octave` / `1/3 octave`
  - `fmin`, `fmax`
  - `n_fft`, `hop_length`
  - `smoothing`: `none` / `median`
- 출력:
  - 기본 지표 JSON
  - 스펙트로그램 PNG
  - 옥타브 응답 PNG
  - Ollama 설명 텍스트

## 5) 해석 모드 차이

- 참조 파일 없음:
  - 측정 파일 자체의 밴드 레벨을 상대 응답으로 표시
  - UI에 "measurement only" 의미로 안내
- 참조 파일 있음:
  - `응답(dB) = L_meas - L_ref` (차분/전달 특성)

## 6) 최소 검증 절차

### A. 간단 테스트 WAV 만들기

```bash
cd /home/scienfun/mcp_example
source .venv/bin/activate
python - <<'PY'
import numpy as np
import soundfile as sf
sr = 48000
sec = 2.0
t = np.arange(int(sr*sec))/sr
x = 0.2*np.sin(2*np.pi*440*t)
sf.write('test_440.wav', x, sr)
print('created test_440.wav')
PY
```

### B. 앱에서 `test_440.wav` 업로드

- `Both`로 실행
- 스펙트로그램/옥타브 응답 이미지가 표시되는지 확인
- 옥타브 `bands`가 비어있지 않고, 중심 주파수가 `fmin~fmax` 범위에 있는지 확인

## 7) 트러블슈팅

- Ollama 에러:
  - `ollama serve` 실행 여부 확인
  - `ollama pull llama3.2` 완료 여부 확인
  - Streamlit의 모델명 입력값이 설치 모델과 일치하는지 확인
- MCP 연결 에러:
  - `fastmcp run ... --port 8000` 프로세스 실행 여부 확인
  - 앱의 MCP URL이 `http://127.0.0.1:8000/mcp`인지 확인
- `m4a` 로딩 실패:
  - 시스템 오디오 디코더 환경에 따라 지원이 달라질 수 있으므로 `wav`로 재시도
