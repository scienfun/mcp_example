# MCP Example (Audio + Thermal/Fan, LLM Routing)

이 프로젝트는 Streamlit 단일 UI에서 자연어 요청을 받아, LLM이 요청을 해석해 `audio` 또는 `fan_noise` 작업으로 라우팅하는 예제입니다.

핵심 포인트:
- 단일 입력창(자연어) 기반 실행
- `audio` 분석과 `fan_noise` 분석을 자동 분기
- `fan_noise`는 파일 1개 조회(`read`) / 2개 비교(`compare`) 모두 지원
- `thermal` 경로에서는 `noise` / `temperature` / `both` 신호 선택 지원
- MCP transport 기본값은 `stdio` (필요 시 `http` 사용 가능)

## 주요 기능

### 1) Audio 분석
- 입력: WAV/MP3/M4A 파일 경로(자연어에서 추출)
- 지원:
  - 기본 지표 분석
  - FFT(단일/비교 overlay)
  - Spectrogram
  - Octave 응답/비교
- 결과: 플롯 + 파일별 지표 + 요약 텍스트

### 2) Fan/Noise/Thermal 분석
- 입력: CSV 로그 파일 경로 1개 또는 2개
- 작업:
  - `read` (1개 파일 조회)
  - `compare` (2개 파일 비교)
  - `generate` (샘플 로그 생성)
  - `list` (로그 목록)
- Thermal 신호 선택:
  - `noise`: 소음 레벨
  - `temperature`: 온도
  - `both`: 소음 + 온도 동시

기본 샘플 파일:
- `fan_noise_samples/fan_noise_sample_a.csv`
- `fan_noise_samples/fan_noise_sample_b.csv`

앱 시작 시 샘플 파일이 없으면 자동 생성합니다.

## 프로젝트 구조

```text
mcp_example/
├── app.py
├── audio_mcp_server.py
├── thermal_mcp_server.py
├── fan_noise_samples/
├── src/
│   ├── audio/
│   └── fan_noise/
├── requirements.txt
└── README.md
```

## 설치

```bash
cd /home/scienfun/mcp_example
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 실행

### A. 기본 권장 (stdio 모드)
별도 MCP 서버를 먼저 띄우지 않아도 됩니다.

```bash
cd /home/scienfun/mcp_example
source .venv/bin/activate
streamlit run app.py
```

### B. http 모드 사용 시
앱에서 transport를 `http`로 바꿀 경우 각 서버를 별도 실행합니다.

터미널 1:
```bash
cd /home/scienfun/mcp_example
source .venv/bin/activate
fastmcp run audio_mcp_server.py:mcp --transport http --port 8000
```

터미널 2:
```bash
cd /home/scienfun/mcp_example
source .venv/bin/activate
fastmcp run thermal_mcp_server.py:mcp --transport http --port 8101
```

터미널 3:
```bash
cd /home/scienfun/mcp_example
source .venv/bin/activate
streamlit run app.py
```

## Ollama

앱은 자연어 라우팅/요약에 Ollama를 사용합니다.

```bash
ollama serve
```

필요 모델은 환경에 맞게 준비하세요. 앱에서 모델명을 직접 바꿀 수 있습니다.

## 자연어 예시

### Audio
- `./test.wav 와 ./test2.wav fft 비교해줘`
- `./test.wav 스펙트로그램과 옥타브 분석해줘`

### Fan/Noise/Thermal
- `fan_noise_sample_a.csv 보여줘`
- `fan_noise_sample_a.csv와 fan_noise_sample_b.csv 비교해줘`
- `thermal로 온도만 비교해줘`
- `thermal로 noise만 보여줘`
- `온도와 소음 둘다 비교해줘`
- `audio와 thermal 둘다 noise 비교해줘`

## source/signal 동작 규칙

- `source`:
  - `audio` / `thermal` / `both` / `auto`
- `signal`:
  - `noise` / `temperature` / `both` / `auto`

자동 해석 시:
- 온도 요청이면 `thermal`로 정합
- `temperature` 또는 `both` 신호는 thermal 경로에서 처리
- noise 비교에서만 `both` 소스 동시 호출을 허용

즉, 불필요한 Audio 결과를 강제로 같이 보여주지 않습니다.

## 환경 변수

기본값:
- `MCP_TRANSPORT_MODE=stdio`
- `THERMAL_MCP_TRANSPORT_MODE=stdio`

선택 변수:
- `MCP_SERVER_URL` (audio http URL)
- `MCP_SERVER_SCRIPT` (audio stdio 스크립트)
- `THERMAL_MCP_SERVER_URL` (thermal http URL)
- `THERMAL_MCP_SCRIPT` (thermal stdio 스크립트)
- `OLLAMA_MODEL`
- `FAN_MCP_SOURCE` (`auto|audio|thermal|both`, 기본 `auto`)

## 트러블슈팅

- Ollama 응답 실패:
  - `ollama serve` 실행 여부 확인
  - 앱에 입력한 모델명이 실제 설치 모델과 일치하는지 확인

- MCP 연결 실패:
  - stdio 모드면 스크립트 경로(`audio_mcp_server.py`, `thermal_mcp_server.py`) 확인
  - http 모드면 서버 프로세스/포트/URL 확인

- fan 로그 조회/비교 실패:
  - CSV 경로가 워크스페이스 내부인지 확인
  - `temperature` 요청 시 CSV에 `temperature_c` 컬럼이 있는지 확인
