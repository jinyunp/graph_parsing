# RAG Chart Analyzer (Images-Only, Ollama + Qwen2.5-VL)

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Ollama](https://img.shields.io/badge/Ollama-qwen2.5--vl-black)
![Status](https://img.shields.io/badge/Status-Active-success)

> **목표**: 폴더(또는 단일 파일)에 있는 **그래프 이미지**를 Vision-Language Model(VLM) **Qwen2.5-VL**(via **Ollama**, HF, OpenRouter)로 분석하여,
> **일관된 dataclass 스키마(JSON)** 로 그래프 구조 및 **의미 중심 키워드**를 생성하고,
> 이어서 **이미지 + 키워드 기반 의미 요약(semantic summary)** 를 자동 생성합니다.

> **GRAPH PARSER** : 그래프 이미지 → 구조화 메타데이터(JSON) + **의미 중심 키워드(10~15)** 추출  
> **GRAPH ANALYZER** : 이미지 + 키워드 → **의미 중심 요약(semantic summary)** 생성/저장

---

## ✨ 핵심 기능

* ✅ **2단계 파이프라인**:
  (1) 그래프 분석(JSON + key_phrases) → (2) 의미 요약(summary)
* ✅ **Ollama / HuggingFace / OpenRouter** 등 다중 백엔드 지원
* ✅ **Qwen2.5-VL-3B-Instruct** 기반, 시각·언어 통합 분석
* ✅ **의미 중심 키워드 인퍼런스** (단순 화학식/수치 나열 억제)
* ✅ **JSON 파싱 실패 시 자동 fallback → 키워드만 재시도**
* ✅ **raw 응답/시간 로그 자동 저장**, 고가용성 처리

---

## 🧭 아키텍처

```mermaid
flowchart LR
    A[Images Folder\nor Single Image] -->|iterate / choose| B(Graph Parser\nQwen2.5-VL)
    B --> C[Structured JSON\n+ key_phrases]
    C --> D(Graph Analyzer\nImage + Keywords)
    D --> E[Semantic Summary (txt)]
```

---

## 📁 프로젝트 구조

```
rag-chart-analyzer-framework/
├─ config.py                    # 환경변수 기반 설정 (백엔드/경로/플래그)
├─ schemas.py                   # dataclass 스키마 (ChartMetadata 등)
├─ prompts_chart_keywords.py    # Step1: 구조 추출 + 의미 중심 키워드 생성 프롬프트
├─ prompts_semantic_summary.py  # Step2: 키워드 기반 의미 요약 프롬프트
├─ vlm_client.py                # VLM 호출(HF/Ollama/OpenRouter) + JSON 파싱/복원/요약
├─ runner.py                    # Step1 실행: 이미지 → 구조화 JSON(+key_phrases)
├─ runner_summary.py            # Step2 실행: JSON에서 이미지+키워드 기반 요약 생성
├─ requirements.txt
├─ README.md
├─ data/
│  └─ images/                   # 입력 이미지 폴더
└─ out/
   ├─ json/                     # Step1 구조화 JSON
   ├─ raw/                      # 원본 응답: *.raw.txt / *.raw.http.json / *.summary.*
   └─ summary/                  # Step2 의미 요약 텍스트 (*.summary.txt)
```

---

## ⚙️ 설치

```bash
# 1) 모델 준비
ollama pull qwen2.5vl:3b

# 2) 파이썬 의존성
python3 -m venv .venv && source .venv/bin/activate
bash setup_qwen_xet.sh
```

> 📝 Ollama가 가동 중이어야 합니다 (`http://localhost:11434`)
> `curl http://localhost:11434/api/tags` 로 상태 확인 가능

---

## 🚀 실행 예시

### 🧩 Step 1 — GRAPH PARSER (이미지 → JSON + 키워드)

#### 폴더 전체 처리

```bash
export INPUT_MODE=folder
export INPUT_IMAGE_DIR=./data/images
python runner.py
```

#### 단일 이미지 처리

```bash
export INPUT_MODE=single
export INPUT_IMAGE_PATH=./data/images/sample.png
python runner.py
```

* 출력: `out/json/{파일명}.json`, `out/raw/{파일명}.raw.*`
* JSON 파싱 실패 시 자동으로 키워드만 재시도 (`kw-retry` 로그 표시)

---

### 🧠 Step 2 — GRAPH ANALYZER (이미지 + 키워드 → 의미 요약)

```bash
python runner_summary.py
```

* 입력: Step1의 JSON(`out/json/`)
* 출력:

  * `out/summary/{파일명}.summary.txt`
  * `out/raw/{파일명}.summary.raw.*` (원응답/HTTP JSON)

---

### Ollama
```bash
# 사전: ollama run qwen2.5vl:3b  등으로 모델 준비
export BACKEND=ollama
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="qwen2.5vl:3b"

# Step1
export INPUT_MODE=folder
export INPUT_IMAGE_DIR=./data/images
python runner.py

# Step2
python runner_summary.py
```

### OpenRouter
```bash
export BACKEND=openrouter
export OPENROUTER_API_KEY="sk-..."
export OPENROUTER_MODEL="qwen/qwen2.5-vl-7b-instruct"
export OPENROUTER_HTTP_REFERER="https://example.com"
export OPENROUTER_TITLE="RAG Chart Analyzer"

# Step1
export INPUT_MODE=folder
export INPUT_IMAGE_DIR=./data/images
python runner.py

# Step2
python runner_summary.py
```

---

## 🔧 config.py 주요 설정

| 키                      | 설명                                        | 기본값             |
| ---------------------- | ----------------------------------------- | --------------- |
| `BACKEND`              | `hf`, `ollama`, `openrouter` 중 선택         | `hf`            |
| `HF_MODEL_ID`          | HF 모델명 (예: `Qwen/Qwen2.5-VL-3B-Instruct`) | -               |
| `OLLAMA_MODEL`         | Ollama 모델명                                | `qwen2.5vl:3b`  |
| `INPUT_MODE`           | 입력 모드(`folder` or `single`)               | `folder`        |
| `OUTPUT_JSON_DIR`      | JSON 출력 폴더                                | `./out/json`    |
| `OUTPUT_SUMMARY_DIR`   | 요약 텍스트 출력 폴더                              | `./out/summary` |
| `SAVE_RAW_RESPONSE`    | 원본 응답 저장 여부                               | `true`          |
| `KEYWORDS_MIN/MAX`     | 키워드 최소/최대 개수                              | `10 / 15`       |
| `SUMMARY_MIN/MAX_SENT` | 요약 문장 수 범위                                | `3 / 6`         |

---

## 📄 출력 예시

```json
{
  "is_chart": true,
  "chart_type": "line",
  "orientation": "vertical",
  "title": {"text": "병균 감염률 (%)", "is_inferred": false},
  "x_axis": {"name": "휴동일 수", "unit": "일", "scale": "linear"},
  "y_axis": {"name": "병균 감염률", "unit": "%", "scale": "linear"},
  "legend": {"present": false, "labels": []},
  "series": [{"label": "Schweigerm(중)", "summary": "10일 내 병균 감염률 변화"}],
  "key_phrases": [
    "병균 감염률 추세", "휴동일 수 영향", "감염률 증가", "시간–감염률 관계",
    "70~90% 고감염 구간", "단기 회복 지연", "감염률 포화 영역"
  ],
  "confidence": 0.9,
  "source": {"image_path": "./data/images/sample.png", "image_sha1": "…"}
}
```

---

## 🧩 의미 요약 출력 예시

```text
이 그래프는 휴동일 수에 따라 병균 감염률이 변화하는 경향을 보여준다.
초기에는 급격히 감소하나, 일정 시점 이후 점차 증가하여 약 70~90% 수준에서 포화된다.
감염률은 단기 회복 지연과 관련된 패턴을 보이며, 장기적 안정 구간이 존재한다.
```

---

## 🧠 프롬프트 설계 요약

| 단계                   | 목적                       | 핵심 규칙                                                 |
| -------------------- | ------------------------ | ----------------------------------------------------- |
| **Step1 – Keywords** | 그래프 구조 분석 + 의미 중심 키워드 생성 | - 수치/화학식 단독 금지<br>- 최소 6개는 의미·관계형 키워드<br>- 중복·유사어 금지  |
| **Step2 – Summary**  | 이미지+키워드 기반 의미 요약         | - 관계/추세/임계점/메커니즘 중심<br>- 불확실성 명시 가능<br>- 간결하고 도메인 친화적 |

---

## 🛠️ 문제 해결

* **`RetryError` / `OSError`** → Ollama 또는 HF 백엔드 불안정 → 재시도 자동 수행
* **`json.loads` 오류** → 자동 fallback: 키워드만 재시도 후 결과 저장
* **`Some parameters are on the meta device`** → HF 모델 오프로딩 중 경고 (무시 가능)
* **`temperature ignored`** → HF generate 파라미터 무시 경고 (무시 가능)

---

## 🧪 테스트 팁

* `data/images/`에 2~3개의 실험 이미지를 배치
* `SAVE_NON_CHART_JSON=True`로 비차트 필터링 확인
* 요약 품질 확인 후 키워드 개수(`KEYWORDS_MIN/MAX`) 조정


---

## 📜 라이선스

MIT License © 2025

---

## 🙌 Acknowledgements

* [Ollama](https://ollama.com/)
* [Qwen2.5-VL](https://modelscope.cn/models/qwen)
* [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---
