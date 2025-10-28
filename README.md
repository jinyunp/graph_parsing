# RAG Chart Extractor (Images-Only, Ollama + Qwen2.5-VL)

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Ollama](https://img.shields.io/badge/Ollama-qwen2.5--vl-black)
![Status](https://img.shields.io/badge/Status-Active-success)

> **목표**: 폴더(또는 단일 파일)에 있는 **그래프 이미지**를 Vision-Language Model(VLM) **Qwen2.5-VL**(via **Ollama**)로 분석하여,  
> **일관된 dataclass 스키마**(축/범례/타입/주석 등)로 **JSON 메타데이터**를 생성합니다.  
> **PDF 전처리(그래프 추출)** 단계는 건너뛰고, **이미지 입력만** 처리합니다.

---

## ✨ 핵심 기능

- ✅ **이미지 입력 전용 파이프라인** (PDF는 이미 선별 완료 가정)
- ✅ **Ollama + Qwen2.5-VL:3b** 로 그래프 메타데이터 추출
- ✅ 견고한 **JSON 파서** + **스키마(dataclass)** 검증
- ✅ **단일 이미지** or **폴더 전체** 처리 (환경변수/`config.py`로 제어)
- ✅ 실패 시 **HTTP/연결/타임아웃** 원인 메시지 출력

---

## 🧭 아키텍처

```mermaid
flowchart LR
    A[Images Folder\nor Single Image] -->|iterate / choose| B(Analyzer: Qwen2.5-VL via Ollama)
    B --> C[JSON Parser\n& Dataclass Mapping]
    C --> D[Output JSON\n(out/json/*.json)]
```

---

## 📁 프로젝트 구조

```
rag-chart-extractor-images-v2/
├─ config.py                # 입력 모드/경로, 출력 폴더, 모델 설정
├─ schemas.py               # dataclass 스키마 (ChartMetadata 등)
├─ vlm_client.py            # Ollama Chat API 호출 + JSON→dataclass 변환
├─ runner.py                # 폴더/단일 이미지 처리 로직 (프리플라이트 포함)
├─ requirements.txt
└─ README.md                # (이 파일)
```

---

## ⚙️ 설치

```bash
# 1) 모델 준비
ollama pull qwen2.5vl:3b

# 2) 파이썬 의존성
pip install -r requirements.txt
```

> 📝 **Ollama 서버가 가동 중**이어야 합니다. (기본: `http://localhost:11434`)  
> `curl http://localhost:11434/api/tags` 로 상태 확인 가능

---

## 🚀 빠른 시작 (Quickstart)

### 폴더 전체 처리
```bash
export INPUT_MODE=folder
export INPUT_IMAGE_DIR=./data/images   # 여기에 PNG/JPG/JPEG/BMP/TIF/TIFF/WEBP
python runner.py
```

### 단일 파일 처리
```bash
export INPUT_MODE=single
export INPUT_IMAGE_PATH=./data/images/sample.png
python runner.py
```

- 결과 JSON은 기본적으로 `./out/json/` 아래에 `{이미지이름}.json`으로 저장됩니다.
- 차트가 아닌 이미지도 저장하려면 `config.py`의 `SAVE_NON_CHART_JSON=True` 유지.

---

## 🔧 설정 (config.py)

| 키 | 설명 | 기본값 |
|---|---|---|
| `OLLAMA_HOST` | Ollama 서버 URL | `http://localhost:11434` |
| `VLM_MODEL` | 사용할 VLM 모델 | `qwen2.5vl:3b` |
| `INPUT_MODE` | 입력 모드(`folder` or `single`) | `folder` |
| `INPUT_IMAGE_DIR` | 폴더 처리 시 이미지 경로 | `./data/images` |
| `INPUT_IMAGE_PATH` | 단일 처리 시 이미지 파일 | `./data/sample.png` |
| `OUTPUT_JSON_DIR` | JSON 출력 폴더 | `./out/json` |
| `SAVE_NON_CHART_JSON` | 차트가 아니어도 JSON 저장 | `True` |

> ⚠️ 환경변수로도 덮어쓸 수 있습니다. (`export KEY=value`)

---

## 📄 출력 예시 (요약)

```json
{
  "is_chart": true,
  "chart_type": "line",
  "orientation": "vertical",
  "title": {"text": "출선온도 변화", "is_inferred": false},
  "x_axis": {"name": "시간", "unit": "h", "is_inferred": false, "scale": "time", "ticks_examples": ["0", "12", "24"]},
  "y_axis": {"name": "온도", "unit": "℃", "is_inferred": false, "scale": "linear", "ticks_examples": ["1400", "1500"]},
  "legend": {"present": true, "labels": ["고로#1", "고로#2"], "location_hint": "top-right"},
  "annotations_present": true,
  "data_series_count": 2,
  "table_like": false,
  "caption_nearby": null,
  "key_phrases": ["출선", "온도", "시간"],
  "quality_flags": {"low_resolution": false, "cropped_or_cutoff": false, "non_korean_text_present": false, "heavy_watermark": false, "skew_or_perspective": false},
  "confidence": 0.82,
  "source": {"image_path": "./data/images/sample.png", "image_sha1": "…"},
  "extra": {}
}
```

---

## 🧠 프롬프트 전략 (요약)

- **단일 JSON 객체만** 생성하도록 강제 (코드펜스/여분 텍스트 금지).
- 확인되지 않으면 `null/false/[]`로 **보수적으로 채우기**.
- 이미지 내 실제 텍스트는 **한국어 그대로**, 추론 값은 `is_inferred=true` 표기.
- 축 스케일/눈금 예시/범례 라벨 등 **RAG 친화 메타**에 집중.

---

## 🛠️ 문제 해결 (Troubleshooting)

- **`RetryError ... HTTPError`**: Ollama가 200이 아닌 응답.  
  - 모델 설치 여부: `ollama pull qwen2.5vl:3b`  
  - 서버 상태: `curl $OLLAMA_HOST/api/tags`  
  - `OLLAMA_HOST` 주소/포트 확인
- **연결 실패/타임아웃**: 네트워크/방화벽, 첫 로딩 지연 가능 → 타임아웃 상향(이미 반영).
- **빈 응답(content 없음)**: 모델이 JSON을 출력하지 않음 → 프롬프트를 강화하거나 이미지 품질 확인.

---

## 🧪 테스트 팁

- 작은 샘플 이미지 2~3장으로 동작 확인 후, 전체 폴더로 확장하세요.
- `SAVE_NON_CHART_JSON=True`로 비차트 검출률/오탐을 함께 점검.

---

## 🗺️ 로드맵

- [ ] bbox 좌표 반환 옵션
- [ ] 캡션 인접 텍스트 OCR + 통합

---

## 📜 라이선스

MIT License © 2025

---

## 🙌 Acknowledgements

- [Ollama](https://ollama.com/)
- [Qwen2.5-VL](https://modelscope.cn/models/qwen)
