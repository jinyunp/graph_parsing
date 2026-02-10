# Graph Parsing / OCR Environment Setup

이 레포는 `setup.sh`로 **가상환경(venv) 생성 + 의존성 설치 + 모델 다운로드**를 자동화하고,  
DeepSeek-OCR2로 PDF 문서를 페이지별로 렌더링 후 OCR하여 Markdown으로 저장합니다.

- 지원 버전: `qwen`, `deepseek`
- 모델 다운로드 위치: `/workspace/models/<MODEL_NAME>`
- venv 위치: `.venv/<version>` (예: `.venv/deepseek`)

---

## 1. 디렉터리 구조 (권장)

```text
graph_parsing/
├─ setup.sh
├─ requirements.qwen.txt
├─ requirements.deepseek.txt
├─ deepseek-ocr.py
├─ data/
│  ├─ docs/            # PDF 입력 폴더
│  └─ output/          # 결과 저장 폴더
└─ config.py           # 모델 ID 등 설정
````

---

## 2. 사전 준비

### 2.1 GPU 확인

```bash
nvidia-smi
```

### 2.2 (권장) 작업 위치

RunPod / Docker 환경 기준으로 `/workspace` 아래에서 작업하는 것을 권장합니다.

---

## 3. setup.sh 사용법

### 3.1 기본 사용

```bash
bash setup.sh --version qwen
bash setup.sh --version deepseek
```

* `--version qwen`

  * 모델: `Qwen/Qwen2.5-VL-3B-Instruct`
  * requirements: `requirements.qwen.txt`
  * venv: `.venv/qwen`
  * 모델 저장: `/workspace/models/Qwen2.5-VL-3B-Instruct`

* `--version deepseek`

  * 모델: `deepseek-ai/DeepSeek-OCR-2`
  * requirements: `requirements.deepseek.txt`
  * venv: `.venv/deepseek`
  * 모델 저장: `/workspace/models/DeepSeek-OCR-2`

### 3.2 Hugging Face 토큰(선택)

Gated 모델 또는 rate-limit 회피가 필요하면 토큰을 넣어 실행합니다.

```bash
HF_TOKEN=hf_xxx bash setup.sh --version deepseek
```

### 3.3 설치 완료 후 venv 활성화

```bash
source .venv/deepseek/bin/activate
# 또는
source .venv/qwen/bin/activate
```

---

## 4. requirements 파일

### 4.1 requirements.qwen.txt (예시)

```txt
transformers>=4.45.0
accelerate>=0.30.0
torch>=2.2.0
torchvision>=0.17.0
pillow
requests
tenacity
safetensors
```

### 4.2 requirements.deepseek.txt (예시)

> flash-attn은 환경 의존성이 커서 setup.sh에서 별도 설치하는 방식을 권장합니다.

```txt
transformers==4.46.3
tokenizers==0.20.3
accelerate>=0.26.0
PyMuPDF
img2pdf
einops
easydict
addict
pillow
numpy
psutil
tqdm
```

---

## 5. config.py 설정

프로젝트 루트에 `config.py`를 두고 모델 ID를 지정합니다.

```python
# config.py
DEEPSEEK_MODEL_ID = "/workspace/models/DeepSeek-OCR-2"
# 또는 Hugging Face repo id 그대로 사용:
# DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"
```

---

## 6. DeepSeek OCR 실행

### 6.1 입력 PDF 폴더

* `data/docs/` 폴더에 PDF를 넣습니다.

예:

```bash
ls data/docs
# a.pdf  b.pdf  c.pdf
```

### 6.2 실행

(venv 활성화 후)

```bash
python deepseek-ocr.py
```

### 6.3 결과 저장 구조

PDF 파일명 기준으로 자동 폴더가 생성됩니다.

```text
data/output/
└─ <PDF파일명>/
   ├─ page_0001.png
   ├─ page_0001/          # infer 저장 산출물(옵션)
   ├─ page_0002.png
   ├─ page_0002/
   └─ <PDF파일명>_1-10.md  # 합쳐진 결과 markdown
```

---

## 7. PDF 페이지 범위 지정

`deepseek-ocr.py`에서 아래 값을 조정합니다.

* 전체 처리:

  ```python
  START_PAGE = None
  END_PAGE = None
  ```
* 일부 페이지 처리(예: 3~10):

  ```python
  START_PAGE = 3
  END_PAGE = 10
  ```

---

## 8. 콘솔에 OCR 결과 출력 막기

`model.infer()` 내부에서 출력되는 텍스트를 숨기려면 아래처럼 사용합니다.

```python
with open(os.devnull, "w") as fnull:
    with redirect_stdout(fnull):
        res = model.infer(...)
```

진행률은 `tqdm`로 확인합니다.

---

## 9. Troubleshooting

### 9.1 GPU를 안 쓰는 것 같아요

아래가 True인지 확인하세요.

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

* `False`면 CUDA torch가 아닌 상태일 수 있습니다.
* deepseek 환경에서 CUDA 11.8 기준 예시:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

### 9.2 `device_map` 에러: Accelerate 필요

에러 예:

> Using `device_map` requires Accelerate...

해결:

```bash
pip install -U "accelerate>=0.26.0"
```

또는 코드에서 `device_map` 제거 후 `.to("cuda")` 사용.

### 9.3 FlashAttention2 관련 에러/경고

* `flash-attn`이 설치되지 않으면 FlashAttention2 사용 불가
* 설치 시 `psutil` 누락으로 실패할 수 있어 deepseek requirements에 `psutil` 포함 권장

설치 예:

```bash
pip install -U psutil
pip install flash-attn==2.7.3 --no-build-isolation
```

FlashAttention2가 불안정하면 코드에서 `attn_implementation="flash_attention_2"`를 제거해도 동작합니다(속도만 저하).

---

## 10. 참고

* PyMuPDF로 PDF 페이지를 이미지로 렌더링 후 OCR합니다.
* DPI를 올리면 정확도는 올라가지만 속도/VRAM 사용량도 증가합니다.

  * 권장: `DPI=200~300`
