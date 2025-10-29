import os

BACKEND = os.environ.get("BACKEND", "hf").lower()

# HF Transformers: Qwen2.5-VL-3B-Instruct 기본
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "models/Qwen2.5-VL-3B-Instruct")
HF_DTYPE = os.environ.get("HF_DTYPE", "auto")  # auto|float16|bfloat16|float32
HF_DEVICE_MAP = os.environ.get("HF_DEVICE_MAP", "auto")
HF_TRUST_REMOTE_CODE = os.environ.get("HF_TRUST_REMOTE_CODE", "true").lower() == "true"
HF_MAX_NEW_TOKENS = int(os.environ.get("HF_MAX_NEW_TOKENS", "2048"))
HF_USE_FLASH_ATTN = os.environ.get("HF_USE_FLASH_ATTN", "false").lower() == "true"
HF_OFFLOAD_FOLDER = os.environ.get("HF_OFFLOAD_FOLDER", "")

# 입력
INPUT_MODE = os.environ.get("INPUT_MODE", "folder").lower()   # folder | single
INPUT_IMAGE_DIR = os.environ.get("INPUT_IMAGE_DIR", "./data/images")
INPUT_IMAGE_PATH = os.environ.get("INPUT_IMAGE_PATH", "./data/sample.png")

# 출력
OUTPUT_JSON_DIR = os.environ.get("OUTPUT_JSON_DIR", "./out/json")
OUTPUT_RAW_DIR = os.environ.get("OUTPUT_RAW_DIR", "./out/raw")

# 동작 옵션
SAVE_NON_CHART_JSON = True
KEYWORDS_MIN = int(os.environ.get("KEYWORDS_MIN", "10"))
KEYWORDS_MAX = int(os.environ.get("KEYWORDS_MAX", "15"))
SAVE_RAW_RESPONSE = os.environ.get("SAVE_RAW_RESPONSE", "true").lower() == "true"
DEBUG = os.environ.get("DEBUG", "true").lower() == "true"
DEBUG_TRACE = os.environ.get("DEBUG_TRACE", "true").lower() == "true"
