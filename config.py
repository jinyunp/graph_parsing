import os

BACKEND = os.environ.get("BACKEND", "hf").lower()

# HF Transformers
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "models/Qwen2.5-VL-3B-Instruct")
HF_DTYPE = os.environ.get("HF_DTYPE", "auto")  # 'auto' | 'float16' | 'bfloat16' | 'float32'
HF_DEVICE_MAP = os.environ.get("HF_DEVICE_MAP", "auto")
HF_TRUST_REMOTE_CODE = os.environ.get("HF_TRUST_REMOTE_CODE", "true").lower() == "true"
HF_MAX_NEW_TOKENS = int(os.environ.get("HF_MAX_NEW_TOKENS", "512"))
HF_USE_FLASH_ATTN = os.environ.get("HF_USE_FLASH_ATTN", "false").lower() == "true"

# Ollama (optional)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:3b")

# OpenRouter (optional)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen2.5-vl-7b-instruct")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_HTTP_REFERER = os.environ.get("OPENROUTER_HTTP_REFERER", "https://example.com")
OPENROUTER_TITLE = os.environ.get("OPENROUTER_TITLE", "RAG Chart Extractor")
OPENROUTER_FORCE_JSON = (os.environ.get("OPENROUTER_FORCE_JSON", "true").lower() == "true")

# Input
INPUT_MODE = os.environ.get("INPUT_MODE", "folder").lower()
INPUT_IMAGE_DIR = os.environ.get("INPUT_IMAGE_DIR", "./data/images")
INPUT_IMAGE_PATH = os.environ.get("INPUT_IMAGE_PATH", "./data/sample.png")

# Output
OUTPUT_JSON_DIR = os.environ.get("OUTPUT_JSON_DIR", "./out/json")
OUTPUT_RAW_DIR = os.environ.get("OUTPUT_RAW_DIR", "./out/raw")

# Behavior
SAVE_NON_CHART_JSON = True
KEYWORDS_MIN = int(os.environ.get("KEYWORDS_MIN", "10"))
KEYWORDS_MAX = int(os.environ.get("KEYWORDS_MAX", "15"))
SAVE_RAW_RESPONSE = os.environ.get("SAVE_RAW_RESPONSE", "true").lower() == "true"
DEBUG = os.environ.get("DEBUG", "true").lower() == "true"
