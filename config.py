import os

# ===== Backend 선택: 'ollama' 또는 'openrouter' =====
BACKEND = os.environ.get("BACKEND", "ollama").lower()

# ===== OLLAMA 설정 =====
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:3b")  # 로컬 모델명

# ===== OPENROUTER 설정 =====
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen2.5-vl-32b-instruct:free")  # 예: qwen2.5-vl-72b-instruct
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_HTTP_REFERER = os.environ.get("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_TITLE = os.environ.get("OPENROUTER_TITLE", "RAG Chart Extractor")

# ===== 입력 모드: 'folder' 또는 'single' =====
INPUT_MODE = os.environ.get("INPUT_MODE", "folder").lower()

# 입력: 이미지 폴더 (INPUT_MODE='folder'일 때 사용)
INPUT_IMAGE_DIR = os.environ.get("INPUT_IMAGE_DIR", "./data/images")

# 입력: 단일 이미지 경로 (INPUT_MODE='single'일 때 사용)
INPUT_IMAGE_PATH = os.environ.get("INPUT_IMAGE_PATH", "./data/sample.png")

# 출력: 구조화 JSON 저장 폴더
OUTPUT_JSON_DIR = os.environ.get("OUTPUT_JSON_DIR", "./out/json")

# 출력: VLM 원본 응답(텍스트/JSON/메시지) 저장 폴더
OUTPUT_RAW_DIR = os.environ.get("OUTPUT_RAW_DIR", "./out/raw")

# 비차트인 경우에도 JSON 저장할지 여부
SAVE_NON_CHART_JSON = True

# 키워드 개수 제약 (프롬프트로 강제)
KEYWORDS_MIN = int(os.environ.get("KEYWORDS_MIN", "10"))
KEYWORDS_MAX = int(os.environ.get("KEYWORDS_MAX", "15"))

# 원본 응답 저장 여부
SAVE_RAW_RESPONSE = os.environ.get("SAVE_RAW_RESPONSE", "true").lower() == "true"
