import os

# Ollama & 모델 설정
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
VLM_MODEL = "qwen2.5vl:3b"  # 고정

# 입력 모드: 'folder' 또는 'single'
INPUT_MODE = os.environ.get("INPUT_MODE", "folder").lower()

# 입력: 이미지 폴더 (INPUT_MODE='folder'일 때 사용)
INPUT_IMAGE_DIR = os.environ.get("INPUT_IMAGE_DIR", "./data/images")

# 입력: 단일 이미지 경로 (INPUT_MODE='single'일 때 사용)
INPUT_IMAGE_PATH = os.environ.get("INPUT_IMAGE_PATH", "./data/images/g_7.png")

# 출력: VLM 결과 JSON 저장 폴더
OUTPUT_JSON_DIR = os.environ.get("OUTPUT_JSON_DIR", "./out/json")

# 비차트인 경우에도 JSON 저장할지 여부
SAVE_NON_CHART_JSON = True
