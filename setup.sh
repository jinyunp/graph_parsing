#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------
# Defaults
# ---------------------------------
VERSION="${VERSION:-qwen}"            # qwen | deepseek
METHOD="${METHOD:-snapshot}"          # snapshot | git
HF_TOKEN="${HF_TOKEN:-}"

BASE_MODEL_DIR="/workspace/models"
BASE_VENV_DIR=".venv"                # venv base directory
PYTHON_SYS="python3"

# ---------------------------------
# Args
# ---------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2;;
    --method) METHOD="$2"; shift 2;;
    --hf-token) HF_TOKEN="$2"; shift 2;;
    *) echo "[WARN] Unknown arg: $1" >&2; shift;;
  esac
done

# ---------------------------------
# Version preset
# ---------------------------------
case "${VERSION}" in
  qwen)
    REPO_ID="Qwen/Qwen2.5-VL-3B-Instruct"
    MODEL_NAME="Qwen2.5-VL-3B-Instruct"
    REQ_FILE="requirements.qwen.txt"
    ;;
  deepseek)
    REPO_ID="deepseek-ai/DeepSeek-OCR-2"
    MODEL_NAME="DeepSeek-OCR-2"
    REQ_FILE="requirements.deepseek.txt"
    ;;
  *)
    echo "[ERROR] --version must be qwen or deepseek" >&2
    exit 1
    ;;
esac

TARGET_DIR="${BASE_MODEL_DIR}/${MODEL_NAME}"
VENV_DIR="${BASE_VENV_DIR}/${VERSION}"
CLONE_URL="https://huggingface.co/${REPO_ID}"

echo "========== SETTINGS =========="
echo "VERSION     : ${VERSION}"
echo "METHOD      : ${METHOD}"
echo "REPO_ID     : ${REPO_ID}"
echo "MODEL PATH  : ${TARGET_DIR}"
echo "VENV PATH   : ${VENV_DIR}"
echo "REQ FILE    : ${REQ_FILE}"
echo "=============================="

# ---------------------------------
# Helpers
# ---------------------------------
have_cmd() { command -v "$1" >/dev/null 2>&1; }

as_root() {
  if have_cmd sudo; then
    sudo "$@"
  else
    "$@"
  fi
}

require_cmd_or_fail() {
  local cmd="$1"
  if ! have_cmd "${cmd}"; then
    echo "[ERROR] Missing command: ${cmd}" >&2
    exit 1
  fi
}

# ---------------------------------
# Base tools
# ---------------------------------
install_base_tools() {
  echo "[*] Installing base tools (git, git-lfs, curl, unzip, python3-pip)..."
  as_root apt-get update -y
  as_root apt-get install -y git git-lfs curl unzip python3 python3-pip python3-venv
  git lfs install || true
}

# ---------------------------------
# VENV create + activate
# ---------------------------------
ensure_venv() {
  echo "[*] Ensuring venv: ${VENV_DIR}"
  mkdir -p "${BASE_VENV_DIR}"

  if [[ ! -d "${VENV_DIR}" ]]; then
    ${PYTHON_SYS} -m venv "${VENV_DIR}"
  fi

  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"

  PYTHON_BIN="${VENV_DIR}/bin/python"
  PIP_BIN="${VENV_DIR}/bin/pip"

  echo "[*] Using python: ${PYTHON_BIN}"
  ${PIP_BIN} install -U pip setuptools wheel
}

# ---------------------------------
# HF login (optional)
# ---------------------------------
hf_login_if_token() {
  echo "[*] Installing huggingface_hub[cli]..."
  ${PIP_BIN} install -U "huggingface_hub[cli]"

  if [[ -n "${HF_TOKEN}" ]]; then
    echo "[*] HuggingFace login with token..."
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || true
  else
    echo "[*] HF_TOKEN not provided. Skipping login."
  fi
}

# ---------------------------------
# Install requirements
# ---------------------------------
install_requirements() {
  echo "[*] Installing requirements from: ${REQ_FILE}"
  if [[ ! -f "${REQ_FILE}" ]]; then
    echo "[ERROR] Requirements file not found: ${REQ_FILE}" >&2
    echo "        Create it in the same directory as setup.sh." >&2
    exit 1
  fi
  ${PIP_BIN} install -r "${REQ_FILE}"
}

# ---------------------------------
# Download model
# ---------------------------------
download_model() {
  echo "[*] Downloading model to: ${TARGET_DIR}"
  mkdir -p "${BASE_MODEL_DIR}"

  if [[ -d "${TARGET_DIR}" ]]; then
    echo "[*] Model already exists. Skipping download."
    return 0
  fi

  if [[ "${METHOD}" == "git" ]]; then
    echo "[*] Cloning via git: ${CLONE_URL}"
    git clone "${CLONE_URL}" "${TARGET_DIR}"
  elif [[ "${METHOD}" == "snapshot" ]]; then
    echo "[*] snapshot_download: ${REPO_ID}"
    ${PYTHON_BIN} - <<PY
from huggingface_hub import snapshot_download
snapshot_download("${REPO_ID}", local_dir="${TARGET_DIR}", local_dir_use_symlinks=False)
print("Done.")
PY
  else
    echo "[ERROR] Unknown METHOD: ${METHOD} (use git|snapshot)" >&2
    exit 1
  fi
}

# ---------------------------------
# DeepSeek extra deps (only if needed)
# ---------------------------------
install_deepseek_extra() {
  if [[ "${VERSION}" != "deepseek" ]]; then
    return 0
  fi

  echo "[*] Installing DeepSeek extras (torch/cu118 + psutil + flash-attn)..."

  # 빌드 툴(없으면 flash-attn 빌드 단계에서 실패하기 쉬움)
  as_root apt-get update -y
  as_root apt-get install -y build-essential ninja-build git

  # torch를 먼저 고정 설치 (이미 설치되어 있어도 재확인 차원)
  ${PIP_BIN} install -U \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

  # flash-attn이 메타데이터 단계에서 요구할 수 있는 패키지
  ${PIP_BIN} install -U psutil

  # flash-attn 설치 (너가 성공한 커맨드 그대로)
  ${PIP_BIN} install flash-attn==2.7.3 --no-build-isolation
}


# ---------------------------------
# Main
# ---------------------------------
install_base_tools
ensure_venv
hf_login_if_token
install_requirements
download_model
install_deepseek_extra

echo
echo "✅ Setup complete!"
echo "Model stored at: ${TARGET_DIR}"
echo "Activate venv:"
echo "  source ${VENV_DIR}/bin/activate"
