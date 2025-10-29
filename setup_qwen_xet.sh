#!/usr/bin/env bash
# setup_qwen_xet.sh
# One-shot environment setup + (optional) HF login + clone Qwen/Qwen2.5-VL-3B-Instruct with git-xet/LFS or snapshot.
# Designed for Ubuntu (Runpod VS Code). Safe to re-run.

set -euo pipefail

# -----------------------------
# Defaults (can override via args)
# -----------------------------
REPO_ID="${REPO_ID:-Qwen/Qwen2.5-VL-3B-Instruct}"
CLONE_URL="https://huggingface.co/${REPO_ID}"
TARGET_DIR="${TARGET_DIR:-models/Qwen2.5-VL-3B-Instruct}"
HF_TOKEN="${HF_TOKEN:-}"                # also accepts --hf-token
METHOD="${METHOD:-git}"                 # "git" (git-xet) or "snapshot"
PYTHON_BIN="${PYTHON_BIN:-python3}"     # override if needed
PIP_BIN="${PIP_BIN:-pip3}"
EXTRA_PIP="${EXTRA_PIP:-}"

# New: requirements location hints (optional)
REQUIREMENTS_PATH="${REQUIREMENTS_PATH:-}"   # accepts --requirements-path
REQUIREMENTS_DIR="${REQUIREMENTS_DIR:-}"     # accepts --requirements-dir

# -----------------------------
# Args
# -----------------------------
usage() {
  cat <<USAGE
Usage: bash \$0 [options]

Options:
  --repo-id <org/name>          HF repo id (default: \${REPO_ID})
  --target-dir <path>           Destination directory (default: \${TARGET_DIR})
  --hf-token <token>            HF token for gated/private models / rate limit boost (optional)
  --method <git|snapshot>       Use git-xet (git) or huggingface_hub snapshot (snapshot). Default: git
  --extra-pip "<pkgs>"          Extra pip packages to install (quoted list). Optional.
  --requirements-path <file>    Path to requirements.txt (takes precedence if provided)
  --requirements-dir <dir>      Directory that contains requirements.txt (alternative to --requirements-path)
  -h, --help                    Show this help

Environment overrides:
  REPO_ID, TARGET_DIR, HF_TOKEN, METHOD, PYTHON_BIN, PIP_BIN, EXTRA_PIP, REQUIREMENTS_PATH, REQUIREMENTS_DIR

Examples:
  HF_TOKEN=hf_xxx bash \$0
  bash \$0 --method snapshot --target-dir /workspace/models/qwen3-4b
  bash \$0 --requirements-dir /workspace/project   # runner.py와 같은 디렉터리
  bash \$0 --requirements-path /workspace/project/requirements.txt
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id) REPO_ID="$2"; CLONE_URL="https://huggingface.co/${REPO_ID}"; shift 2;;
    --target-dir) TARGET_DIR="$2"; shift 2;;
    --hf-token) HF_TOKEN="$2"; shift 2;;
    --method) METHOD="$2"; shift 2;;
    --extra-pip) EXTRA_PIP="$2"; shift 2;;
    --requirements-path) REQUIREMENTS_PATH="$2"; shift 2;;
    --requirements-dir) REQUIREMENTS_DIR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[WARN] Unknown arg: $1" >&2; shift;;
  esac
done

echo "===== Settings ====="
echo "REPO_ID          : ${REPO_ID}"
echo "METHOD           : ${METHOD}"
echo "TARGET_DIR       : ${TARGET_DIR}"
echo "PYTHON_BIN       : ${PYTHON_BIN}"
echo "PIP_BIN          : ${PIP_BIN}"
echo "EXTRA_PIP        : ${EXTRA_PIP}"
echo "REQ_PATH(opt)    : ${REQUIREMENTS_PATH:-<auto>}"
echo "REQ_DIR (opt)    : ${REQUIREMENTS_DIR:-<auto>}"
echo "===================="

# -----------------------------
# Helpers
# -----------------------------
have_cmd() { command -v "$1" >/dev/null 2>&1; }

install_base_tools() {
  echo "[*] Installing base tools (git, git-lfs, curl, unzip, python3-pip)..."
  sudo apt-get update -y
  sudo apt-get install -y git git-lfs curl unzip python3 python3-pip
  git lfs install || true
}

ensure_git_xet_if_needed() {
  if [[ "${METHOD}" != "git" ]]; then
    echo "[*] METHOD != git, skipping git-xet installation."
    return 0
  fi
  if have_cmd git-xet; then
    echo "[*] git-xet already installed: $(git-xet --version || echo 'unknown version')"
  else
    echo "[*] Installing git-xet..."
    curl --proto '=https' --tlsv1.2 -sSf \
      https://raw.githubusercontent.com/huggingface/xet-core/refs/heads/main/git_xet/install.sh | bash
  fi
  echo "[*] Activating git-xet filters (global)"
  git-xet install || true
  git-xet --version || true
}

ensure_python_pkgs() {
  echo "[*] Upgrading pip & installing huggingface_hub[cli] ..."
  ${PIP_BIN} install -U pip
  ${PIP_BIN} install -U "huggingface_hub[cli]"
  if [[ -n "${EXTRA_PIP}" ]]; then
    echo "[*] Installing extra pip packages: ${EXTRA_PIP}"
    # shellcheck disable=SC2086
    ${PIP_BIN} install -U ${EXTRA_PIP}
  fi
}

hf_login_if_token() {
  if [[ -n "${HF_TOKEN}" ]]; then
    echo "[*] Logging-in to Hugging Face via token (non-interactive)..."
    ${PYTHON_BIN} - <<PY
import os, sys, subprocess
token=os.environ.get("HF_TOKEN","").strip()
if not token:
    sys.exit(0)
cmd = ["huggingface-cli","login","--token", token, "--add-to-git-credential"]
print("[CLI]", " ".join(cmd))
subprocess.run(cmd, check=False)
PY
  else
    echo "[*] HF token not provided. Skipping login."
  fi
}

_git_verify_and_lfs_pull() {
  echo "[*] Verifying large files presence..."
  local small_pointer
  small_pointer=$(find . -type f -name "*.safetensors" -size -4k | head -n 1 || true)
  if [[ -n "${small_pointer}" ]]; then
    echo "[!] Detected a possibly small/pointer file: ${small_pointer}"
    echo "[*] Running 'git lfs pull' to fetch binary content..."
    git lfs pull || true
  fi
}

clone_with_git_xet() {
  local url="$1"
  local dest="$2"
  if [[ -d "${dest}/.git" ]]; then
    echo "[*] Target directory already looks like a git repo: ${dest}"
  else
    mkdir -p "$(dirname "$dest")"
    echo "[*] Cloning with git (git-xet filters enabled): ${url} -> ${dest}"
    git clone "${url}" "${dest}"
  fi
  pushd "${dest}" >/dev/null
  _git_verify_and_lfs_pull
  echo "[*] Listing main files:"
  ls -lh | sed -n '1,80p'
  popd >/dev/null
}

download_with_snapshot() {
  local repo_id="$1"
  local dest="$2"
  echo "[*] Using huggingface_hub.snapshot_download (no git history): ${repo_id} -> ${dest}"
  mkdir -p "$(dirname "$dest")"
  ${PYTHON_BIN} - <<PY
from huggingface_hub import snapshot_download
repo_id="${repo_id}"
local_dir="${dest}"
print(f"Downloading snapshot of {repo_id} to {local_dir} ...")
snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
print("Done.")
PY
  echo "[*] Listing files:"
  ls -lh "${dest}" | sed -n '1,80p' || true
}

# -----------------------------
# Requirements installer (improved)
# -----------------------------
_install_requirements_file() {
  local req_file="$1"
  if [[ -f "${req_file}" ]]; then
    echo "[*] Installing requirements from: ${req_file}"
    ${PIP_BIN} install -r "${req_file}"
    return 0
  fi
  return 1
}

install_requirements_smart() {
  echo "[*] Resolving requirements.txt location..."

  # 1) If user provided explicit path
  if [[ -n "${REQUIREMENTS_PATH}" ]]; then
    if _install_requirements_file "${REQUIREMENTS_PATH}"; then
      echo "[*] Done installing from explicit REQUIREMENTS_PATH."
      return 0
    else
      echo "[WARN] REQUIREMENTS_PATH provided but not found: ${REQUIREMENTS_PATH}"
    fi
  fi

  # 2) If user provided a directory
  if [[ -n "${REQUIREMENTS_DIR}" ]]; then
    local candidate="${REQUIREMENTS_DIR%/}/requirements.txt"
    if _install_requirements_file "${candidate}"; then
      echo "[*] Done installing from REQUIREMENTS_DIR."
      return 0
    else
      echo "[WARN] requirements.txt not found under REQUIREMENTS_DIR: ${candidate}"
    fi
  fi

  # 3) Current working directory
  if _install_requirements_file "${PWD}/requirements.txt"; then
    echo "[*] Done installing from current directory."
    return 0
  fi

  # 4) Find runner.py, then use its directory/requirements.txt
  echo "[*] Searching for runner.py (depth<=3) to locate sibling requirements.txt ..."
  local runner_path
  runner_path="$(find "${PWD}" -maxdepth 3 -type f -name 'runner.py' | head -n 1 || true)"
  if [[ -n "${runner_path}" ]]; then
    local runner_dir
    runner_dir="$(dirname "${runner_path}")"
    local candidate="${runner_dir}/requirements.txt"
    if _install_requirements_file "${candidate}"; then
      echo "[*] Done installing from runner.py directory."
      return 0
    else
      echo "[WARN] runner.py found at ${runner_path}, but no requirements.txt next to it."
    fi
  else
    echo "[INFO] runner.py not found within depth 3 from ${PWD}."
  fi

  # 5) Fallback to TARGET_DIR (legacy behavior)
  local fallback="${TARGET_DIR}/requirements.txt"
  if _install_requirements_file "${fallback}"; then
    echo "[*] Done installing from TARGET_DIR (fallback)."
    return 0
  fi

  echo "[*] No requirements.txt found via smart search. Skipping."
  return 0
}

# -----------------------------
# Main
# -----------------------------
install_base_tools
ensure_git_xet_if_needed
ensure_python_pkgs
hf_login_if_token

if [[ "${METHOD}" == "git" ]]; then
  clone_with_git_xet "${CLONE_URL}" "${TARGET_DIR}"
elif [[ "${METHOD}" == "snapshot" ]]; then
  download_with_snapshot "${REPO_ID}" "${TARGET_DIR}"
else
  echo "[ERROR] Unknown METHOD: ${METHOD}" >&2
  exit 1
fi

install_requirements_smart

echo
echo "✅ All done."
echo "Repo path: ${TARGET_DIR}"
