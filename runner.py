import os
import json
import requests
from typing import List

from config import (
    INPUT_MODE, INPUT_IMAGE_DIR, INPUT_IMAGE_PATH,
    OUTPUT_JSON_DIR, SAVE_NON_CHART_JSON, OLLAMA_HOST, VLM_MODEL
)
from vlm_client import infer_chart_metadata_from_image
from schemas import to_json_dict

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def _ensure_dirs():
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

def _list_images(folder: str) -> List[str]:
    images = []
    for root, _, files in os.walk(folder):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SUPPORTED_EXTS:
                images.append(os.path.join(root, fn))
    return sorted(images)

def _is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTS

def _preflight():
    tags_url = f"{OLLAMA_HOST.rstrip('/')}/api/tags"
    try:
        r = requests.get(tags_url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"[사전 점검 실패] Ollama 서버에 연결할 수 없습니다: {tags_url}\n"
            f"- Ollama 실행: `ollama serve` 또는 `ollama run {VLM_MODEL}`\n"
            f"- OLLAMA_HOST 확인: 현재 '{OLLAMA_HOST}'\n"
            f"- 방화벽/프록시 확인\n원인: {e}"
        ) from e

    tags = r.json().get("models", [])
    names = {m.get("name") for m in tags}
    if VLM_MODEL not in names:
        raise RuntimeError(
            f"[모델 없음] '{VLM_MODEL}' 미설치.\n"
            f"- 설치: `ollama pull {VLM_MODEL}`\n"
            f"- 설치 목록: {sorted(list(names))}"
        )

def _save_json(meta, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_json_dict(meta), f, ensure_ascii=False, indent=2)
    print(f"    + 저장: {out_path}")

def process_folder(img_dir: str):
    print(f"[Images folder] {img_dir}")
    images = _list_images(img_dir)
    if not images:
        print("이미지 파일이 없습니다. PNG/JPG/JPEG/BMP/TIF/TIFF/WEBP 지원.")
        return

    for img_path in images:
        print(f"  - 분석: {img_path}")
        try:
            meta = infer_chart_metadata_from_image(img_path)
        except Exception as e:
            print(f"    * 실패: {e}")
            continue

        if not meta.is_chart and not SAVE_NON_CHART_JSON:
            print("    * 차트 아님 → 저장 생략 (config.SAVE_NON_CHART_JSON=False)")
            continue

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(OUTPUT_JSON_DIR, f"{base}.json")
        _save_json(meta, out_path)

def process_single(img_path: str):
    print(f"[Single image] {img_path}")
    if not os.path.exists(img_path):
        print("경로에 파일이 없습니다.", img_path)
        return
    if not _is_supported(img_path):
        print("지원하지 않는 확장자입니다. PNG/JPG/JPEG/BMP/TIF/TIFF/WEBP", img_path)
        return

    try:
        meta = infer_chart_metadata_from_image(img_path)
    except Exception as e:
        print(f"    * 실패: {e}")
        return

    if not meta.is_chart and not SAVE_NON_CHART_JSON:
        print("    * 차트 아님 → 저장 생략 (config.SAVE_NON_CHART_JSON=False)")
        return

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(OUTPUT_JSON_DIR, f"{base}.json")
    _save_json(meta, out_path)

def main():
    _ensure_dirs()
    _preflight()

    mode = INPUT_MODE.strip().lower()
    if mode == "folder":
        process_folder(INPUT_IMAGE_DIR)
    elif mode == "single":
        process_single(INPUT_IMAGE_PATH)
    else:
        print(f"알 수 없는 INPUT_MODE='{mode}' (folder|single 중 선택)")

if __name__ == "__main__":
    main()
