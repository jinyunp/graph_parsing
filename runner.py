import os, json, time
from typing import List
from config import (
    BACKEND, INPUT_MODE, INPUT_IMAGE_DIR, INPUT_IMAGE_PATH,
    OUTPUT_JSON_DIR, OUTPUT_RAW_DIR,
    SAVE_NON_CHART_JSON, SAVE_RAW_RESPONSE
)
from vlm_client import infer_chart_metadata_from_image
from schemas import to_json_dict

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def _ensure_dirs():
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    os.makedirs(OUTPUT_RAW_DIR, exist_ok=True)

def _list_images(folder: str) -> List[str]:
    images = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in SUPPORTED_EXTS:
                images.append(os.path.join(root, fn))
    return sorted(images)

def _save_json(meta, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_json_dict(meta), f, ensure_ascii=False, indent=2)
    print(f"    + 구조화 JSON 저장: {out_path}")

def _save_raw_pair(base_name: str, raw_text: str, raw_http_json: dict):
    raw_txt_path = os.path.join(OUTPUT_RAW_DIR, f"{base_name}.raw.txt")
    raw_json_path = os.path.join(OUTPUT_RAW_DIR, f"{base_name}.raw.http.json")
    with open(raw_txt_path, "w", encoding="utf-8") as f:
        f.write(raw_text or "")
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_http_json or {}, f, ensure_ascii=False, indent=2)
    print(f"    + 원본 응답 저장: {raw_txt_path}, {raw_json_path}")

def _is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTS

def process_path(img_path: str):
    print(f"  - 분석: {img_path}")
    base = os.path.splitext(os.path.basename(img_path))[0]

    raw_text_backup = ""
    raw_http_backup: dict = {}

    try:
        t0 = time.perf_counter()
        meta, raw_text, raw_http, timings = infer_chart_metadata_from_image(img_path)
        t1 = time.perf_counter()
        raw_text_backup, raw_http_backup = raw_text, raw_http

        gen_sec = timings.get("gen_sec", 0.0)
        struct_sec = timings.get("struct_sec", 0.0)
        retry_flag = " (kw-retry)" if timings.get("keywords_retry") else ""
        print(f"    + time: gen {gen_sec:.2f}s, struct {struct_sec:.2f}s, total {(t1 - t0):.2f}s{retry_flag}")

        if meta.is_chart or SAVE_NON_CHART_JSON:
            _save_json(meta, os.path.join(OUTPUT_JSON_DIR, f"{base}.json"))
        if SAVE_RAW_RESPONSE:
            _save_raw_pair(base, raw_text, raw_http)

    except Exception as e:
        print(f"    * step1 실패: {e}")
        if SAVE_RAW_RESPONSE:
            if not raw_text_backup:
                raw_text_backup = f"[EXCEPTION] {repr(e)}"
            _save_raw_pair(base, raw_text_backup, raw_http_backup)

def process_folder(img_dir: str):
    print(f"[Images folder] {img_dir}  (backend={BACKEND})")
    images = _list_images(img_dir)
    if not images:
        print("이미지 파일이 없습니다. PNG/JPG/JPEG/BMP/TIF/TIFF/WEBP 지원.")
        return
    for img in images:
        process_path(img)

def process_single(img_path: str):
    print(f"[Single image] {img_path}  (backend={BACKEND})")
    if not os.path.exists(img_path):
        print("경로에 파일이 없습니다.", img_path); return
    if not _is_supported(img_path):
        print("지원하지 않는 확장자입니다. PNG/JPG/JPEG/BMP/TIF/TIFF/WEBP", img_path); return
    process_path(img_path)

def main():
    _ensure_dirs()
    mode = os.environ.get("INPUT_MODE", "folder").lower()
    if mode == "folder":
        process_folder(os.environ.get("INPUT_IMAGE_DIR", "./data/images"))
    elif mode == "single":
        process_single(os.environ.get("INPUT_IMAGE_PATH", "./data/sample.png"))
    else:
        print(f"알 수 없는 INPUT_MODE='{mode}' (folder|single 중 선택)")

if __name__ == "__main__":
    main()
