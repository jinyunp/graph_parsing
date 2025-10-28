import os, json
from typing import List
from config import (
    INPUT_MODE, INPUT_IMAGE_DIR, INPUT_IMAGE_PATH,
    OUTPUT_JSON_DIR, OUTPUT_RAW_DIR, SAVE_NON_CHART_JSON, SAVE_RAW_RESPONSE,
    HF_MODEL_ID
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
            ext = os.path.splitext(fn)[1].lower()
            if ext in SUPPORTED_EXTS:
                images.append(os.path.join(root, fn))
    return sorted(images)

def _is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTS

def _preflight():
    print(f"[백엔드] HF Transformers / 모델: {HF_MODEL_ID}")

def _save_json(meta, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_json_dict(meta), f, ensure_ascii=False, indent=2)
    print(f"    + 구조화 JSON 저장: {out_path}")

def _save_raw(base_name: str, raw_text: str, raw_http_json: dict):
    raw_txt_path = os.path.join(OUTPUT_RAW_DIR, f"{base_name}.raw.txt")
    raw_json_path = os.path.join(OUTPUT_RAW_DIR, f"{base_name}.raw.http.json")
    with open(raw_txt_path, "w", encoding="utf-8") as f:
        f.write(raw_text)
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_http_json, f, ensure_ascii=False, indent=2)
    print(f"    + 원본 응답 저장: {raw_txt_path}, {raw_json_path}")

def process_path(img_path: str):
    print(f"  - 분석: {img_path}")
    base = os.path.splitext(os.path.basename(img_path))[0]
    raw_text_forced = ""
    raw_http_forced: dict = {}

    try:
        meta, raw_text, raw_http_json, timings = infer_chart_metadata_from_image(img_path)
        raw_text_forced, raw_http_forced = raw_text, raw_http_json  # 성공 시 raw 백업

        # 시간 로깅
        gen_sec = timings.get("gen_sec", 0.0)
        struct_sec = timings.get("struct_sec", 0.0)
        retry_flag = " (kw-retry)" if timings.get("keywords_retry") else ""
        print(f"    + time: gen {gen_sec:.2f}s, struct {struct_sec:.2f}s, total {(gen_sec+struct_sec):.2f}s{retry_flag}")

        out_path = os.path.join(OUTPUT_JSON_DIR, f"{base}.json")
        if meta.is_chart or SAVE_NON_CHART_JSON:
            _save_json(meta, out_path)
        else:
            print("    * 차트 아님 → 저장 생략 (config.SAVE_NON_CHART_JSON=False)")
        if SAVE_RAW_RESPONSE:
            _save_raw(base, raw_text, raw_http_json)

    except Exception as e:
        print(f"    * 실패: {e}")
        # 예외여도 raw 저장 시도
        if SAVE_RAW_RESPONSE:
            if not raw_text_forced:
                raw_text_forced = f"[EXCEPTION] {repr(e)}"
            _save_raw(base, raw_text_forced, raw_http_forced)

def process_folder(img_dir: str):
    print(f"[Images folder] {img_dir}")
    images = _list_images(img_dir)
    if not images:
        print("이미지 파일이 없습니다. PNG/JPG/JPEG/BMP/TIF/TIFF/WEBP 지원.")
        return
    for img_path in images:
        process_path(img_path)

def process_single(img_path: str):
    print(f"[Single image] {img_path}")
    if not os.path.exists(img_path):
        print("경로에 파일이 없습니다.", img_path); return
    if not _is_supported(img_path):
        print("지원하지 않는 확장자입니다. PNG/JPG/JPEG/BMP/TIF/TIFF/WEBP", img_path); return
    process_path(img_path)

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
