import os, json, time
from typing import List
from config import (
    BACKEND,
    OUTPUT_JSON_DIR, OUTPUT_RAW_DIR, OUTPUT_SUMMARY_DIR,
    SAVE_RAW_RESPONSE
)
from vlm_client import generate_semantic_summary

def _ensure_dirs():
    os.makedirs(OUTPUT_SUMMARY_DIR, exist_ok=True)
    os.makedirs(OUTPUT_RAW_DIR, exist_ok=True)

def _list_jsons(folder: str) -> List[str]:
    out = []
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        if fn.lower().endswith(".json"):
            out.append(os.path.join(folder, fn))
    return sorted(out)

def _save_raw_pair(base_name: str, raw_text: str, raw_http_json: dict):
    raw_txt_path = os.path.join(OUTPUT_RAW_DIR, f"{base_name}.summary.raw.txt")
    raw_json_path = os.path.join(OUTPUT_RAW_DIR, f"{base_name}.summary.raw.http.json")
    with open(raw_txt_path, "w", encoding="utf-8") as f:
        f.write(raw_text or "")
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_http_json or {}, f, ensure_ascii=False, indent=2)
    print(f"    + 요약 원응답 저장: {raw_txt_path}, {raw_json_path}")

def _save_summary_text(base_name: str, summary_text: str):
    path = os.path.join(OUTPUT_SUMMARY_DIR, f"{base_name}.summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary_text or "")
    print(f"    + 의미 요약 저장: {path}")

def _load_meta(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    img = ((data.get("source") or {}).get("image_path")) or ""
    kws = list(data.get("key_phrases") or [])
    return img, kws

def process_json(json_path: str):
    base = os.path.splitext(os.path.basename(json_path))[0]
    print(f"  - 요약: {json_path}")

    try:
        image_path, keywords = _load_meta(json_path)
        if not image_path or not os.path.exists(image_path):
            print("    * image_path가 없거나 파일이 존재하지 않습니다 → 스킵")
            return
        if not keywords:
            print("    * key_phrases 비어있음 → 스킵")
            return

        t0 = time.perf_counter()
        summary_text, raw_http = generate_semantic_summary(image_path, keywords)
        t1 = time.perf_counter()

        _save_summary_text(base, summary_text)
        if SAVE_RAW_RESPONSE:
            _save_raw_pair(base, summary_text, raw_http)
        print(f"    + time: step2 summary {(t1 - t0):.2f}s")

    except Exception as e:
        print(f"    * step2 실패: {e}")

def main():
    _ensure_dirs()
    print(f"[Summary from JSONs] {OUTPUT_JSON_DIR}  (backend={BACKEND})")
    jsons = _list_jsons(OUTPUT_JSON_DIR)
    if not jsons:
        print("요약할 JSON이 없습니다. 먼저 runner.py를 실행하여 분석 결과를 생성하세요.")
        return
    for jp in jsons:
        process_json(jp)

if __name__ == "__main__":
    main()
