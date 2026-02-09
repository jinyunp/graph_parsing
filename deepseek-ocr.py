from pathlib import Path
import fitz
from transformers import AutoModel, AutoTokenizer
from contextlib import redirect_stdout
import torch
from tqdm import tqdm
import io
import os
import re

from config import DEEPSEEK_MODEL_ID

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = DEEPSEEK_MODEL_ID

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation="flash_attention_2",
    trust_remote_code=True,
    use_safetensors=True,
)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\n<|grounding|>Convert the document to markdown. The layout of the document shows both sides in one image. \
    Please separate the content of the left and right sides into two sections in the markdown output.\n<|end|>\n"


# -----------------------------------
# 🔹 PDF 입력
# -----------------------------------
pdf_path = Path("/root/graph_parsing/data/docs/Iron Making Text Book 2008.pdf")

# 👉 페이지 범위 설정 (1-index 기준)
START_PAGE = 9     # None이면 처음부터
END_PAGE = 12     # None이면 끝까지

# 파일명만 추출 (확장자 제거)
pdf_stem = pdf_path.stem  # ex) contract_2024

# 출력 루트
base_output_dir = Path("data/output")

# 👉 data/output/{파일명}/ 자동 생성
output_root = base_output_dir / pdf_stem
output_root.mkdir(parents=True, exist_ok=True)

print(f"📂 Output directory: {output_root.resolve()}")

# -----------------------------------
# PDF 렌더링
# -----------------------------------
DPI = 200
ZOOM = DPI / 72.0

doc = fitz.open(pdf_path)
total_pages = len(doc)
all_markdown = []

# 페이지 범위 보정
if START_PAGE is None:
    START_PAGE = 1
if END_PAGE is None:
    END_PAGE = total_pages

# 유효성 체크
if START_PAGE < 1 or END_PAGE > total_pages or START_PAGE > END_PAGE:
    raise ValueError(
        f"Invalid page range: {START_PAGE}~{END_PAGE} (Total pages: {total_pages})"
    )

print(f"📄 Processing pages {START_PAGE} ~ {END_PAGE} / {total_pages}")

# 0-index 변환
start_idx = START_PAGE - 1
end_idx = END_PAGE  # python range에서 끝은 미포함이라 그대로 사용

DPI = 200
ZOOM = DPI / 72.0

all_markdown = []

for page_idx in tqdm(range(start_idx, end_idx), desc="Processing pages"):
    page = doc.load_page(page_idx)

    mat = fitz.Matrix(ZOOM, ZOOM)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    page_number = page_idx + 1
    page_img_path = output_root / f"page_{page_number:04d}.png"
    pix.save(str(page_img_path))

    page_out_dir = output_root / f"page_{page_number:04d}"
    page_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 Processing page {page_number} ...")

    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull):
            res = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=str(page_img_path),
                output_path=str(page_out_dir),
                base_size=1024,
                image_size=768,
                crop_mode=True,
                save_results=True,
            )

    print(f"✅ Page {page_number} done.")


    if isinstance(res, str):
        page_md = res
    else:
        page_md = getattr(res, "text", None) or repr(res)

    all_markdown.append(f"\n\n<!-- Page {page_number} -->\n\n{page_md}")

# -----------------------------------
# 최종 저장
# -----------------------------------
final_md_path = output_root / f"{pdf_stem}_{START_PAGE}-{END_PAGE}.md"
final_md_path.write_text("".join(all_markdown), encoding="utf-8")

print(f"✅ Done. Saved to: {final_md_path.resolve()}")