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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=dtype,
    attn_implementation="flash_attention_2",
    device_map="cuda" if device.type == "cuda" else None,
    use_safetensors=True,
)
model = model.to(device).eval()


prompt = "<image>\n<|grounding|>Convert the document to markdown. The layout of the document may show both sides in one image or just one page in a image. \
    If the image shows two sides, please separate the content of the left and right sides into two sections in the markdown output.\n<|end|>\n"


# -----------------------------------
# 🔹 PDF 입력
# -----------------------------------
# pdf_path = Path("/root/graph_parsing/data/docs/Iron Making Text Book 2008.pdf")
docs_dir = Path("/root/graph_parsing/data/docs")
pdf_paths = sorted(docs_dir.glob("*.pdf"))
if not pdf_paths:
    raise FileNotFoundError(f"No PDF files found in: {docs_dir}")


# 👉 페이지 범위 설정 (1-index 기준)
START_PAGE = None     # None이면 처음부터
END_PAGE = None     # None이면 끝까지

DPI = 200 
ZOOM = DPI / 72.0

base_output_dir = Path("data/output")
base_output_dir.mkdir(parents=True, exist_ok=True)

for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
    pdf_stem = pdf_path.stem
    output_root = base_output_dir / pdf_stem
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\n📄 PDF: {pdf_path}")
    print(f"📂 Output directory: {output_root.resolve()}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # 페이지 범위 보정 (문서별로 total_pages 반영)
    start_page = 1 if START_PAGE is None else START_PAGE
    end_page = total_pages if END_PAGE is None else END_PAGE

    if start_page < 1 or end_page > total_pages or start_page > end_page:
        raise ValueError(
            f"Invalid page range for {pdf_path.name}: {start_page}~{end_page} (Total pages: {total_pages})"
        )

    print(f"📄 Processing pages {start_page} ~ {end_page} / {total_pages}")

    start_idx = start_page - 1
    end_idx = end_page  # range 끝 미포함

    all_markdown = []

    for page_idx in tqdm(range(start_idx, end_idx), desc=f"Pages ({pdf_path.name})", leave=False):
        page = doc.load_page(page_idx)

        mat = fitz.Matrix(ZOOM, ZOOM)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        page_number = page_idx + 1
        page_img_path = output_root / f"page_{page_number:04d}.png"
        pix.save(str(page_img_path))

        page_out_dir = output_root / f"page_{page_number:04d}"
        page_out_dir.mkdir(parents=True, exist_ok=True)

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

        if isinstance(res, str):
            page_md = res
        else:
            page_md = getattr(res, "text", None) or repr(res)

        all_markdown.append(f"\n\n<!-- Page {page_number} -->\n\n{page_md}")

    final_md_path = output_root / f"{pdf_stem}_{start_page}-{end_page}.md"
    final_md_path.write_text("".join(all_markdown), encoding="utf-8")
    print(f"✅ Done. Saved to: {final_md_path.resolve()}")
