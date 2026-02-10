import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================
# Regex helpers
# =========================

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")
NUMBERED_PREFIX_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*(?:[)\].:-]\s*)?(.*)$")
FIG_CAPTION_RE = re.compile(r"^\s*(?:fig\.|figure)\s*([0-9]+(?:\.[0-9]+)?)\s*[:.\-]?\s*(.*)$", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^\s*table\s*([0-9]+(?:\.[0-9]+)?)\s*[:.\-]?\s*(.*)$", re.IGNORECASE)

FIG_REF_RE = re.compile(r"\b(?:fig\.|figure)\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)
TABLE_REF_RE = re.compile(r"\btable\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)

MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
MD_TABLE_HEADER_RE = re.compile(r"^\s*\|.*\|\s*$")
MD_TABLE_SEP_RE = re.compile(r"^\s*\|\s*:?-{2,}\s*(?:\|\s*:?-{2,}\s*)+\|\s*$")


@dataclass
class FigureItem:
    doc_id: str
    filename: str
    section_path: Optional[str]
    img_no: str
    caption: str
    image_link: Optional[str]
    page: Optional[int] = None


@dataclass
class TableItem:
    doc_id: str
    filename: str
    section_path: Optional[str]
    table_no: str
    caption: str
    table_md: str
    page: Optional[int] = None


def build_prefix(filename: str, section_path: Optional[str]) -> str:
    fp = f"[문서: {filename}]"
    sp = f" [경로: {section_path}]" if section_path else ""
    return fp + sp + "\n"


def is_numbered_title(title: str) -> bool:
    m = NUMBERED_PREFIX_RE.match(title.strip())
    if not m:
        return False
    num = (m.group(1) or "").strip()
    return bool(re.fullmatch(r"\d+(?:\.\d+)*", num))


def normalize_section_path(headings_stack: List[Tuple[int, str]]) -> str:
    return " > ".join([t for _, t in headings_stack])


def parse_markdown_tables(lines: List[str], start_idx: int) -> Tuple[Optional[str], int]:
    i = start_idx
    if i + 1 >= len(lines):
        return None, start_idx

    if not MD_TABLE_HEADER_RE.match(lines[i]):
        return None, start_idx
    if not MD_TABLE_SEP_RE.match(lines[i + 1]):
        return None, start_idx

    buf = [lines[i].rstrip("\n"), lines[i + 1].rstrip("\n")]
    i += 2
    while i < len(lines) and MD_TABLE_HEADER_RE.match(lines[i]):
        buf.append(lines[i].rstrip("\n"))
        i += 1

    return "\n".join(buf).strip() + "\n", i


def parse_doc(mmd_path: Path) -> Tuple[List[Dict[str, Any]], List[FigureItem], List[TableItem]]:
    filename = mmd_path.name
    doc_id = mmd_path.stem

    raw = mmd_path.read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines(True)

    headings_stack: List[Tuple[int, str]] = []
    text_blocks: List[Dict[str, Any]] = []
    figures: List[FigureItem] = []
    tables: List[TableItem] = []

    current_lines: List[str] = []
    current_title: Optional[str] = None
    current_section_path: Optional[str] = None

    def flush_text_block():
        nonlocal current_lines, current_title, current_section_path
        content = "".join(current_lines).strip()
        if content:
            text_blocks.append({
                "title": current_title,
                "section_path": current_section_path,
                "content": content
            })
        current_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        h = HEADING_RE.match(line.strip("\n"))
        if h:
            flush_text_block()

            level = len(h.group(1))
            title = h.group(2).strip()

            while headings_stack and headings_stack[-1][0] >= level:
                headings_stack.pop()
            headings_stack.append((level, title))

            current_title = title
            current_section_path = normalize_section_path(headings_stack)
            i += 1
            continue

        cap = FIG_CAPTION_RE.match(line.strip())
        if cap:
            img_no = cap.group(1).strip()
            caption_rest = (cap.group(2) or "").strip()
            caption_full = f"Fig. {img_no}" + (f": {caption_rest}" if caption_rest else "")

            image_link = None
            for k in range(1, 4):
                if i + k >= len(lines):
                    break
                mimg = MD_IMAGE_RE.search(lines[i + k])
                if mimg:
                    image_link = mimg.group(1).strip()
                    break

            figures.append(FigureItem(
                doc_id=doc_id,
                filename=filename,
                section_path=current_section_path,
                img_no=str(img_no),
                caption=caption_full,
                image_link=image_link
            ))

            # 캡션은 텍스트에 넣지 않음
            i += 1
            continue

        tcap = TABLE_CAPTION_RE.match(line.strip())
        if tcap:
            table_no = tcap.group(1).strip()
            caption_rest = (tcap.group(2) or "").strip()
            caption_full = f"Table {table_no}" + (f": {caption_rest}" if caption_rest else "")

            table_md = None
            next_i = i + 1
            scan_limit = min(len(lines), i + 25)
            j = i + 1
            while j < scan_limit:
                maybe_table, j2 = parse_markdown_tables(lines, j)
                if maybe_table:
                    table_md = maybe_table
                    next_i = j2
                    break
                j += 1

            if table_md is None:
                table_md = ""

            tables.append(TableItem(
                doc_id=doc_id,
                filename=filename,
                section_path=current_section_path,
                table_no=str(table_no),
                caption=caption_full,
                table_md=table_md
            ))

            i = next_i
            continue

        current_lines.append(line)
        i += 1

    flush_text_block()
    return text_blocks, figures, tables


def split_by_numbered_subheadings(content: str) -> List[str]:
    lines = content.splitlines()
    indices = []
    for idx, ln in enumerate(lines):
        if HEADING_RE.match(ln.strip()):
            continue
        m = NUMBERED_PREFIX_RE.match(ln.strip())
        if m and re.fullmatch(r"\d+(?:\.\d+)*", (m.group(1) or "").strip()):
            indices.append(idx)

    if len(indices) <= 1:
        return [content.strip()]

    chunks = []
    for s, e in zip(indices, indices[1:] + [len(lines)]):
        part = "\n".join(lines[s:e]).strip()
        if part:
            chunks.append(part)
    return chunks if chunks else [content.strip()]


def chunk_text_blocks(
    doc_id: str,
    filename: str,
    text_blocks: List[Dict[str, Any]],
    figures: List[FigureItem],
    tables: List[TableItem],
    max_chars: int = 2500
) -> List[Dict[str, Any]]:
    fig_path_map: Dict[str, Optional[str]] = {f.img_no: f.image_link for f in figures}
    table_path_map: Dict[str, str] = {t.table_no: f"{doc_id}::table::{t.table_no}" for t in tables}

    out: List[Dict[str, Any]] = []
    chunk_idx = 0

    for block in text_blocks:
        section_path = block.get("section_path")
        title = block.get("title") or ""
        content = block.get("content") or ""

        parts = split_by_numbered_subheadings(content) if is_numbered_title(title) else [content.strip()]

        for part in parts:
            p = part.strip()
            if not p:
                continue

            # 길이 분할
            if len(p) <= max_chars:
                subparts = [p]
            else:
                paras = [x.strip() for x in re.split(r"\n\s*\n", p) if x.strip()]
                subparts, buf = [], ""
                for para in paras:
                    if not buf:
                        buf = para
                    elif len(buf) + 2 + len(para) <= max_chars:
                        buf += "\n\n" + para
                    else:
                        subparts.append(buf)
                        buf = para
                if buf:
                    subparts.append(buf)

                final_subparts = []
                for sp in subparts:
                    if len(sp) <= max_chars:
                        final_subparts.append(sp)
                    else:
                        for k in range(0, len(sp), max_chars):
                            final_subparts.append(sp[k:k + max_chars])
                subparts = final_subparts

            for sp in subparts:
                fig_refs = [m.group(1) for m in FIG_REF_RE.finditer(sp)]
                table_refs = [m.group(1) for m in TABLE_REF_RE.finditer(sp)]

                multi_list: List[str] = []
                multi_path: List[str] = []

                for no in fig_refs:
                    key = f"fig_{no}"
                    if key not in multi_list:
                        multi_list.append(key)
                        multi_path.append(fig_path_map.get(no) or "")

                for no in table_refs:
                    key = f"table_{no}"
                    if key not in multi_list:
                        multi_list.append(key)
                        multi_path.append(table_path_map.get(no, ""))

                prefix = build_prefix(filename, section_path)
                item: Dict[str, Any] = {
                    "id": f"{doc_id}#c{chunk_idx}",
                    "filename": filename,
                    "section_path": section_path,
                    "text": prefix + sp.strip(),
                    "multi_data_list": multi_list,
                    "multi_data_path": multi_path,
                }
                out.append(item)
                chunk_idx += 1

    return out


def build_images_sum_final(figures: List[FigureItem]) -> List[Dict[str, Any]]:
    out = []
    for f in figures:
        prefix = build_prefix(f.filename, f.section_path)
        original = f.caption.strip() if f.caption.strip() else "No Description"
        out.append({
            "id": f"{f.doc_id}#fig{f.img_no}",
            "placeholder": None,
            "component_type": "image",
            "original": original,
            "text": prefix + original,
            "keyword": [],
            "image_link": f.image_link,
            "section_path": f.section_path,
            "filename": f.filename,
            "page": f.page,
            "img_no": f.img_no,
        })
    return out


def build_tables_str_final(tables: List[TableItem]) -> List[Dict[str, Any]]:
    out = []
    for t in tables:
        prefix = build_prefix(t.filename, t.section_path)
        original = {
            "caption": t.caption.strip() if t.caption.strip() else "No Description",
            "table_md": t.table_md.strip()
        }
        out.append({
            "id": f"{t.doc_id}#table{t.table_no}",
            "component_type": "table",
            "original": original,
            "text": prefix + original["caption"],
            "image_link": None,
            "section_path": t.section_path,
            "filename": t.filename,
            "page": t.page,
            "placeholder": None,
            "table_no": t.table_no,
        })
    return out


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run(mmd_dir: Path, out_dir: Path):
    mmd_files = sorted(mmd_dir.rglob("*.mmd"))
    if not mmd_files:
        raise FileNotFoundError(f"No .mmd files found under: {mmd_dir}")

    all_texts: List[Dict[str, Any]] = []
    all_tables_str: List[Dict[str, Any]] = []
    all_tables_unstr: List[Dict[str, Any]] = []
    all_images_formula: List[Dict[str, Any]] = []
    all_images_sum: List[Dict[str, Any]] = []
    all_images_trans: List[Dict[str, Any]] = []

    for mmd_path in mmd_files:
        text_blocks, figures, tables = parse_doc(mmd_path)
        doc_id = mmd_path.stem
        filename = mmd_path.name

        all_texts.extend(chunk_text_blocks(
            doc_id=doc_id,
            filename=filename,
            text_blocks=text_blocks,
            figures=figures,
            tables=tables,
            max_chars=2500
        ))

        all_images_sum.extend(build_images_sum_final(figures))
        all_tables_str.extend(build_tables_str_final(tables))

    final_dir = out_dir / "final"
    write_json(final_dir / "texts_final.json", all_texts)
    write_json(final_dir / "tables_str_final.json", all_tables_str)
    write_json(final_dir / "tables_unstr_final.json", all_tables_unstr)
    write_json(final_dir / "images_formula_final.json", all_images_formula)
    write_json(final_dir / "images_sum_final.json", all_images_sum)
    write_json(final_dir / "images_trans_final.json", all_images_trans)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmd_dir", type=str, required=True, help="Directory containing .mmd files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory (will create out_dir/final/...)")
    args = parser.parse_args()
    run(Path(args.mmd_dir), Path(args.out_dir))
