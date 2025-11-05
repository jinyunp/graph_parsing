import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

def _is_tool_available(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def _try_import_docx2pdf():
    try:
        from docx2pdf import convert as docx2pdf_convert
        return docx2pdf_convert
    except Exception:
        return None

def _ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _convert_with_docx2pdf(docx2pdf_convert, src: Path, dst: Path) -> bool:
    try:
        _ensure_parent_dir(dst)
        docx2pdf_convert(str(src), str(dst))
        return True
    except Exception as e:
        print(f"[docx2pdf 실패] {src} -> {dst}: {e}", file=sys.stderr)
        return False

def _convert_with_libreoffice(src: Path, dst: Path) -> bool:
    if not _is_tool_available("soffice"):
        print("[경고] LibreOffice 'soffice' 명령을 찾을 수 없습니다.", file=sys.stderr)
        return False
    try:
        _ensure_parent_dir(dst)
        outdir = dst.parent
        cmd = [
            "soffice",
            "--headless",
            "--norestore",
            "--convert-to", "pdf",
            "--outdir", str(outdir),
            str(src)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        produced = outdir / (src.stem + ".pdf")
        if produced.exists():
            if produced.resolve() != dst.resolve():
                produced.replace(dst)
            return True
        print(f"[LibreOffice 실패] 출력 파일이 보이지 않습니다: {produced}", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:
        out = e.stdout.decode("utf-8", errors="ignore") if e.stdout else ""
        err = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
        print(f"[LibreOffice 에러] {src}\nSTDOUT:\n{out}\nSTDERR:\n{err}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[LibreOffice 예외] {src}: {e}", file=sys.stderr)
        return False

def _should_convert(path: Path) -> bool:
    return path.suffix.lower() in {".docx", ".doc"}

def convert_folder_to_pdf(
    input_dir: str,
    output_dir: str,
    skip_existing: bool = True,
) -> Tuple[int, int, int, int]:
    """
    폴더 내 .docx/.doc 파일을 재귀적으로 찾아 PDF로 변환합니다.
    - MS Word + docx2pdf가 있으면 우선 사용
    - 아니면 LibreOffice(soffice) 사용
    반환값: (총 대상, 성공, 건너뜀, 실패)
    """
    in_root = Path(input_dir).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()
    if not in_root.exists() or not in_root.is_dir():
        raise FileNotFoundError(f"입력 폴더가 존재하지 않거나 폴더가 아닙니다: {in_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    docx2pdf_convert = _try_import_docx2pdf()
    has_docx2pdf = docx2pdf_convert is not None
    has_soffice = _is_tool_available("soffice")
    if not has_docx2pdf and not has_soffice:
        raise RuntimeError("사용 가능한 변환 도구가 없습니다. 'pip install docx2pdf' 또는 LibreOffice 설치 후 'soffice' 사용")

    total = success = skipped = failed = 0

    for root, _, files in os.walk(in_root):
        root_path = Path(root)
        for name in files:
            src = root_path / name
            if not _should_convert(src):
                continue
            rel = src.relative_to(in_root)
            dst = (out_root / rel).with_suffix(".pdf")

            total += 1
            if skip_existing and dst.exists():
                skipped += 1
                continue

            ok = False
            if has_docx2pdf:
                ok = _convert_with_docx2pdf(docx2pdf_convert, src, dst)
            if not ok and has_soffice:
                ok = _convert_with_libreoffice(src, dst)

            if ok and dst.exists():
                success += 1
            else:
                failed += 1

    return total, success, skipped, failed