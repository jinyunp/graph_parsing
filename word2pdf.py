# convert_docs_to_pdf.py
import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path

def is_tool_available(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def try_import_docx2pdf():
    try:
        from docx2pdf import convert as docx2pdf_convert
        return docx2pdf_convert
    except Exception:
        return None

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def convert_with_docx2pdf(docx2pdf_convert, src: Path, dst: Path) -> bool:
    """
    docx2pdf는 출력 경로가 파일이면 파일 단위 변환을 지원 (Windows/macOS, Word 필요)
    """
    try:
        ensure_dir(dst)
        # docx2pdf는 src가 파일일 때 dst에 '파일경로'를 넣으면 해당 파일로 출력
        docx2pdf_convert(str(src), str(dst))
        return True
    except Exception as e:
        print(f"[docx2pdf 실패] {src} -> {dst}: {e}", file=sys.stderr)
        return False

def convert_with_libreoffice(src: Path, dst: Path) -> bool:
    """
    LibreOffice(soffice) headless 모드. 출력 디렉터리 기준 변환이라 파일명을 우리가 맞춰야 함.
    실제로는 outdir에 같은 파일명.pdf로 생성됨 → 생성 후 원하는 위치로 이동.
    """
    if not is_tool_available("soffice"):
        print("[경고] LibreOffice 'soffice' 명령을 찾을 수 없습니다.", file=sys.stderr)
        return False
    try:
        tmp_outdir = dst.parent
        ensure_dir(dst)
        # --convert-to pdf:writer_pdf_Export도 가능하지만 기본 pdf로 충분
        cmd = [
            "soffice",
            "--headless",
            "--norestore",
            "--convert-to", "pdf",
            "--outdir", str(tmp_outdir),
            str(src)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # LibreOffice가 만든 파일은 원본파일명.pdf
        produced = tmp_outdir / (src.stem + ".pdf")
        if produced.exists():
            if dst.exists() and dst.resolve() != produced.resolve():
                dst.unlink()  # 덮어쓰기 대비
            # 파일명이 같으면 move 불필요, 경로만 다르면 이동
            if produced.resolve() != dst.resolve():
                produced.replace(dst)
            return True
        else:
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

def should_convert(path: Path) -> bool:
    return path.suffix.lower() in {".docx", ".doc"}

def main():
    parser = argparse.ArgumentParser(description="폴더 내 DOCX/DOC를 PDF로 일괄 변환")
    parser.add_argument("input_dir", type=str, help="입력 폴더 경로")
    parser.add_argument("--output", type=str, default=None, help="출력 루트 폴더 (기본: ./pdf)")
    parser.add_argument("--skip-existing", action="store_true", help="이미 존재하는 PDF는 건너뛰기")
    args = parser.parse_args()

    in_root = Path(args.input_dir).expanduser().resolve()
    if not in_root.exists() or not in_root.is_dir():
        print(f"[오류] 입력 폴더가 존재하지 않습니다: {in_root}", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.output).expanduser().resolve() if args.output else (Path.cwd() / "pdf")
    out_root.mkdir(parents=True, exist_ok=True)

    # 변환 백엔드 준비
    docx2pdf_convert = try_import_docx2pdf()
    has_docx2pdf = docx2pdf_convert is not None
    has_soffice = is_tool_available("soffice")

    if not has_docx2pdf and not has_soffice:
        print(
            "[오류] 사용할 수 있는 변환 도구가 없습니다.\n"
            " - 권장 1: pip install docx2pdf (Windows/macOS에서 MS Word 필요)\n"
            " - 권장 2: LibreOffice 설치 후 'soffice' 명령어 사용 가능하도록 PATH 설정",
            file=sys.stderr
        )
        sys.exit(2)

    total = 0
    converted = 0
    skipped = 0
    failed = 0

    for root, dirs, files in os.walk(in_root):
        root_path = Path(root)
        for name in files:
            src = root_path / name
            if not should_convert(src):
                continue

            # 상대 경로 보존해서 출력 경로 결정
            rel = src.relative_to(in_root)
            dst = (out_root / rel).with_suffix(".pdf")

            total += 1

            if args.skip_existing and dst.exists():
                print(f"[건너뜀] 이미 존재: {dst}")
                skipped += 1
                continue

            print(f"[변환] {src} -> {dst}")
            ok = False

            # 1) docx2pdf 우선
            if has_docx2pdf:
                ok = convert_with_docx2pdf(docx2pdf_convert, src, dst)

            # 2) 실패 시 LibreOffice 시도
            if not ok and has_soffice:
                ok = convert_with_libreoffice(src, dst)

            if ok and dst.exists():
                converted += 1
            else:
                print(f"[실패] {src}", file=sys.stderr)
                failed += 1

    print("\n==== 결과 ====")
    print(f"대상 파일: {total}")
    print(f"성공: {converted}")
    print(f"건너뜀: {skipped}")
    print(f"실패: {failed}")

    if failed > 0:
        print(
            "\n[힌트]\n"
            "- 파일이 암호화되어 있거나 손상된 경우 실패할 수 있습니다.\n"
            "- .doc(97-2003) 파일은 LibreOffice가 더 잘 처리하는 경우가 많습니다.\n"
            "- macOS/Windows에서 MS Word가 설치되어 있으면 docx2pdf 품질이 좋습니다.",
            file=sys.stderr
        )

if __name__ == "__main__":
    main()