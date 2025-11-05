import argparse
from .core import convert_folder_to_pdf

def main():
    parser = argparse.ArgumentParser(description="DOC/DOCX → PDF 일괄 변환")
    parser.add_argument("input_dir", help="워드 문서들이 있는 폴더")
    parser.add_argument("output_dir", help="PDF를 저장할 별도 폴더")
    parser.add_argument("--no-skip", action="store_true", help="기존 PDF가 있어도 덮어쓰기")
    args = parser.parse_args()

    total, success, skipped, failed = convert_folder_to_pdf(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip,
    )
    print("==== 결과 ====")
    print(f"대상 파일: {total}")
    print(f"성공: {success}")
    print(f"건너뜀: {skipped}")
    print(f"실패: {failed}")

if __name__ == "__main__":
    main()