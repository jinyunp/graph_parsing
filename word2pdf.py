[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "doc2pdf-batch"
version = "0.1.0"
description = "Batch convert .doc/.docx to PDF using docx2pdf or LibreOffice"
readme = "README.md"
requires-python = ">=3.8"
authors = [{name="Your Name"}]
keywords = ["docx2pdf", "LibreOffice", "batch", "convert", "pdf"]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

[project.optional-dependencies]
docx2pdf = ["docx2pdf>=0.1.8"]

[project.scripts]
doc2pdf-batch = "doc2pdf_batch.__main__:main"

[tool.setuptools]
packages = ["doc2pdf_batch"]







# doc2pdf-batch

워드 문서(.doc / .docx)가 들어있는 **입력 폴더**를 통째로 훑어서, 동일한 하위 디렉터리 구조로 **출력 폴더**에 PDF를 생성합니다.

## 주요 기능
- `docx2pdf`(Word 기반, Windows/macOS 고품질) → 실패 시 **LibreOffice(soffice)** 자동 시도
- 재귀 탐색, 상대 경로 구조 유지
- 이미 존재하는 PDF는 기본적으로 **건너뜀** (옵션으로 덮어쓰기 가능)
- **함수**와 **CLI** 모두 제공

## 설치

```bash
pip install build
python -m build
pip install dist/doc2pdf_batch-0.1.0-py3-none-any.whl

