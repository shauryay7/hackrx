# app/document_parser.py

import tempfile
import requests
import mimetypes
from pathlib import Path

import fitz  # PyMuPDF
import docx


def download_file_from_url(url: str) -> Path:
    """Download a file from the given URL and store it temporarily."""
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download file: {response.status_code}")

    content_type = response.headers.get("content-type", "application/octet-stream")
    suffix = mimetypes.guess_extension(content_type) or ".bin"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(response.content)
    tmp.close()
    return Path(tmp.name)


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    doc = fitz.open(file_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    doc = docx.Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)


def load_document_from_url(url: str) -> str:
    """
    Download a file from a URL and extract its text.
    Supports .pdf and .docx formats.
    """
    file_path = download_file_from_url(url)

    if file_path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_path.suffix.lower() == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
