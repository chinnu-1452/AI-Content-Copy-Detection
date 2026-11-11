
from pypdf import PdfReader
from docx import Document
from pathlib import Path


def extract_text(file_path):

    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        reader = PdfReader(str(file_path))
        text = ""

        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"

        return text

    elif extension == ".docx":
        doc = Document(str(file_path))

        return "\n".join(
            paragraph.text for paragraph in doc.paragraphs
        )

    elif extension == ".txt":
        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    else:
        raise ValueError("Unsupported file format")
