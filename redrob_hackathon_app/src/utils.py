import logging
from typing import List, Dict, Any
from pypdf import PdfReader
from docx import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_file(file_bytes, extension: str) -> str:
    """Extracts raw string content out of document binary contents."""
    text = ""
    if extension == "pdf":
        reader = PdfReader(file_bytes)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif extension == "docx":
        doc = Document(file_bytes)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        text = file_bytes.read().decode("utf-8", errors="ignore")
    return text

def generate_explanation(cand: Dict[str, Any]) -> str:
    """Generates structured, deterministic analytical descriptions."""
    b = cand["breakdown"]
    skills = ", ".join(cand["skills"][:3])
    
    status = "Highly matching profile" if not cand["is_penalized"] else "Flags detected but holds strong alignment"
    return (f"{status} demonstrating a robust technical skill set including: {skills}. "
            f"Strongest score component was {max(b, key=b.get)} at {max(b.values())}%.")