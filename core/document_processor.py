"""Text extraction for PDF, DOCX, TXT, CSV and XLSX files, plus text cleaning."""
from __future__ import annotations

import csv
import io
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from core.errors import (
    DocumentProcessingError, EmptyDocumentError, ScannedPDFError, UnsupportedFileError,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".csv", ".xlsx")
ROWS_PER_SEGMENT = 25
MAX_TABLE_ROWS = 20000


@dataclass(frozen=True)
class Segment:
    """A piece of extracted text with a *real* location (never invented)."""
    text: str
    page: int | None = None
    location: str = ""


@dataclass
class ProcessedDocument:
    segments: list[Segment]
    warnings: list[str] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(len(s.text) for s in self.segments)


def clean_text(text: str) -> str:
    """Normalise unicode/whitespace while keeping paragraph breaks."""
    text = text.replace("\x00", "")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            if encoding == "utf-16" and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
                continue
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


# ---- loaders ------------------------------------------------------------
def _load_pdf(path: Path) -> ProcessedDocument:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            ok = reader.decrypt("")
        except Exception:
            ok = 0
        if not ok:
            raise DocumentProcessingError("This PDF is password-protected. Remove the password and upload it again.")
    pages = reader.pages
    n_pages = len(pages)
    if n_pages == 0:
        raise EmptyDocumentError("This PDF has no pages.")
    segments: list[Segment] = []
    warnings: list[str] = []
    unreadable = 0
    for number, page in enumerate(pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            logger.warning("Could not extract text from %s page %d", path.name, number, exc_info=True)
            raw = ""
        text = clean_text(raw)
        if text:
            segments.append(Segment(text=text, page=number, location=f"page {number}"))
        else:
            unreadable += 1
    total = sum(len(s.text) for s in segments)
    if total < 20:
        raise ScannedPDFError(
            "No extractable text was found. This looks like a scanned (image-only) PDF. "
            "OCR is not supported in this app; run the file through an OCR tool "
            "(for example OCRmyPDF or Adobe Acrobat) and upload the searchable version."
        )
    if unreadable:
        warnings.append(f"{unreadable} of {n_pages} pages had no extractable text (images or scans) and were skipped.")
    return ProcessedDocument(segments, warnings)


def _load_docx(path: Path) -> ProcessedDocument:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    segments: list[Segment] = []
    heading = ""
    lines: list[str] = []

    def flush() -> None:
        text = clean_text("\n".join(lines))
        if text:
            segments.append(Segment(text=text, location=f"section: {heading}" if heading else "document"))
        lines.clear()

    for block in doc.iter_inner_content():
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style = (block.style.name if block.style is not None else "") or ""
            if style.lower().startswith(("heading", "title")):
                flush()
                heading = text[:80]
                lines.append(f"# {text}")
            else:
                lines.append(text)
        elif isinstance(block, Table):
            for row in block.rows:
                cells = []
                for cell in row.cells:
                    value = cell.text.strip().replace("\n", " ")
                    if value and (not cells or cells[-1] != value):  # skip merged-cell repeats
                        cells.append(value)
                if cells:
                    lines.append(" | ".join(cells))
    flush()
    if not segments:
        raise EmptyDocumentError("This Word document contains no text.")
    return ProcessedDocument(segments)


def _load_txt(path: Path) -> ProcessedDocument:
    text = clean_text(decode_bytes(path.read_bytes()))
    if not text:
        raise EmptyDocumentError("This text file is empty.")
    return ProcessedDocument([Segment(text=text, location="text file")])


def _rows_to_segments(header: list[str], rows, label: str = "") -> tuple[list[Segment], bool]:
    """Group table rows into segments of `ROWS_PER_SEGMENT`, rendering each row as 'col: value'."""
    segments: list[Segment] = []
    batch: list[str] = []
    start = None
    truncated = False
    count = 0
    for row_number, row in rows:
        count += 1
        if count > MAX_TABLE_ROWS:
            truncated = True
            break
        parts = []
        for index, value in enumerate(row):
            if value is None or str(value).strip() == "":
                continue
            name = header[index] if index < len(header) and header[index] else f"column {index + 1}"
            parts.append(f"{name}: {str(value).strip()}")
        if not parts:
            continue
        if start is None:
            start = row_number
        batch.append("; ".join(parts))
        last = row_number
        if len(batch) >= ROWS_PER_SEGMENT:
            segments.append(Segment("\n".join(batch), location=f"{label}rows {start}-{last}"))
            batch, start = [], None
    if batch:
        segments.append(Segment("\n".join(batch), location=f"{label}rows {start}-{last}"))
    return segments, truncated


def _load_csv(path: Path) -> ProcessedDocument:
    text = decode_bytes(path.read_bytes())
    if not text.strip():
        raise EmptyDocumentError("This CSV file is empty.")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text), dialect)
    try:
        header = [h.strip() for h in next(reader)]
    except StopIteration:
        raise EmptyDocumentError("This CSV file is empty.") from None
    rows = ((i, r) for i, r in enumerate(reader, start=2))
    segments, truncated = _rows_to_segments(header, rows)
    if not segments:
        raise EmptyDocumentError("This CSV file has a header but no data rows.")
    warnings = [f"Only the first {MAX_TABLE_ROWS} rows were indexed."] if truncated else []
    return ProcessedDocument(segments, warnings)


def _load_xlsx(path: Path) -> ProcessedDocument:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    segments: list[Segment] = []
    warnings: list[str] = []
    try:
        for ws in wb.worksheets:
            row_iter = ((i, r) for i, r in enumerate(ws.iter_rows(values_only=True), start=1))
            header: list[str] = []
            header_row = 0
            for number, row in row_iter:
                if any(v is not None and str(v).strip() for v in row):
                    header = [str(v).strip() if v is not None else "" for v in row]
                    header_row = number
                    break
            if not header_row:
                continue
            seg, truncated = _rows_to_segments(header, row_iter, label=f"sheet '{ws.title}', ")
            segments.extend(seg)
            if truncated:
                warnings.append(f"Sheet '{ws.title}': only the first {MAX_TABLE_ROWS} rows were indexed.")
    finally:
        wb.close()
    if not segments:
        raise EmptyDocumentError("This Excel workbook contains no data rows.")
    return ProcessedDocument(segments, warnings)


_LOADERS: dict[str, Callable[[Path], ProcessedDocument]] = {
    ".pdf": _load_pdf, ".docx": _load_docx, ".txt": _load_txt,
    ".csv": _load_csv, ".xlsx": _load_xlsx,
}


def process_file(path: Path) -> ProcessedDocument:
    """Extract cleaned text segments from a document. Raises DocumentProcessingError subclasses."""
    ext = path.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFileError(
            f"Unsupported file type '{ext or 'unknown'}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    try:
        return loader(path)
    except DocumentProcessingError:
        raise
    except Exception as exc:
        logger.exception("Failed to process %s", path)
        raise DocumentProcessingError(
            f"'{path.name}' could not be read. The file may be corrupted or not a valid {ext[1:].upper()} file."
        ) from exc
