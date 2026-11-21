import csv

import pytest

from core.document_processor import clean_text, process_file
from core.errors import (
    DocumentProcessingError, EmptyDocumentError, ScannedPDFError, UnsupportedFileError,
)


def make_pdf(path, pages):
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    path.write_bytes(bytes(out))


def test_clean_text_normalises_whitespace():
    assert clean_text("a  b\t c\r\n\r\n\r\n\r\nd \x00") == "a b c\n\nd"


def test_pdf_extracts_text_with_real_page_numbers(tmp_path):
    p = tmp_path / "doc.pdf"
    make_pdf(p, ["First page about apples", "Second page about oranges"])
    result = process_file(p)
    assert [s.page for s in result.segments] == [1, 2]
    assert "oranges" in result.segments[1].text
    assert result.segments[0].location == "page 1"


def test_pdf_with_partially_empty_pages_warns(tmp_path):
    p = tmp_path / "mixed.pdf"
    make_pdf(p, ["Some real text on page one", ""])
    result = process_file(p)
    assert len(result.segments) == 1 and result.warnings


def test_scanned_pdf_gives_ocr_message(tmp_path):
    p = tmp_path / "scan.pdf"
    make_pdf(p, ["", ""])
    with pytest.raises(ScannedPDFError) as err:
        process_file(p)
    assert "OCR" in str(err.value)


def test_corrupted_pdf_is_reported(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"this is not a pdf at all")
    with pytest.raises(DocumentProcessingError):
        process_file(p)


def test_txt_encodings_and_empty(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("caf\u00e9 notes".encode("cp1252"))
    assert "caf" in process_file(p).segments[0].text
    e = tmp_path / "empty.txt"
    e.write_text("  \n ")
    with pytest.raises(EmptyDocumentError):
        process_file(e)


def test_csv_rows_become_labelled_segments(tmp_path):
    p = tmp_path / "d.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "city"])
        for i in range(30):
            w.writerow([f"person{i}", "Paris"])
    segs = process_file(p).segments
    assert len(segs) == 2
    assert segs[0].location == "rows 2-26"
    assert "name: person0; city: Paris" in segs[0].text


def test_xlsx_sheets(tmp_path):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["region", "revenue"])
    ws.append(["North", 100])
    ws2 = wb.create_sheet("Empty")
    wb.save(tmp_path / "s.xlsx")
    segs = process_file(tmp_path / "s.xlsx").segments
    assert len(segs) == 1
    assert "sheet 'Sales'" in segs[0].location
    assert "region: North; revenue: 100" in segs[0].text


def test_docx_headings_and_tables(tmp_path):
    from docx import Document

    d = Document()
    d.add_heading("Introduction", level=1)
    d.add_paragraph("This report covers solar energy.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text = "Year"
    t.cell(0, 1).text = "Output"
    t.cell(1, 0).text = "2023"
    t.cell(1, 1).text = "50 GW"
    d.save(tmp_path / "r.docx")
    segs = process_file(tmp_path / "r.docx").segments
    assert segs[0].location == "section: Introduction"
    assert "solar energy" in segs[0].text and "2023 | 50 GW" in segs[0].text


def test_unsupported_extension(tmp_path):
    p = tmp_path / "x.exe"
    p.write_bytes(b"x")
    with pytest.raises(UnsupportedFileError):
        process_file(p)
