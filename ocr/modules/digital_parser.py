# Digital Parser module (Person 4)
# Contains pdf2docx logic
# digital_parser.py

# run pip install pdf2docx before starting this mode
# run pip install PyMuPDF before starting this mode


from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

try:
    from pdf2docx import Converter
except Exception:
    Converter = None  # type: ignore

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore


logger = logging.getLogger(__name__)


class DigitalParser:

    def __init__(self):
        if Converter is None:
            raise RuntimeError(
                "pdf2docx is not installed. Run: pip install pdf2docx"
            )
        if fitz is None:
            logger.warning(
                "PyMuPDF is not installed. Run: pip install PyMuPDF"
            )

    # ---------------------------------------------------------
    # 1. Check whether PDF is a digital version or scanned version
    # ---------------------------------------------------------

    # Algorithm explaination:
    # -Check that the PDF file exists and that fitz is installed.

    # -Open the PDF and read the first page.

    # -Extract text and strip whitespace.

    # -If text is too short → likely a scanned image → return False.

    # -Compute text density = number of characters / page area.

    # -Compare density to min_text_ratio.

    #    +High → digital → True

    #    +Low → scanned → False

    # -Catch exceptions to avoid crashing.

    @staticmethod
    def is_digital_pdf(pdf_path: str | Path, min_text_ratio: float = 0.001) -> bool:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File does not exist")

        if fitz is None:
            return False

        try:
            doc = fitz.open(str(pdf_path))
            page = doc.load_page(0)

            text = page.get_text().strip()
            if len(text) < 10:
                return False

            # Measure the thickness of words on a page
            rect = page.rect
            area = rect.width * rect.height
            if area == 0:
                return False

            ratio = len(text) / area
            return ratio >= min_text_ratio

        except Exception:
            return False

    # ---------------------------------------------------------
    # 2. Convert PDF digital → DOCX
    # ---------------------------------------------------------
    @staticmethod
    def convert(
        pdf_path: str | Path,
        docx_path: Optional[str | Path] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        force: bool = False,
    ) -> Path:
        """
        Change PDF digital → DOCX.

        Args:
            pdf_path: source of pdf
            docx_path: file .docx destionation (default = change .pdf → .docx)
            start, end: page limits (0-indexed)
            force: if PDF is not digital, automatically warn errors
                   but if force=True, it will be converted no matter what.
        """
        if Converter is None:
            raise RuntimeError("pdf2docx is not installed.")

        pdf_path = Path(pdf_path)

        if docx_path is None:
            docx_path = pdf_path.with_suffix(".docx")
        docx_path = Path(docx_path)

        # Auto-detect PDF digital
        is_digital = DigitalParser.is_digital_pdf(pdf_path)

        if not is_digital and not force:
            raise RuntimeError(
                "PDF is not a digital version "
                "Please use OCR pipeline or set force=True."
            )

        logger.info("Converting PDF (digital) → DOCX: %s", pdf_path)
        cv = None
        try:
            cv = Converter(str(pdf_path))
            s = 0 if start is None else int(start)
            e = None if end is None else int(end)

            cv.convert(str(docx_path), start=s, end=e)
            logger.info("Successfully converted → %s", docx_path)
            return docx_path

        except Exception as e:
            raise RuntimeError(f"pdf2docx failed to convert: {e}")

        finally:
            if cv:
                cv.close()
