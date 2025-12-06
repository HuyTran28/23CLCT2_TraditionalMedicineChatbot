# Entry point for OCR phase
# Parses arguments and coordinates OCR modules

import argparse
import sys
from pathlib import Path

from ocr.modules.digital_parser import DigitalParser
# from ocr.modules.ocr_engine import OCREngine
from ocr.modules.preprocessor import Preprocessor
from ocr.modules.exporter import json_to_docx, inject_break_tag


def main():
    parser = argparse.ArgumentParser(description="Document OCR & Conversion Pipeline")

    parser.add_argument("--mode", required=True, choices=["scan", "digital", "math"],
                        help="Type of processing: scan | digital | math")

    parser.add_argument("--input", required=True,
                        help="Input PDF file")

    parser.add_argument("--preprocess", default=False, action="store_true",
                        help="Enable preprocessing for scanned PDFs")

    args = parser.parse_args()

    pdf_path = Path(args.input)

    if not pdf_path.exists():
        print(f"Input file does not exist")
        sys.exit(1)

    # ========== MODE: DIGITAL DOCUMENT ==========
    if args.mode == "digital":
        print("Running in DIGITAL mode...")
        parser = DigitalParser()

        try:
            out_path = parser.convert(pdf_path)
            print(f"Converted digital PDF → DOCX")
        except Exception as e:
            print(f"Error: {e}")

        return

    # ========== MODE: SCANNED DOCUMENT ==========
    # if args.mode == "scan":
    #     print("Running in SCAN mode...")

    #     pre = Preprocessor()
    #     ocr = OCREngine()
    #     exporter = WordExporter()

    #     # Step 1: Preprocess
    #     if args.preprocess:
    #         print("Preprocessing images...")
    #         # (Bạn có thể tự nối với hàm crop/enhance tùy pipeline)
    #         pass

    #     # Step 2: OCR pipeline
    #     print("Running OCR...")
    #     text, images, tables = ocr.extract_layout(pdf_path)

    #     # Step 3: Export to Word
    #     print("Exporting to DOCX...")
    #     output = exporter.write_to_word({
    #         "text": text,
    #         "images": images,
    #         "tables": tables
    #     })

    #     print(f"Exported to: {output}")
    #     return

    # # ========== MODE: MATH PDF ==========
    # if args.mode == "math":
    #     print("Running in MATH mode...")

    #     ocr = OCREngine()
    #     latex = ocr.detect_formula(pdf_path)

    #     print("LaTeX Output:")
    #     print(latex)
    #     return


if __name__ == "__main__":
    main()

