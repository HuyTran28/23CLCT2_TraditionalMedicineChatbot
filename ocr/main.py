# Entry point for OCR phase
# Parses arguments and coordinates OCR modules

import argparse
import sys
from pathlib import Path


from modules.digital_parser import DigitalParser
from modules.preprocessor import Preprocessor
from modules.exporter import WordExporter
from modules.ocr_engine import OCREngine

def check_ocr_dependencies():
    """Check if all required libraries for enhanced OCR are available"""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        from craft_text_detector import Craft
    except ImportError:
        missing.append("craft-text-detector")
    
    try:
        from vietocr.tool.predictor import Predictor
    except ImportError:
        missing.append("vietocr")
    
    if missing:
        print("❌ Missing required libraries:")
        for lib in missing:
            print(f"   - {lib}")
        print("\nInstall missing libraries with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Document OCR & Conversion Pipeline")

    parser.add_argument("--mode", required=True, choices=["scan", "digital", "math"],
                        help="Type of processing: scan | digital | math")

    parser.add_argument("--input", required=True,
                        help="Input PDF or image file")
    
    parser.add_argument("--output", default=None,
                        help="Output file path (default: auto-generated)")

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
        digital_parser = DigitalParser()

        try:
            output_path = args.output if args.output else pdf_path.with_suffix(".docx")
            out_path = digital_parser.convert(pdf_path, output_path)
            print(f"✓ Converted digital PDF → DOCX: {out_path}")
        except Exception as e:
            print(f"✗ Error: {e}")
            sys.exit(1)

        return


    # ========== MODE: SCANNED DOCUMENT ========== 
    if args.mode == "scan":
        # Check dependencies before running OCR
        if not check_ocr_dependencies():
            sys.exit(1)
        
        print("🔍 Running in SCAN mode (Enhanced Vietnamese OCR)...")
        exporter = WordExporter()

        # Step 1: Preprocess (optional)
        if args.preprocess:
            print("Preprocessing images...")
            pre = Preprocessor()
            # Implement preprocessing if needed
            pass

        # Step 2: Enhanced OCR pipeline (CRAFT + VietOCR)
        print("Using Enhanced Vietnamese OCR pipeline (CRAFT + VietOCR)...")
        ocr = OCREngine()
        
        # Convert PDF to image if needed
        # For now, assume input is an image (PNG/JPG)
        # If PDF, you need to convert PDF pages to images first
        image_path = str(pdf_path)
        
        try:
            results = ocr.run(image_path, visualize=False)
            
            # Export results to JSON
            import json
            json_filename = pdf_path.stem + "_ocr_results.json"
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✓ Saved {len(results)} detections to '{json_filename}'")
            
            # Export to DOCX
            output_path = args.output if args.output else pdf_path.stem + "_output.docx"
            exporter.write_to_word(results, output_path)
            print(f"✓ Exported to DOCX: {output_path}")
        except Exception as e:
            print(f"✗ OCR Error: {e}")
            sys.exit(1)
        
        return

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

