# Entry point for OCR phase
# Parses arguments and coordinates OCR modules

import argparse
import sys
import logging
from pathlib import Path

from modules.pipeline import OCRPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_ocr_dependencies():
    """Check if all required libraries for marker-pdf OCR are available"""
    missing = []
    try:
        from marker.converters.pdf import PdfConverter
    except ImportError:
        missing.append("marker-pdf")
    if missing:
        print("❌ Missing required libraries:")
        for lib in missing:
            print(f"   - {lib}")
        print("\nInstall missing libraries with:")
        print("   pip install -r requirements.txt")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Document OCR & Conversion Pipeline - Convert PDFs to Word documents"
    )

    parser.add_argument(
        "--input",
        required=False,
        default=None,
        help="Input PDF file or directory containing PDFs (defaults to config.INPUT_DIR if present)"
    )
    
    parser.add_argument(
        "--output",
        default="./output",
        help="Output directory for converted files (default: ./output)"
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "scan", "digital"],
        default="auto",
        help="Processing mode: auto (detect) | scan (OCR) | digital (direct conversion)"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all PDFs in input directory"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads for page-level parallelism (default: CPU count)"
    )

    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable preprocessing for scanned PDFs"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF to image conversion (default: 300)"
    )
    
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Disable image extraction and embedding"
    )
    
    parser.add_argument(
        "--no-layout",
        action="store_true",
        help="Disable layout analysis (headers/footers/structure preservation)"
    )
    
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Disable table extraction and processing"
    )
    
    parser.add_argument(
        "--easydataset",
        action="store_true",
        help="Generate EasyDataset format output for post-processing"
    )
    
    parser.add_argument(
        "--llm-correction",
        action="store_true",
        help="Enable FREE spelling correction for OCR output (no API costs!)"
    )
    
    

    args = parser.parse_args()

    # CLI arguments always take precedence over config values.
    try:
        import config as cfg
        cfg_loaded = True
        print("Loaded configuration from 'config.py'")
    except Exception:
        cfg = None
        cfg_loaded = False

    # Check dependencies before running
    if not check_ocr_dependencies():
        sys.exit(1)

    # Resolve input path: CLI arg > config.INPUT_DIR > error
    input_arg = args.input
    if input_arg is None and cfg_loaded and hasattr(cfg, "INPUT_DIR"):
        input_arg = cfg.INPUT_DIR

    if input_arg is None:
        print("❌ No input path provided. Pass --input or set INPUT_DIR in config.py")
        sys.exit(1)

    input_path = Path(input_arg)
    if not input_path.exists():
        print(f"❌ Input path does not exist: {input_path}")
        sys.exit(1)


    # Apply config defaults where CLI uses defaults
    if cfg_loaded:
        # Override output if user did not supply a custom value
        if args.output == "./output" and hasattr(cfg, "OUTPUT_DIR"):
            args.output = cfg.OUTPUT_DIR

        # Override dpi if user did not supply a custom value
        if args.dpi == 300 and hasattr(cfg, "DPI"):
            args.dpi = cfg.DPI

    # Determine enable_preprocessing
    if args.no_preprocess:
        enable_preprocessing = False
    else:
        if cfg_loaded and hasattr(cfg, "ENABLE_PREPROCESSING"):
            enable_preprocessing = bool(cfg.ENABLE_PREPROCESSING)
        else:
            enable_preprocessing = True

    # Determine auto-detect behavior for pipeline
    if args.mode == "auto":
        auto_detect = cfg.AUTO_DETECT if (cfg_loaded and hasattr(cfg, "AUTO_DETECT")) else True
        mode_arg = None
    else:
        auto_detect = False
        mode_arg = args.mode

    # Initialize pipeline
    pipeline = OCRPipeline(
        output_dir=args.output,
        temp_dir="./temp",
        dpi=args.dpi,
        enable_preprocessing=enable_preprocessing,
        auto_detect=auto_detect,
        extract_images=not args.no_images,
        analyze_layout=not args.no_layout,
        extract_tables=not args.no_tables,
        use_llm_correction=args.llm_correction
    )

    mode = None if args.mode == "auto" else args.mode

    try:
        # Disallow batch processing / directories — process one PDF at a time only
        if args.batch or input_path.is_dir():
            print("❌ Batch processing is disabled. Provide a single PDF file as --input.")
            print("   To process multiple files, run the script separately for each PDF.")
            sys.exit(1)

        # Initialize pipeline with worker setting
        pipeline.max_workers = args.workers

        # Single file processing
        print(f"\n📄 Processing: {input_path.name}")
        output_path = pipeline.process_pdf(input_path, mode=mode)
        print(f"\n✓ Success! Output saved to: {output_path}")
        
        # Generate EasyDataset format if requested
        if args.easydataset:
            print("\n📊 Generating EasyDataset format...")
            from modules.easydataset_processor import EasyDatasetProcessor
            
            processor = EasyDatasetProcessor(chunk_size=512, overlap=50)
            
            # Find the OCR results JSON
            json_path = Path(args.output) / f"{input_path.stem}_ocr_results.json"
            
            if json_path.exists():
                # Process to EasyDataset format
                easydataset_path = Path(args.output) / f"{input_path.stem}_easydataset.json"
                dataset = processor.process_ocr_results(json_path, easydataset_path)
                
                # Export for Q&A generation
                qa_path = Path(args.output) / f"{input_path.stem}_qa.json"
                processor.export_for_qa_generation(dataset, qa_path)
                
                # Export for retrieval
                retrieval_path = Path(args.output) / f"{input_path.stem}_retrieval.json"
                processor.export_for_retrieval(dataset, retrieval_path)
                
                print(f"  ✓ EasyDataset format: {easydataset_path}")
                print(f"  ✓ Q&A format: {qa_path}")
                print(f"  ✓ Retrieval format: {retrieval_path}")
            else:
                print(f"  ⚠ OCR results JSON not found: {json_path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

