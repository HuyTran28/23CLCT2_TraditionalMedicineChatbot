# Pipeline orchestrator
# Coordinates the entire OCR workflow

from __future__ import annotations
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import os
import concurrent.futures

from .digital_parser import DigitalParser
from .pdf_converter import PDFConverter
from .preprocessor import Preprocessor
from .ocr_engine import OCREngine
from .exporter import WordExporter
from .image_extractor import ImageExtractor
from .layout_analyzer import LayoutAnalyzer

logger = logging.getLogger(__name__)


class OCRPipeline:
    """
    Complete OCR pipeline for processing PDF documents
    
    Automatically detects if PDF is digital or scanned, and applies
    appropriate conversion method.
    """
    
    def __init__(
        self,
        output_dir: str | Path = "./output",
        temp_dir: str | Path = "./temp",
        dpi: int = 300,
        enable_preprocessing: bool = True,
        auto_detect: bool = True,
        extract_images: bool = True,
        analyze_layout: bool = True
    ):
        """
        Initialize OCR Pipeline
        
        Args:
            output_dir: Directory for final outputs
            temp_dir: Directory for temporary files
            dpi: Resolution for PDF to image conversion
            enable_preprocessing: Enable image preprocessing for scanned docs
            auto_detect: Automatically detect if PDF is digital or scanned
            extract_images: Extract and embed images from PDF
            analyze_layout: Analyze and preserve document layout/structure
        """
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.dpi = dpi
        self.enable_preprocessing = enable_preprocessing
        self.auto_detect = auto_detect
        self.extract_images = extract_images
        self.analyze_layout = analyze_layout
        # Maximum number of worker threads for page-level parallelism.
        # If None, defaults to number of CPUs.
        self.max_workers: Optional[int] = None
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize modules
        self.digital_parser = DigitalParser()
        self.pdf_converter = PDFConverter(dpi=dpi)
        self.preprocessor = Preprocessor(output_dir=Path(temp_dir) / "preprocessed")
        self.ocr_engine = OCREngine(output_dir=Path(temp_dir) / "craft_output")
        self.exporter = WordExporter()
        self.image_extractor = ImageExtractor(output_dir=Path(temp_dir) / "extracted_images")
        self.layout_analyzer = LayoutAnalyzer()
    
    def process_pdf(
        self,
        pdf_path: str | Path,
        output_path: Optional[str | Path] = None,
        mode: Optional[str] = None
    ) -> Path:
        """
        Process a single PDF file
        
        Args:
            pdf_path: Path to input PDF
            output_path: Path for output DOCX (auto-generated if None)
            mode: Processing mode ('digital', 'scan', or None for auto-detect)
        
        Returns:
            Path to output DOCX file
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Determine output path
        if output_path is None:
            output_path = self.output_dir / f"{pdf_path.stem}.docx"
        output_path = Path(output_path)
        
        # Auto-detect mode if not specified
        if mode is None and self.auto_detect:
            is_digital = self.digital_parser.is_digital_pdf(pdf_path)
            mode = "digital" if is_digital else "scan"
            logger.info(f"Auto-detected mode: {mode}")
        elif mode is None:
            mode = "scan"  # Default to scan if auto-detect disabled
        
        # Process based on mode
        if mode == "digital":
            logger.info(f"Processing as digital PDF: {pdf_path.name}")
            return self._process_digital(pdf_path, output_path)
        elif mode == "scan":
            logger.info(f"Processing as scanned PDF: {pdf_path.name}")
            return self._process_scanned(pdf_path, output_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _process_digital(self, pdf_path: Path, output_path: Path) -> Path:
        """Process digital PDF using pdf2docx"""
        try:
            self.digital_parser.convert(pdf_path, output_path)
            logger.info(f"✓ Digital conversion complete: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Digital conversion failed: {e}")
            raise
    
    def _process_scanned(self, pdf_path: Path, output_path: Path) -> Path:
        """Process scanned PDF using OCR pipeline with CV-based figure detection and layout analysis"""
        temp_images_dir = self.temp_dir / f"{pdf_path.stem}_images"
        temp_images_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Convert PDF to images
            # For scanned PDFs, skip embedded image extraction since they're just page scans
            logger.info("Step 1: Converting PDF pages to images...")
            image_paths = self.pdf_converter.pdf_to_images(
                pdf_path,
                output_dir=temp_images_dir
            )
            
            if not image_paths:
                raise RuntimeError("No images generated from PDF")
            
            # Step 2: Extract figures from scanned pages using computer vision
            extracted_images = []
            if self.extract_images:
                logger.info("Step 2: Detecting figures in scanned pages using CV methods...")
                try:
                    for page_idx, image_path in enumerate(image_paths, start=1):
                        page_figures = self.image_extractor.extract_figures_from_page_image(
                            image_path,
                            page_num=page_idx,
                            prefix=f"{pdf_path.stem}_fig",
                            use_multiple_strategies=True  # Use all CV strategies
                        )
                        extracted_images.extend(page_figures)
                    
                    if len(extracted_images) > 0:
                        logger.info(f"✓ Detected {len(extracted_images)} figures from scanned pages")
                    else:
                        logger.info("No figures detected in scanned pages")
                except Exception as e:
                    logger.warning(f"Figure detection failed: {e}")
            
            # Get page dimensions for layout analysis
            import cv2
            first_page = cv2.imread(str(image_paths[0]))
            page_height, page_width = first_page.shape[:2] if first_page is not None else (3000, 2000)
            
            # Step 3: Process each page with OCR (parallelized)
            num_pages = len(image_paths)
            logger.info(f"Step 3: Running OCR on {num_pages} pages...")

            # Determine worker count - reduce workers to avoid CPU contention
            # OCR is CPU-intensive, so limit to 2-4 workers even on high-core systems
            max_optimal_workers = min(4, os.cpu_count() or 1)
            workers = self.max_workers if self.max_workers is not None else max_optimal_workers
            logger.info(f"Using {workers} worker threads for page processing")

            def _process_page(task: tuple[int, Path]) -> Dict[str, Any]:
                page_idx, image_path = task
                
                try:
                    # Optional preprocessing
                    if self.enable_preprocessing:
                        import cv2
                        img = cv2.imread(str(image_path))
                        img = self.preprocessor.deskew_page(img)
                        img = self.preprocessor.enhance_contrast(img)
                        cv2.imwrite(str(image_path), img)

                    # Run OCR
                    results = self.ocr_engine.run(str(image_path), visualize=False)

                    # Add page number to each result
                    for r in results:
                        r["page_id"] = page_idx

                    return {"page_num": page_idx, "results": results, "image_path": str(image_path)}
                except Exception as e:
                    logger.error(f"[ERROR] Page {page_idx} processing failed: {e}")
                    return {"page_num": page_idx, "results": [], "error": str(e)}

            tasks = [(i + 1, p) for i, p in enumerate(image_paths)]
            all_pages_results: List[Dict[str, Any]] = []

            # Use ThreadPoolExecutor for I/O and external-API-bound work
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                future_to_page = {ex.submit(_process_page, t): t[0] for t in tasks}

                for fut in concurrent.futures.as_completed(future_to_page):
                    page_num = future_to_page[fut]
                    try:
                        page_result = fut.result()
                        all_pages_results.append(page_result)
                        logger.info(f"✓ Completed page {page_num}/{num_pages}")
                    except Exception as e:
                        logger.error(f"Page {page_num} processing failed: {e}")
                        # Include an empty page result to preserve ordering
                        all_pages_results.append({"page_num": page_num, "results": [], "error": str(e)})

            # Ensure results are ordered by page number
            all_pages_results.sort(key=lambda x: x["page_num"])
            
            # Step 4: Layout analysis and classification (if enabled)
            if self.analyze_layout:
                logger.info("Analyzing document layout and structure...")
                for page_result in all_pages_results:
                    results = page_result.get("results", [])
                    if results:
                        # Classify text elements
                        classified = self.layout_analyzer.classify_text_elements(
                            results,
                            page_height=page_height,
                            page_width=page_width
                        )
                        
                        # Merge with images for this page
                        page_num = page_result.get("page_num", 1)
                        page_images = [img for img in extracted_images if img.get("page_num") == page_num]
                        
                        if page_images:
                            classified = self.layout_analyzer.merge_with_images(classified, page_images)
                        
                        page_result["results"] = classified
                logger.info("✓ Layout analysis complete")
            
            # Step 5: Save intermediate JSON
            json_path = self.output_dir / f"{pdf_path.stem}_ocr_results.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "pages": all_pages_results,
                    "images": extracted_images
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ Saved OCR results: {json_path}")
            
            # Step 6: Export to DOCX
            logger.info("Exporting to Word document...")
            self.exporter.write_to_word(
                {"pages": all_pages_results}, 
                str(output_path),
                images=extracted_images
            )
            logger.info(f"✓ OCR processing complete: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise
        finally:
            # Cleanup temporary images
            if temp_images_dir.exists():
                shutil.rmtree(temp_images_dir, ignore_errors=True)
    
    def process_batch(
        self,
        input_dir: str | Path,
        pattern: str = "*.pdf",
        mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory
        
        Args:
            input_dir: Directory containing PDF files
            pattern: File pattern to match (default: "*.pdf")
            mode: Processing mode for all files (None for auto-detect)
        
        Returns:
            List of processing results with status for each file
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        pdf_files = sorted(input_dir.glob(pattern))
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        results = []
        
        for idx, pdf_path in enumerate(pdf_files, start=1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing file {idx}/{len(pdf_files)}: {pdf_path.name}")
            logger.info(f"{'='*60}")
            
            try:
                output_path = self.process_pdf(pdf_path, mode=mode)
                results.append({
                    "input": str(pdf_path),
                    "output": str(output_path),
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                results.append({
                    "input": str(pdf_path),
                    "output": None,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Summary
        success_count = sum(1 for r in results if r["status"] == "success")
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch processing complete: {success_count}/{len(results)} successful")
        logger.info(f"{'='*60}")
        
        return results
