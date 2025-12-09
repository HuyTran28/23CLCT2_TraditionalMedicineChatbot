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
from .ocr_engine import OCREngine
from .exporter import WordExporter
from .markdown_processor import MarkdownProcessor
 

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
        analyze_layout: bool = True,
        extract_tables: bool = True,
        use_llm_correction: bool = False
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
            extract_tables: Extract and process tables from scanned pages
        """
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.dpi = dpi
        self.enable_preprocessing = enable_preprocessing
        self.auto_detect = auto_detect
        self.extract_images = extract_images
        self.analyze_layout = analyze_layout
        self.extract_tables = extract_tables
        self.use_llm_correction = use_llm_correction
        # Maximum number of worker threads for page-level parallelism.
        # If None, defaults to number of CPUs.
        self.max_workers: Optional[int] = None
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize modules
        self.digital_parser = DigitalParser()
        self.pdf_converter = PDFConverter(dpi=dpi)
        # self.preprocessor = Preprocessor(output_dir=Path(temp_dir) / "preprocessed")
        self.ocr_engine = OCREngine(output_dir=Path(temp_dir) / "craft_output")
        self.exporter = WordExporter()
        self.markdown_processor = MarkdownProcessor(
            use_llm_correction=use_llm_correction
        )
        
    
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
        """Process scanned PDF using marker-pdf OCR pipeline"""
        
        try:
            # Step 1: Use marker-pdf to process the entire PDF
            logger.info("Step 1: Processing PDF with marker-pdf...")
            marker_result = self.ocr_engine.process_pdf(pdf_path)
            
            # Step 2: Extract any additional images if needed (marker-pdf handles most)
            logger.info(f"✓ Extracted {len(marker_result['images'])} images/formulas")
            
            # Step 3: Post-process markdown (insert breaks, fix spelling)
            logger.info("Step 3: Post-processing markdown...")
            processed_markdown = self.markdown_processor.process(
                marker_result['markdown'],
                images=marker_result['images']
            )
            
            # Step 4: Save processed markdown
            markdown_path = self.output_dir / f"{pdf_path.stem}_ocr_results.md"
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(processed_markdown)
            logger.info(f"✓ Saved processed markdown: {markdown_path}")
            
            # Step 5: Export to DOCX
            logger.info("Step 5: Converting markdown to Word document...")
            self.exporter.markdown_to_word(
                processed_markdown,
                str(output_path),
                images=marker_result['images']
            )
            logger.info(f"✓ OCR processing complete: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Marker-pdf processing failed: {e}")
            raise
    
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
    
    def _filter_overlapping_tables_and_images(
        self,
        tables: List[Dict[str, Any]],
        images: List[Dict[str, Any]],
        overlap_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Filter out tables that significantly overlap with detected images.
        Images are more reliably detected, so we prefer keeping images over tables in case of overlap.
        
        Args:
            tables: List of detected tables with bbox
            images: List of detected images with bbox
            overlap_threshold: IoU threshold to consider overlap (0.5 = 50% overlap)
        
        Returns:
            Filtered list of tables with overlapping ones removed
        """
        filtered_tables = []
        
        for table in tables:
            table_bbox = table.get('bbox')
            page_num = table.get('page_num')
            
            if not table_bbox:
                filtered_tables.append(table)
                continue
            
            # Check overlap with all images on the same page
            overlaps_with_image = False
            for image in images:
                if image.get('page_num') != page_num:
                    continue
                
                image_bbox = image.get('bbox')
                if not image_bbox:
                    continue
                
                # Calculate Intersection over Union (IoU)
                iou = self._calculate_iou(table_bbox, image_bbox)
                
                if iou > overlap_threshold:
                    logger.debug(f"Table {table.get('table_id', 'unknown')} overlaps with "
                               f"image {image.get('image_id', 'unknown')} (IoU: {iou:.2f})")
                    overlaps_with_image = True
                    break
            
            if not overlaps_with_image:
                filtered_tables.append(table)
        
        return filtered_tables
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: [x0, y0, x1, y1]
            bbox2: [x0, y0, x1, y1]
        
        Returns:
            IoU score (0-1)
        """
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0  # No overlap
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
