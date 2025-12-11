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
from .metrics import PipelineMetrics
 

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
        
        # Create images output folder
        self.images_output_dir = self.output_dir / "extracted_images"
        self.images_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize modules
        self.digital_parser = DigitalParser()
        self.pdf_converter = PDFConverter(dpi=dpi)
        # self.preprocessor = Preprocessor(output_dir=Path(temp_dir) / "preprocessed")
        self.ocr_engine = OCREngine(output_dir=Path(temp_dir) / "craft_output")
        self.exporter = WordExporter()
        self.markdown_processor = MarkdownProcessor(
            use_llm_correction=use_llm_correction
        )
        
        # Initialize metrics tracker
        self.metrics = PipelineMetrics(output_dir=self.output_dir)
        
    
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
        self.metrics.start_processing()
        
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
            # Track input file
            self.metrics.add_file_processed(pdf_path)
            
            # Determine output path
            if output_path is None:
                output_path = self.output_dir / f"{pdf_path.stem}.docx"
            output_path = Path(output_path)
            
            # Auto-detect mode if not specified
            if mode is None and self.auto_detect:
                is_digital = self.digital_parser.is_digital_pdf(pdf_path)
                mode = "digital" if is_digital else "scan"
            elif mode is None:
                mode = "scan"
            
            # Process based on mode
            if mode == "digital":
                result = self._process_digital(pdf_path, output_path)
            elif mode == "scan":
                result = self._process_scanned(pdf_path, output_path)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Update metrics with output file sizes
            self.metrics.set_output_files_size(docx_path=output_path)
            self.metrics.end_processing()
            
            return result
            
        except Exception as e:
            self.metrics.add_error(str(e))
            self.metrics.end_processing()
            raise
    
    def _process_digital(self, pdf_path: Path, output_path: Path) -> Path:
        """Process digital PDF using pdf2docx"""
        try:
            self.digital_parser.convert(pdf_path, output_path)
            return output_path
        except Exception as e:
            logger.error(f"Digital conversion failed: {e}")
            raise
    
    def _process_scanned(self, pdf_path: Path, output_path: Path) -> Path:
        """Process scanned PDF using marker-pdf OCR pipeline"""
        
        try:
            # Step 1: Use marker-pdf to process the entire PDF
            marker_result = self.ocr_engine.process_pdf(pdf_path, safe_mode=True)
            
            # Step 2: Extract and organize images
            organized_images = self._organize_extracted_images(
                marker_result['images'],
                pdf_path
            )
            self.metrics.add_images_extracted([img['output_path'] for img in organized_images])
            
            # Step 3: Post-process markdown with image IDs
            processed_markdown = self.markdown_processor.process(
                marker_result['markdown'],
                images=organized_images
            )
            
            # Step 4: Count lines in processed markdown
            line_count = len(processed_markdown.split('\n'))
            sample_count = self._count_samples(processed_markdown, organized_images)
            self.metrics.set_line_count(line_count)
            self.metrics.set_sample_count(sample_count)
            
            # Step 5: Save processed markdown
            markdown_path = self.output_dir / f"{pdf_path.stem}_ocr_results.md"
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(processed_markdown)
            self.metrics.set_output_files_size(markdown_path=markdown_path)
            
            # Step 6: Export to DOCX
            self.exporter.markdown_to_word(
                processed_markdown,
                str(output_path),
                images=organized_images
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Marker-pdf processing failed: {e}")
            self.metrics.add_error(f"Scanned processing: {str(e)}")
            raise
    
    def _organize_extracted_images(self, images: List[Dict[str, Any]], 
                                   pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Organize extracted images into output folder with distinct IDs
        
        Args:
            images: List of extracted image data
            pdf_path: Path to source PDF
        
        Returns:
            List of images with organized paths and IDs
        """
        organized_images = []
        
        for idx, image in enumerate(images, 1):
            try:
                original_path = Path(image.get('file_path', ''))
                
                if not original_path.exists():
                    logger.warning(f"Image file not found: {original_path}")
                    continue
                
                # Create unique image ID mapping to this image
                image_id = f"{pdf_path.stem}_img_{idx:03d}"
                
                # Copy image to organized output folder
                output_filename = f"{image_id}.png"
                output_path = self.images_output_dir / output_filename
                
                shutil.copy2(original_path, output_path)
                
                # Update image data with organized path and ID
                organized_image = image.copy()
                organized_image['output_path'] = str(output_path)
                organized_image['image_id'] = image_id
                organized_image['original_file_path'] = str(original_path)
                
                organized_images.append(organized_image)
                logger.debug(f"Organized image {idx}: {image_id} -> {output_filename}")
                
            except Exception as e:
                logger.warning(f"Failed to organize image {idx}: {e}")
                self.metrics.add_error(f"Image organization: {str(e)}")
                continue
        
        # Create index file mapping image IDs
        index_path = self.output_dir / "images_index.json"
        index_data = {
            'source_pdf': str(pdf_path),
            'output_folder': str(self.images_output_dir),
            'total_images': len(organized_images),
            'images': [
                {
                    'id': img['image_id'],
                    'filename': img['output_path'].split('/')[-1] if '/' in img['output_path'] else img['output_path'].split('\\')[-1],
                    'path': img['output_path']
                }
                for img in organized_images
            ]
        }
        
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        return organized_images
    
    def _count_samples(self, markdown_text: str, images: List[Dict[str, Any]]) -> int:
        """
        Count OCR samples/elements (text blocks, images, tables)
        
        Args:
            markdown_text: Processed markdown text
            images: List of extracted images
        
        Returns:
            Total count of samples
        """
        # Count headings
        import re
        headings = len(re.findall(r'^#+\s', markdown_text, re.MULTILINE))
        
        # Count paragraphs
        lines = [l.strip() for l in markdown_text.split('\n') if l.strip()]
        
        # Count tables
        tables = markdown_text.count('\n|')
        
        # Total samples
        total_samples = headings + len(lines) + tables + len(images)
        
        return total_samples
    
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
        
        results = []
        
        for _, pdf_path in enumerate(pdf_files, start=1):     
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
        return results