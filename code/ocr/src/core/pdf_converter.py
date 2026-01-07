# PDF Converter module
# Converts PDF pages to images for OCR processing

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFConverter:
    """Convert PDF pages to images for OCR processing"""
    
    def __init__(self, dpi: int = 300):
        """
        Initialize PDF converter
        
        Args:
            dpi: Resolution for image conversion (default: 300)
        """
        self.dpi = dpi
        self.zoom = dpi / 72  # PDF standard is 72 DPI
    
    def pdf_to_images(
        self,
        pdf_path: str | Path,
        output_dir: Optional[str | Path] = None,
        start_page: int = 0,
        end_page: Optional[int] = None
    ) -> List[Path]:
        """
        Convert PDF pages to PNG images
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images (default: same as PDF)
            start_page: Starting page number (0-indexed)
            end_page: Ending page number (exclusive, None for all pages)
            
        Returns:
            List of paths to generated images
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Set output directory
        if output_dir is None:
            output_dir = pdf_path.parent / "temp_images"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        
        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            
            # Determine page range
            end = end_page if end_page is not None else total_pages
            end = min(end, total_pages)
            
            logger.info(f"Converting PDF pages {start_page} to {end-1} from {pdf_path.name}")
            
            for page_num in range(start_page, end):
                page = doc.load_page(page_num)
                
                # Create transformation matrix for higher resolution
                mat = fitz.Matrix(self.zoom, self.zoom)
                
                # Render page to image
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Save image
                image_filename = f"{pdf_path.stem}_page_{page_num + 1:03d}.png"
                image_path = output_dir / image_filename
                pix.save(str(image_path))
                
                image_paths.append(image_path)
                logger.info(f"  Page {page_num + 1}/{total_pages} → {image_filename}")
            
            doc.close()
            logger.info(f"Converted {len(image_paths)} pages to images")
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {e}")
        
        return image_paths
    
    def get_page_count(self, pdf_path: str | Path) -> int:
        """Get total number of pages in PDF"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF: {e}")
