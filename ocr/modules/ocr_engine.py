import os
from PIL import Image
from pathlib import Path
import logging
import shutil

logger = logging.getLogger(__name__)


class OCREngine:
    """
    Modern OCR Engine using marker-pdf for Vietnamese scanned documents.
    
    marker-pdf is a modern library that:
    - Handles Vietnamese text well
    - Extracts formulas as images
    - Preserves table structure
    - Maintains reading order
    - Outputs clean markdown
    """
    def __init__(self, output_dir='./marker_output', extract_images_dir='./temp/extracted_images'):
        """
        Initialize the Marker OCR Engine.
        
        Args:
            output_dir: Directory to save markdown outputs
            extract_images_dir: Directory to save extracted images and formulas
        """
        self.output_dir = Path(output_dir)
        self.extract_images_dir = Path(extract_images_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extract_images_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
            
            # Initialize marker models
            logger.info("Initializing marker-pdf models...")
            self.model_dict = create_model_dict()
            self.converter = PdfConverter(
                artifact_dict=self.model_dict,
            )
            logger.info("✓ Marker-pdf engine initialized successfully")
            
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import marker-pdf. Install it with: pip install marker-pdf\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize marker-pdf engine: {e}")

    def process_pdf(self, pdf_path, page_range=None):
        """
        Process a PDF file and extract text, images, and formulas.
        
        Args:
            pdf_path: Path to the PDF file
            page_range: Optional tuple (start, end) for page range
            
        Returns:
            dict: Processing results with markdown text and image references
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Processing PDF with marker-pdf: {pdf_path.name}")
        
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            
            # Convert PDF to markdown
            rendered = self.converter(str(pdf_path))
            
            # Extract markdown text
            markdown_text = rendered.markdown
            
            # Process images and formulas
            images_info = []
            if hasattr(rendered, 'images') and rendered.images:
                for idx, (image_key, image_data) in enumerate(rendered.images.items(), 1):
                    # Save image to disk
                    image_filename = f"{pdf_path.stem}_img_{idx}.png"
                    image_path = self.extract_images_dir / image_filename
                    
                    # Save the image
                    if hasattr(image_data, 'save'):
                        image_data.save(image_path)
                    elif isinstance(image_data, bytes):
                        with open(image_path, 'wb') as f:
                            f.write(image_data)
                    else:
                        # Try to convert to PIL Image and save
                        try:
                            # Try to save image_data as PIL Image directly
                            try:
                                image_data.save(image_path)
                            except Exception:
                                logger.warning(f"Could not save image {idx} as PIL Image")
                                continue
                        except:
                            logger.warning(f"Could not save image {idx}")
                            continue
                    
                    images_info.append({
                        'image_id': f'img_{idx}',
                        'file_path': str(image_path),
                        'original_key': image_key,
                        'type': 'image'
                    })
                    
                    # Replace image reference in markdown with placeholder
                    if image_key in markdown_text:
                        placeholder = f"[IMAGE_PLACEHOLDER_{idx}]"
                        markdown_text = markdown_text.replace(image_key, placeholder)
            
            # Save markdown output
            markdown_path = self.output_dir / f"{pdf_path.stem}.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            
            logger.info(f"✓ Extracted {len(images_info)} images/formulas")
            logger.info(f"✓ Saved markdown to: {markdown_path}")
            
            return {
                'markdown': markdown_text,
                'markdown_path': str(markdown_path),
                'images': images_info,
                'metadata': {
                    'num_images': len(images_info),
                    'source_pdf': str(pdf_path)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to process PDF with marker-pdf: {e}")
            raise

    def run(self, pdf_path, visualize=False):
        """
        Legacy interface for backward compatibility.
        Processes a PDF and returns results in OCR format.
        
        Args:
            pdf_path: Path to PDF or image file
            visualize: Ignored for marker-pdf
            
        Returns:
            list: OCR results (for compatibility, returns markdown segments)
        """
        # Check if input is a PDF or image
        pdf_path = Path(pdf_path)
        
        if pdf_path.suffix.lower() == '.pdf':
            # Process as PDF
            result = self.process_pdf(pdf_path)
            
            # Convert markdown to line-by-line results for compatibility
            lines = result['markdown'].split('\n')
            ocr_results = []
            
            for idx, line in enumerate(lines, 1):
                if line.strip():
                    ocr_results.append({
                        'id': idx,
                        'text': line,
                        'confidence': 1.0,
                        'type': 'text'
                    })
            
            # Add image placeholders
            for img_info in result['images']:
                ocr_results.append({
                    'id': img_info['image_id'],
                    'text': f"[IMAGE: {img_info['file_path']}]",
                    'confidence': 1.0,
                    'type': 'image',
                    'file_path': img_info['file_path']
                })
            
            return ocr_results
        else:
            # For single images, return empty (marker-pdf works on PDFs)
            logger.warning("Marker-pdf works best with PDF files. For single images, consider using the PDF workflow.")
            return []

            return []

    @staticmethod
    def visualize_results(img_rgb, results):
        """
        Visualize OCR results on an image (for debugging).
        Note: marker-pdf doesn't use bounding boxes, so this is simplified.
        """
        logger.info(f"Visualization not fully supported for marker-pdf engine")
        logger.info(f"Extracted {len(results)} text segments")