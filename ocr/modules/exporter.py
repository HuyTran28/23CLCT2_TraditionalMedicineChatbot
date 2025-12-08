# Contains python-docx & tagging logic
import json
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class WordExporter:
    def __init__(self, base_spacing=1.0, font_size=12, max_image_width=6.0):
        self.base_spacing = base_spacing
        self.font_size = font_size
        self.max_image_width = max_image_width  # Maximum width in inches for embedded images

    @staticmethod
    def center_y(box):
        return sum(p[1] for p in box) / len(box)
    
    def add_image_to_document(self, doc, image_data):
        """
        Add an image to the Word document with proper formatting
        
        Args:
            doc: Document object
            image_data: Dictionary with 'file_path', 'image_id', 'width', 'height'
        
        Returns:
            bool: True if image was added successfully, False otherwise
        """
        try:
            image_path = Path(image_data.get('file_path', ''))
            
            # Skip if file doesn't exist (filtered out during extraction)
            if not image_path.exists():
                logger.debug(f"Skipping image {image_data.get('image_id', 'unknown')}: file not found (likely filtered)")
                return False
            
            # Add image with caption
            image_id = image_data.get('image_id', 'unknown')
            
            # Add paragraph for image
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Calculate image dimensions (maintain aspect ratio)
            original_width = image_data.get('width', 800)
            original_height = image_data.get('height', 600)

            # Validate dimensions and set defaults if invalid
            if original_width <= 0:
                logger.warning(f"Invalid width for image {image_id}, using default")
                original_width = 800
            if original_height <= 0:
                logger.warning(f"Invalid height for image {image_id}, using default")
                original_height = 600

            aspect_ratio = original_height / original_width
            width_inches = min(self.max_image_width, original_width / 100)  # Rough conversion
            height_inches = width_inches * aspect_ratio

            # Add the image
            run = p.add_run()
            run.add_picture(str(image_path), width=Inches(width_inches))
            
            # Add caption below image
            caption_p = doc.add_paragraph()
            caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption_run = caption_p.add_run(f"[{image_id}]")
            caption_run.font.size = Pt(10)
            caption_run.font.italic = True
            caption_run.font.color.rgb = RGBColor(128, 128, 128)
            
            logger.debug(f"Added image {image_id} to document")
            return True
            
        except FileNotFoundError:
            logger.debug(f"Image file not found: {image_data.get('image_id', 'unknown')}")
            return False
        except Exception as e:
            logger.warning(f"Failed to add image {image_data.get('image_id', 'unknown')}: {e}")
            return False

    @staticmethod
    def inject_break_tag(text: str) -> str:
        """Insert </break> tag after level 2 headings (e.g., 2.1, 2.2, 3.1, etc.)"""
        # Pattern matches level 2 headings: X.Y where X and Y are digits
        pattern = r"^\s*\d+\.\d+\s"
        if re.match(pattern, text.strip()):
            # Check if it's exactly level 2 (one dot, not 2.1.1)
            heading_part = text.strip().split()[0] if text.strip().split() else ""
            if heading_part.count('.') == 1:
                return text.rstrip() + " </break>"
        return text

    def write_to_word(self, data, output_path="output.docx", images=None):
        """
        Export OCR results to Word document
        
        Args:
            data: Can be:
                - str: path to JSON file
                - list: OCR results (assumes single page)
                - dict with 'results' key: OCR results
                - dict with 'pages' key: Multi-page OCR results
            output_path: str, path to save docx
            images: Optional list of image dictionaries to embed
        
        Returns:
            Path to saved document
        """
        if isinstance(data, str):
            # Assume it's a path to JSON file
            with open(data, "r", encoding="utf-8") as f:
                data = json.load(f)

        if isinstance(data, dict) and "results" in data:
            items = data["results"]
        elif isinstance(data, dict) and "pages" in data:
            # Multi-page format: {"pages": [{page_num: 1, results: [...]}, ...]}
            return self._write_multipage_to_word(data["pages"], output_path, images)
        elif isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # For layout export: expects keys 'text', 'images', 'tables'
            doc = Document()
            if data.get("text"):
                doc.add_paragraph(data["text"])
            # Images and tables can be handled here if needed
            doc.save(output_path)
            print("Saved:", output_path)
            return output_path
        else:
            raise ValueError("Unsupported data format for export")

        # OCR results export - handle both with and without page_id
        pages = {}
        for it in items:
            page_id = it.get("page_id", 1)
            pages.setdefault(page_id, []).append(it)

        doc = Document()

        for page_index, (page_id, page_items) in enumerate(pages.items()):
            # Separate text items and image items
            text_items = [it for it in page_items if it.get("element_type") != "image"]
            image_items = [it for it in page_items if it.get("element_type") == "image"]
            
            # Sort text items by position
            text_items = sorted(
                text_items,
                key=lambda t: (self.center_y(t.get("box", [[0,0]])), t.get("box", [[0,0]])[0][0])
            )

            last_y = None
            last_height = 12

            for it in text_items:
                # Skip headers and footers if marked
                if it.get("skip", False):
                    continue
                
                text = it.get("text", "")
                box = it.get("box", [[0, 0], [0, 0], [0, 0], [0, 0]])
                element_type = it.get("element_type", "paragraph")

                # Apply break tag for level 2 headings
                text = self.inject_break_tag(text)

                y = self.center_y(box)
                height = abs(box[2][1] - box[0][1])

                if last_y is None:
                    spacing = self.base_spacing
                else:
                    gap = y - last_y
                    spacing = max(self.base_spacing, gap / last_height)

                p = doc.add_paragraph()
                run = p.add_run(text)
                run.font.size = Pt(self.font_size)
                
                # Apply heading formatting
                if element_type == "heading":
                    run.font.bold = True
                    heading_level = it.get("heading_level", 0)
                    if heading_level == 1:
                        run.font.size = Pt(16)
                    elif heading_level == 2:
                        run.font.size = Pt(14)
                    elif heading_level >= 3:
                        run.font.size = Pt(13)

                p.paragraph_format.line_spacing = spacing

                last_y = y
                last_height = max(height, 8)
            
            # Add images for this page
            if image_items:
                added_count = 0
                for img_item in image_items:
                    if self.add_image_to_document(doc, img_item):
                        added_count += 1
                if added_count > 0:
                    logger.info(f"Added {added_count}/{len(image_items)} images for page {page_id}")

            if page_index < len(pages) - 1:
                doc.add_page_break()

        doc.save(output_path)
        print("Saved:", output_path)
        return output_path
    
    def _write_multipage_to_word(self, pages_data, output_path="output.docx", images=None):
        """
        Export multi-page OCR results to Word document
        
        Args:
            pages_data: List of dicts with 'page_num' and 'results' keys
            output_path: str, path to save docx
            images: Optional list of image dictionaries to embed
        
        Returns:
            Path to saved document
        """
        doc = Document()
        
        # Sort pages by page number
        pages_data = sorted(pages_data, key=lambda p: p.get("page_num", 1))
        
        # Group images by page if provided
        images_by_page = {}
        if images:
            for img in images:
                page_num = img.get("page_num", 1)
                images_by_page.setdefault(page_num, []).append(img)
        
        for page_idx, page_data in enumerate(pages_data):
            page_num = page_data.get("page_num", page_idx + 1)
            results = page_data.get("results", [])
            
            if not results:
                continue
            
            # Separate text and image elements
            text_results = [r for r in results if r.get("element_type") != "image"]
            image_results = [r for r in results if r.get("element_type") == "image"]
            
            # Sort text results by vertical position, then horizontal
            text_results = sorted(
                text_results,
                key=lambda t: (self.center_y(t.get("box", [[0,0]])), t.get("box", [[0,0]])[0][0])
            )
            
            last_y = None
            last_height = 12
            
            for it in text_results:
                # Skip headers and footers
                if it.get("skip", False):
                    continue
                
                text = it.get("text", "")
                box = it.get("box", [[0, 0], [0, 0], [0, 0], [0, 0]])
                element_type = it.get("element_type", "paragraph")
                
                # Apply break tag
                text = self.inject_break_tag(text)
                
                y = self.center_y(box)
                height = abs(box[2][1] - box[0][1])
                
                if last_y is None:
                    spacing = self.base_spacing
                else:
                    gap = y - last_y
                    spacing = max(self.base_spacing, gap / last_height)
                
                p = doc.add_paragraph()
                run = p.add_run(text)
                run.font.size = Pt(self.font_size)
                
                # Apply heading formatting
                if element_type == "heading":
                    run.font.bold = True
                    heading_level = it.get("heading_level", 0)
                    if heading_level == 1:
                        run.font.size = Pt(16)
                    elif heading_level == 2:
                        run.font.size = Pt(14)
                    elif heading_level >= 3:
                        run.font.size = Pt(13)
                
                p.paragraph_format.line_spacing = spacing
                
                last_y = y
                last_height = max(height, 8)
            
            # Add images from results
            added_from_results = 0
            for img_item in image_results:
                if self.add_image_to_document(doc, img_item):
                    added_from_results += 1
            
            # Add images from separate image list
            added_from_list = 0
            if page_num in images_by_page:
                for img in images_by_page[page_num]:
                    if self.add_image_to_document(doc, img):
                        added_from_list += 1
            
            # Log summary
            total_added = added_from_results + added_from_list
            total_attempted = len(image_results) + len(images_by_page.get(page_num, []))
            if total_attempted > 0 and total_added > 0:
                logger.info(f"Page {page_num}: Added {total_added}/{total_attempted} images")
            
            # Add page break between pages (except last page)
            if page_idx < len(pages_data) - 1:
                doc.add_page_break()
        
        doc.save(output_path)
        print(f"Saved {len(pages_data)} pages to: {output_path}")
        return output_path

if __name__ == "__main__":
    # Example usage
    exporter = WordExporter()
    exporter.write_to_word("vietnamese_ocr_results.json", "output.docx")