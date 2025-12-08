# Layout Analyzer Module
# Analyzes document layout to preserve structure (headings, paragraphs, lists)

import re
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class LayoutAnalyzer:
    """
    Analyze document layout to identify structural elements:
    - Headers and footers
    - Headings (different levels)
    - Paragraphs
    - Lists
    - Tables
    - Image captions
    """
    
    def __init__(
        self,
        page_margin_top: float = 0.1,
        page_margin_bottom: float = 0.1,
        heading_pattern: str = r'^\s*\d+(\.\d+)*\.?\s+',
        list_pattern: str = r'^\s*[-•·∙◦▪▫○●]\s+|^\s*\d+\.\s+'
    ):
        """
        Initialize Layout Analyzer
        
        Args:
            page_margin_top: Top margin ratio to detect headers (0-1)
            page_margin_bottom: Bottom margin ratio to detect footers (0-1)
            heading_pattern: Regex pattern to detect numbered headings
            list_pattern: Regex pattern to detect list items
        """
        self.page_margin_top = page_margin_top
        self.page_margin_bottom = page_margin_bottom
        self.heading_pattern = re.compile(heading_pattern)
        self.list_pattern = re.compile(list_pattern)
    
    def classify_text_elements(
        self,
        ocr_results: List[Dict[str, Any]],
        page_height: int,
        page_width: int
    ) -> List[Dict[str, Any]]:
        """
        Classify OCR text elements by type (header, footer, heading, paragraph, etc.)
        
        Args:
            ocr_results: List of OCR results with 'text', 'box', 'bbox'
            page_height: Height of the page in pixels
            page_width: Width of the page in pixels
            
        Returns:
            OCR results with added 'element_type' and 'heading_level' fields
        """
        classified = []
        
        header_threshold = page_height * self.page_margin_top
        footer_threshold = page_height * (1 - self.page_margin_bottom)
        
        for item in ocr_results:
            text = item.get("text", "").strip()
            box = item.get("box", [])
            
            if not text or not box:
                continue
            
            # Calculate vertical position
            y_center = sum(p[1] for p in box) / len(box)
            
            # Determine element type
            element_type = "paragraph"
            heading_level = 0
            skip = False
            
            # Check for header/footer
            if y_center < header_threshold:
                element_type = "header"
                skip = True  # Headers are typically excluded
            elif y_center > footer_threshold:
                element_type = "footer"
                skip = True  # Footers are typically excluded
            else:
                # Check for heading
                heading_match = self.heading_pattern.match(text)
                if heading_match:
                    element_type = "heading"
                    # Determine heading level by counting dots
                    heading_number = heading_match.group(0).strip()
                    heading_level = heading_number.count('.') + 1
                    if not heading_number.endswith('.'):
                        heading_level = heading_number.count('.') + 1
                
                # Check for list item
                elif self.list_pattern.match(text):
                    element_type = "list_item"
                
                # Check for potential caption (contains "Hình", "Bảng", "Figure", "Table")
                elif re.search(r'\b(Hình|Bảng|Figure|Table|Chart|Biểu đồ)\s+\d+', text, re.IGNORECASE):
                    element_type = "caption"
            
            # Add classification to item
            item_copy = item.copy()
            item_copy["element_type"] = element_type
            item_copy["heading_level"] = heading_level
            item_copy["skip"] = skip
            
            classified.append(item_copy)
        
        logger.info(f"Classified {len(classified)} text elements")
        return classified
    
    def detect_heading_level(self, text: str) -> Tuple[int, str]:
        """
        Detect heading level from text (e.g., "2.1 Introduction" -> level 2)
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (heading_level, heading_number_string)
            heading_level: 0 if not a heading, otherwise 1, 2, 3, etc.
            heading_number_string: The heading number (e.g., "2.1")
        """
        match = self.heading_pattern.match(text.strip())
        if not match:
            return 0, ""
        
        heading_number = match.group(0).strip()
        # Remove trailing dot if present
        heading_number = heading_number.rstrip('.')
        
        # Count dots to determine level
        level = heading_number.count('.') + 1
        
        return level, heading_number
    
    def is_level_2_heading(self, text: str) -> bool:
        """
        Check if text is a level 2 heading (e.g., "2.1", "3.2", etc.)
        
        Args:
            text: Text to check
            
        Returns:
            True if text is a level 2 heading
        """
        level, heading_number = self.detect_heading_level(text)
        
        # Level 2 heading has exactly one dot (e.g., "2.1", "3.2")
        return level == 2
    
    def should_insert_break(self, text: str) -> bool:
        """
        Determine if </break> tag should be inserted after this text
        
        Args:
            text: Text to check
            
        Returns:
            True if </break> should be inserted after this text
        """
        return self.is_level_2_heading(text)
    
    def group_into_sections(
        self,
        classified_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group classified results into logical sections based on headings
        
        Args:
            classified_results: List of classified OCR results
            
        Returns:
            List of sections with metadata
        """
        sections = []
        current_section = {
            "heading": None,
            "heading_level": 0,
            "content": [],
            "images": []
        }
        
        for item in classified_results:
            if item.get("skip", False):
                continue  # Skip headers and footers
            
            element_type = item.get("element_type", "paragraph")
            
            if element_type == "heading":
                # Start new section
                if current_section["content"] or current_section["heading"]:
                    sections.append(current_section)
                
                current_section = {
                    "heading": item.get("text", ""),
                    "heading_level": item.get("heading_level", 0),
                    "content": [],
                    "images": []
                }
            else:
                current_section["content"].append(item)
        
        # Add last section
        if current_section["content"] or current_section["heading"]:
            sections.append(current_section)
        
        logger.info(f"Grouped content into {len(sections)} sections")
        return sections
    
    def merge_with_images(
        self,
        classified_results: List[Dict[str, Any]],
        images: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge images into OCR results based on spatial position
        
        Args:
            classified_results: List of classified OCR results
            images: List of extracted images with bbox information
            
        Returns:
            Merged list with images inserted at appropriate positions
        """
        if not images:
            return classified_results
        
        merged = []
        
        # Create a combined list of text and images
        all_elements = []
        
        for item in classified_results:
            if item.get("skip", False):
                continue
            
            box = item.get("box", [])
            if box:
                y_center = sum(p[1] for p in box) / len(box)
                all_elements.append({
                    "type": "text",
                    "data": item,
                    "y_position": y_center
                })
        
        for img in images:
            bbox = img.get("bbox")
            if bbox:
                # bbox is [x0, y0, x1, y1]
                y_center = (bbox[1] + bbox[3]) / 2
                all_elements.append({
                    "type": "image",
                    "data": img,
                    "y_position": y_center
                })
        
        # Sort by vertical position
        all_elements.sort(key=lambda x: x["y_position"])
        
        # Extract sorted data
        for elem in all_elements:
            if elem["type"] == "text":
                merged.append(elem["data"])
            else:
                # Add image as special element
                img_element = elem["data"].copy()
                img_element["element_type"] = "image"
                merged.append(img_element)
        
        logger.info(f"Merged {len(images)} images with text elements")
        return merged
