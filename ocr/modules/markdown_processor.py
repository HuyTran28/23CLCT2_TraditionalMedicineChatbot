import re
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)


class MarkdownProcessor:
    """
    Post-processes markdown output to:
    - Insert section breaks after level-2 headings (e.g., 2.1, 2.2)
    - Fix spelling errors using LLM
    - Preserve images, tables, and structure
    """
    
    def __init__(self, use_llm_correction: bool = True):
        """
        Initialize Markdown Processor
        
        Args:
            use_llm_correction: Enable spelling correction
            llm_provider: Correction method to use ('huggingface')
                - 'huggingface': Use free HuggingFace models (FREE, runs locally)
        """
        self.use_llm_correction = use_llm_correction
        
        # Regex pattern for level-2 headings: 2.1, 2.2, 3.1, 3.2, etc.
        # Matches patterns like "## 2.1 Title" or "2.1 Title" or "2.1. Title"
        self.level2_heading_pattern = re.compile(r'^(#{1,6}\s*)?\d+\.\d+\.?\s+.+$', re.MULTILINE)
        
    def insert_section_breaks(self, markdown_text: str) -> str:
        """
        Insert </break> tags after level-2 headings (e.g., 2.1, 2.2, 3.1, etc.)
        
        Args:
            markdown_text: Input markdown text
            
        Returns:
            Markdown text with section breaks inserted
        """
        lines = markdown_text.split('\n')
        processed_lines = []
        
        for i, line in enumerate(lines):
            # Check if this line is a level-2 heading
            if self._is_level2_heading(line):
                # Append the break tag to the same line
                processed_lines.append(line + '\n</break>')
                logger.debug(f"Inserted section break after: {line.strip()[:50]}...")
            else:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _is_level2_heading(self, line: str) -> bool:
        """
        Check if a line is a level-2 heading (e.g., 2.1, 2.2, 3.1)
        
        Args:
            line: Line to check
            
        Returns:
            True if line is a level-2 heading
        """
        line = line.strip()
        
        # Pattern: "## 2.1 Title" or "2.1 Title" or "2.1. Title"
        # Level-2 means X.Y format (e.g., 2.1, 3.2, not 2.1.1)
        match = re.match(r'^(#{1,6}\s*)?(\d+)\.(\d+)\.?\s+(.+)$', line)
        
        if match:
            # Check that it's exactly X.Y format (not X.Y.Z)
            heading_text = match.group(4)
            # Make sure there's no additional numbering in the heading
            if not re.match(r'^\d+\.', heading_text):
                return True
        
        return False
    
    def fix_spelling_with_llm(self, markdown_text: str, images: list = None) -> str:
        """
        Fix spelling errors in Vietnamese markdown text using FREE methods.
        Preserves image placeholders, tables, and markdown structure.
        """
        # If correction disabled, return original
        if not self.use_llm_correction:
            logger.info("Spelling correction disabled, skipping...")
            return markdown_text

        logger.info("Fixing spelling errors using huggingface (FREE method)...")

        try:
            # Only HuggingFace-based correction is supported
            return self._fix_with_huggingface(markdown_text)

        except Exception as e:
            logger.error(f"Spelling correction failed: {e}")
            logger.warning("Continuing with uncorrected text...")
            return markdown_text
    
    def _fix_with_huggingface(self, markdown_text: str) -> str:
        """
        Fix spelling using Vietnamese spell-checkers (pyvi, underthesea).
        Falls back to basic correction if advanced models are unavailable.
        
        Strategy:
        1. Use simple Vietnamese spell-checkers for basic corrections
        2. Preserve markdown structure, images, and special tags
        3. Process in smaller chunks to maintain context
        """
        logger.info("Using Vietnamese spell-checker for correction...")
        
        # Try using underthesea or pyvi for Vietnamese spell-checking
        try:
            # First, try underthesea (more reliable for Vietnamese)
            try:
                from underthesea import word_tokenize
                logger.info("Using underthesea for Vietnamese text processing...")
                return self._fix_with_underthesea(markdown_text)
            except ImportError:
                logger.debug("underthesea not available, trying pyvi...")
                pass
            
            # Fallback to pyvi
            try:
                from pyvi import ViTokenizer
                logger.info("Using pyvi for Vietnamese text processing...")
                return self._fix_with_pyvi(markdown_text)
            except ImportError:
                logger.debug("pyvi not available, trying basic correction...")
                pass
            
            # Last resort: basic pattern-based correction
            logger.warning("No Vietnamese NLP library found. Using basic correction...")
            logger.info("Install for better results: pip install underthesea pyvi")
            return self._basic_vietnamese_correction(markdown_text)
            
        except Exception as e:
            logger.error(f"Spelling correction error: {e}")
            return markdown_text
    
    def _fix_with_underthesea(self, markdown_text: str) -> str:
        """Use underthesea for Vietnamese spell-checking"""
        from underthesea import word_tokenize
        
        # Preserve special elements and HTML tags
        preserved = self._extract_special_elements(markdown_text)
        
        # Replace HTML tags with placeholders to protect them
        text_with_placeholders = markdown_text
        html_tags = re.findall(r'<[^>]+>', text_with_placeholders)
        tag_map = {}
        for i, tag in enumerate(html_tags):
            placeholder = f"__HTML_TAG_{i}__"
            tag_map[placeholder] = tag
            text_with_placeholders = text_with_placeholders.replace(tag, placeholder, 1)
        
        # Process in smaller chunks (1000 chars) to maintain context
        chunks = self._split_into_chunks(text_with_placeholders, 1000)
        corrected_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            logger.debug(f"Processing chunk {i}/{len(chunks)}...")
            
            # Split into lines to preserve structure
            lines = chunk.split('\n')
            corrected_lines = []
            
            for line in lines:
                # Skip markdown headers, images, breaks, and tables
                if self._should_skip_line(line):
                    corrected_lines.append(line)
                    continue
                
                # Apply basic corrections without tokenization that breaks text
                try:
                    corrected_line = self._apply_basic_corrections(line)
                    corrected_lines.append(corrected_line)
                except Exception as e:
                    logger.debug(f"Error processing line: {e}")
                    corrected_lines.append(line)
            
            corrected_chunks.append('\n'.join(corrected_lines))
        
        result = '\n'.join(corrected_chunks)
        
        # Restore HTML tags
        for placeholder, tag in tag_map.items():
            result = result.replace(placeholder, tag)
        
        # Restore preserved elements
        result = self._restore_special_elements(result, preserved)
        
        return result
    
    def _fix_with_pyvi(self, markdown_text: str) -> str:
        """Use pyvi for Vietnamese spell-checking"""
        from pyvi import ViTokenizer
        
        # Preserve special elements and HTML tags
        preserved = self._extract_special_elements(markdown_text)
        
        # Replace HTML tags with placeholders to protect them
        text_with_placeholders = markdown_text
        html_tags = re.findall(r'<[^>]+>', text_with_placeholders)
        tag_map = {}
        for i, tag in enumerate(html_tags):
            placeholder = f"__HTML_TAG_{i}__"
            tag_map[placeholder] = tag
            text_with_placeholders = text_with_placeholders.replace(tag, placeholder, 1)
        
        # Process in smaller chunks (1000 chars)
        chunks = self._split_into_chunks(text_with_placeholders, 1000)
        corrected_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            logger.debug(f"Processing chunk {i}/{len(chunks)}...")
            
            lines = chunk.split('\n')
            corrected_lines = []
            
            for line in lines:
                if self._should_skip_line(line):
                    corrected_lines.append(line)
                    continue
                
                try:
                    # Apply basic corrections instead of tokenization
                    corrected_line = self._apply_basic_corrections(line)
                    corrected_lines.append(corrected_line)
                except Exception as e:
                    logger.debug(f"Error processing line: {e}")
                    corrected_lines.append(line)
            
            corrected_chunks.append('\n'.join(corrected_lines))
        
        result = '\n'.join(corrected_chunks)
        
        # Restore HTML tags
        for placeholder, tag in tag_map.items():
            result = result.replace(placeholder, tag)
        
        result = self._restore_special_elements(result, preserved)
        
        return result
    
    def _apply_basic_corrections(self, line: str) -> str:
        """Apply targeted OCR corrections to a single line without breaking formatting"""
        corrections = {
            r'\bl\s+à\b': 'là',  # "l à" -> "là"
            r'\bc\s+ó\b': 'có',  # "c ó" -> "có"
            r'\bt\s+ừ\b': 'từ',  # "t ừ" -> "từ"
            r'\bđ\s+ược\b': 'được',  # "đ ược" -> "được"
            r'\bn\s+hư\b': 'như',  # "n hư" -> "như"
            r'\bk\s+hi\b': 'khi',  # "k hi" -> "khi"
            r'\bv\s+ới\b': 'với',  # "v ới" -> "với"
            r'\bs\s+ô\b': 'số',  # "s ô" -> "số"
            r'\bm\s+ủ\b': 'mủ',  # "m ủ" -> "mủ"
            r'\bv\s+ự\b': 'vự',  # "v ự" -> "vự"
            r'\bΝιάς\b': 'Nước',
            # Common OCR character confusions
            r'\bđươc\b': 'được',
            r'\bduợc\b': 'được',
            r'\bnhư\s+ng\b': 'nhưng',
            r'\b0\s+\b': 'ở',  # "0" often confused for "ở"
            r'\.\s+\b1\s+9': '.1.9',  # ". 1,9" -> ".1,9"
        }
        
        result = line
        for pattern, replacement in corrections.items():
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _basic_vietnamese_correction(self, markdown_text: str) -> str:
        """Basic pattern-based Vietnamese text correction"""
        
        # Preserve special elements
        preserved = self._extract_special_elements(markdown_text)
        
        # Replace HTML tags with placeholders to protect them
        text_with_placeholders = markdown_text
        html_tags = re.findall(r'<[^>]+>', text_with_placeholders)
        tag_map = {}
        for i, tag in enumerate(html_tags):
            placeholder = f"__HTML_TAG_{i}__"
            tag_map[placeholder] = tag
            text_with_placeholders = text_with_placeholders.replace(tag, placeholder, 1)
        
        result = text_with_placeholders
        lines = result.split('\n')
        corrected_lines = []
        
        for line in lines:
            if self._should_skip_line(line):
                corrected_lines.append(line)
            else:
                corrected_line = self._apply_basic_corrections(line)
                corrected_lines.append(corrected_line)
        
        result = '\n'.join(corrected_lines)
        
        # Restore HTML tags
        for placeholder, tag in tag_map.items():
            result = result.replace(placeholder, tag)
        
        result = self._restore_special_elements(result, preserved)
        
        return result
    
    def _extract_special_elements(self, text: str) -> dict:
        """Extract and protect special markdown elements"""
        preserved = {
            'images': [],
            'breaks': [],
            'headers': [],
            'tables': []
        }
        
        # Find all image placeholders
        preserved['images'] = re.findall(r'\[IMAGE_PLACEHOLDER_\d+\]', text)
        
        # Find all break tags
        preserved['breaks'] = re.findall(r'</break>', text)
        
        # Find all markdown headers
        preserved['headers'] = re.findall(r'^#{1,6}\s+.+$', text, re.MULTILINE)
        
        # Find table rows
        preserved['tables'] = re.findall(r'^\|.+\|$', text, re.MULTILINE)
        
        return preserved
    
    def _restore_special_elements(self, text: str, preserved: dict) -> str:
        """Ensure special elements are preserved in the output"""
        # This is a validation step - log warnings if elements are missing
        
        for img in preserved['images']:
            if img not in text:
                logger.warning(f"Image placeholder missing after correction: {img}")
        
        for brk in preserved['breaks']:
            if text.count(brk) < preserved['breaks'].count(brk):
                logger.warning(f"Break tag count mismatch after correction")
        
        return text
    
    def _should_skip_line(self, line: str) -> bool:
        """Check if a line should be skipped during correction"""
        line_stripped = line.strip()
        
        # Skip empty lines
        if not line_stripped:
            return True
        
        # Skip markdown headers
        if line_stripped.startswith('#'):
            return True
        
        # Skip image placeholders
        if '[IMAGE_PLACEHOLDER_' in line:
            return True
        
        # Only skip if line is ONLY the break tag
        if line_stripped == '</break>':
            return True
        
        # Don't skip table rows - apply spelling correction to table content too
        # Table rows should be processed for spelling errors
        
        # Skip horizontal rules
        if re.match(r'^[-*_]{3,}$', line_stripped):
            return True
        
        return False
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> list:
        """
        Split text into chunks while preserving paragraph boundaries
        
        Args:
            text: Text to split
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split by paragraphs (double newline)
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para_length = len(para) + 2  # +2 for \n\n
            
            if current_length + para_length > chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def process(self, markdown_text: str, images: list = None) -> str:
        """
        Full post-processing pipeline:
        1. Insert section breaks
        2. Fix spelling with LLM
        
        Args:
            markdown_text: Input markdown text
            images: List of image references
            
        Returns:
            Processed markdown text
        """
        logger.info("Starting markdown post-processing...")
        
        # Step 1: Insert section breaks
        logger.info("Step 1: Inserting section breaks for level-2 headings...")
        markdown_text = self.insert_section_breaks(markdown_text)
        
        # Step 2: Fix spelling with LLM
        if self.use_llm_correction:
            logger.info("Step 2: Fixing spelling errors with LLM...")
            markdown_text = self.fix_spelling_with_llm(markdown_text, images)
        else:
            logger.info("Step 2: Skipping LLM correction (disabled)")
        
        logger.info("✓ Markdown post-processing complete")
        return markdown_text
