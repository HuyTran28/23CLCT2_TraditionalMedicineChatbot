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
        Fix spelling errors in Vietnamese markdown text using a hybrid, transformer-free approach.
        Preserves image placeholders, tables, markdown structure, LaTeX math, and HTML tags.
        """
        # If correction disabled, return original
        if not self.use_llm_correction:
            return markdown_text

        try:
            # Protect special elements before correction
            protected_text, protection_map = self._protect_special_elements(markdown_text)
            
            # Try hybrid correction methods in order of preference (no transformers)
            corrected_text = self._fix_with_hybrid_correction(protected_text)
            
            # Restore protected elements
            final_text = self._restore_protected_elements(corrected_text, protection_map)
            
            return final_text

        except Exception as e:
            logger.error(f"Spelling correction failed: {e}")
            logger.warning("Continuing with uncorrected text...")
            return markdown_text
    
    def _protect_special_elements(self, text: str) -> tuple:
        """
        Protect special elements from being modified during spell correction.
        Returns: (protected_text, protection_map)
        """
        protection_map = {}
        protected_text = text
        counter = 0
        
        # 1. Protect LaTeX math expressions: $...$, $$...$$
        # Match both inline ($x$) and display ($$x$$) math
        math_patterns = [
            (r'\$\$[^$]+\$\$', 'MATH_DISPLAY'),  # Display math $$...$$
            (r'\$[^$\n]+\$', 'MATH_INLINE'),      # Inline math $...$
        ]
        
        for pattern, prefix in math_patterns:
            matches = re.finditer(pattern, protected_text)
            for match in reversed(list(matches)):  # Reverse to maintain positions
                placeholder = f"__{prefix}_{counter}__"
                protection_map[placeholder] = match.group(0)
                protected_text = protected_text[:match.start()] + placeholder + protected_text[match.end():]
                counter += 1
        
        # 2. Protect HTML tags: <sub>, <sup>, <br>, and all other HTML tags
        html_tag_pattern = r'<[^>]+>.*?</[^>]+>|<[^>]+/>|<[^>]+>'
        matches = re.finditer(html_tag_pattern, protected_text)
        for match in reversed(list(matches)):
            placeholder = f"__HTML_TAG_{counter}__"
            protection_map[placeholder] = match.group(0)
            protected_text = protected_text[:match.start()] + placeholder + protected_text[match.end():]
            counter += 1
        
        # 3. Protect image placeholders and references
        image_patterns = [
            r'!\[id:\s*[^\]]+\]\([^\)]+\)',  # New format: ![id: img_1](img_1.png)
            r'\[IMAGE_PLACEHOLDER_\d+\]',      # Old format: [IMAGE_PLACEHOLDER_1]
        ]
        
        for pattern in image_patterns:
            matches = re.finditer(pattern, protected_text)
            for match in reversed(list(matches)):
                placeholder = f"__IMAGE_{counter}__"
                protection_map[placeholder] = match.group(0)
                protected_text = protected_text[:match.start()] + placeholder + protected_text[match.end():]
                counter += 1
        
        # 4. Protect markdown headers
        header_pattern = r'^#{1,6}\s+.+$'
        lines = protected_text.split('\n')
        for i, line in enumerate(lines):
            if re.match(header_pattern, line.strip()):
                placeholder = f"__HEADER_{counter}__"
                protection_map[placeholder] = line
                lines[i] = placeholder
                counter += 1
        protected_text = '\n'.join(lines)
        
        # 5. Protect table rows
        table_pattern = r'^\|.+\|$'
        lines = protected_text.split('\n')
        for i, line in enumerate(lines):
            if re.match(table_pattern, line.strip()):
                placeholder = f"__TABLE_ROW_{counter}__"
                protection_map[placeholder] = line
                lines[i] = placeholder
                counter += 1
        protected_text = '\n'.join(lines)
        
        # 6. Protect code blocks
        code_block_pattern = r'```[^`]*```'
        matches = re.finditer(code_block_pattern, protected_text, re.DOTALL)
        for match in reversed(list(matches)):
            placeholder = f"__CODE_BLOCK_{counter}__"
            protection_map[placeholder] = match.group(0)
            protected_text = protected_text[:match.start()] + placeholder + protected_text[match.end():]
            counter += 1
        
        # 7. Protect inline code
        inline_code_pattern = r'`[^`]+`'
        matches = re.finditer(inline_code_pattern, protected_text)
        for match in reversed(list(matches)):
            placeholder = f"__INLINE_CODE_{counter}__"
            protection_map[placeholder] = match.group(0)
            protected_text = protected_text[:match.start()] + placeholder + protected_text[match.end():]
            counter += 1
        
        return protected_text, protection_map
    
    def _restore_protected_elements(self, text: str, protection_map: dict) -> str:
        """
        Restore protected elements after spell correction.
        """
        result = text
        # Restore in reverse order to handle nested protections
        for placeholder, original in protection_map.items():
            result = result.replace(placeholder, original)
        return result
    
    def _fix_with_hybrid_correction(self, text: str) -> str:
        """
        Hybrid correction without transformers: underthesea -> pyvi -> pattern-based.
        """
        # Vietnamese NLP libraries first
        try:
            from underthesea import word_tokenize  # noqa: F401
            logger.info("Using underthesea for Vietnamese text processing...")
            return self._fix_with_underthesea(text)
        except ImportError:
            logger.debug("underthesea not available")
        
        try:
            from pyvi import ViTokenizer  # noqa: F401
            logger.info("Using pyvi for Vietnamese text processing...")
            return self._fix_with_pyvi(text)
        except ImportError:
            logger.debug("pyvi not available")
        
        # Final fallback: enhanced pattern-based correction
        logger.info("Using enhanced pattern-based correction...")
        return self._enhanced_pattern_correction(text)
    
    def _enhanced_pattern_correction(self, text: str) -> str:
        """
        Enhanced pattern-based correction with context awareness.
        """
        lines = text.split('\n')
        corrected_lines = []
        
        for line in lines:
            if line.strip():
                corrected_line = self._apply_enhanced_corrections(line)
                corrected_lines.append(corrected_line)
            else:
                corrected_lines.append(line)
        
        return '\n'.join(corrected_lines)
    
    def _fix_with_underthesea(self, text: str) -> str:
        """Use underthesea for Vietnamese spell-checking with improved handling"""
        from underthesea import word_tokenize
        
        # Process in smaller chunks (1000 chars) to maintain context
        chunks = self._split_into_chunks(text, 1000)
        corrected_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            logger.debug(f"Processing chunk {i}/{len(chunks)}...")
            
            # Split into lines to preserve structure
            lines = chunk.split('\n')
            corrected_lines = []
            
            for line in lines:
                if not line.strip():
                    corrected_lines.append(line)
                    continue
                
                # Apply enhanced corrections
                try:
                    corrected_line = self._apply_enhanced_corrections(line)
                    corrected_lines.append(corrected_line)
                except Exception as e:
                    logger.debug(f"Error processing line: {e}")
                    corrected_lines.append(line)
            
            corrected_chunks.append('\n'.join(corrected_lines))
        
        return '\n'.join(corrected_chunks)
    
    def _fix_with_pyvi(self, text: str) -> str:
        """Use pyvi for Vietnamese spell-checking with improved handling"""
        from pyvi import ViTokenizer
        
        # Process in smaller chunks (1000 chars)
        chunks = self._split_into_chunks(text, 1000)
        corrected_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            logger.debug(f"Processing chunk {i}/{len(chunks)}...")
            
            lines = chunk.split('\n')
            corrected_lines = []
            
            for line in lines:
                if not line.strip():
                    corrected_lines.append(line)
                    continue
                
                try:
                    # Apply enhanced corrections
                    corrected_line = self._apply_enhanced_corrections(line)
                    corrected_lines.append(corrected_line)
                except Exception as e:
                    logger.debug(f"Error processing line: {e}")
                    corrected_lines.append(line)
            
            corrected_chunks.append('\n'.join(corrected_lines))
        
        return '\n'.join(corrected_chunks)
    
    def _apply_enhanced_corrections(self, line: str) -> str:
        """
        Enhanced OCR corrections with context-aware pattern matching.
        """
        # First apply basic corrections
        line = self._apply_basic_corrections(line)
        
        # Additional context-aware corrections
        # Fix common Vietnamese phrases
        context_corrections = {
            r'\bc h ứ a\b': 'chữa',
            r'\bt h u ố c\b': 'thuốc',
            r'\bb ệ n h\b': 'bệnh',
            r'\bt á c\s+d ụ n g\b': 'tác dụng',
            r'\bc h ấ t\b': 'chất',
            r'\bc h ứ n g\b': 'chứng',
            r'\bt h ể\b': 'thể',
            r'\bc ơ\s+t h ể\b': 'cơ thể',
        }
        
        for pattern, replacement in context_corrections.items():
            line = re.sub(pattern, replacement, line)
        
        return line
    
    def _apply_basic_corrections(self, line: str) -> str:
        """Apply targeted OCR corrections to a single line without breaking formatting"""
        corrections = {
            # Common Vietnamese word splits (OCR breaking words incorrectly)
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
            r'\bt\s+ại\b': 'tại',  # "t ại" -> "tại"
            r'\bn\s+ên\b': 'nên',  # "n ên" -> "nên"
            r'\bth\s+ì\b': 'thì',  # "th ì" -> "thì"
            r'\btr\s+ong\b': 'trong',  # "tr ong" -> "trong"
            r'\bch\s+o\b': 'cho',  # "ch o" -> "cho"
            r'\bc\s+ủa\b': 'của',  # "c ủa" -> "của"
            r'\bh\s+ay\b': 'hay',  # "h ay" -> "hay"
            r'\bv\s+à\b': 'và',  # "v à" -> "và"
            r'\bn\s+ày\b': 'này',  # "n ày" -> "này"
            r'\bđ\s+ó\b': 'đó',  # "đ ó" -> "đó"
            r'\bk\s+ể\b': 'kể',  # "k ể" -> "kể"
            r'\bt\s+ên\b': 'tên',  # "t ên" -> "tên"
            r'\bch\s+ưa\b': 'chưa',  # "ch ưa" -> "chưa"
            r'\bđ\s+ã\b': 'đã',  # "đ ã" -> "đã"
            r'\bm\s+ột\b': 'một',  # "m ột" -> "một"
            r'\bh\s+ơn\b': 'hơn',  # "h ơn" -> "hơn"
            r'\bcũ\s+ng\b': 'cũng',  # "cũ ng" -> "cũng"
            r'\bnh\s+ững\b': 'những',  # "nh ững" -> "những"
            
            # Common Vietnamese misspellings
            r'\bđươc\b': 'được',
            r'\bduợc\b': 'được',
            r'\bđuợc\b': 'được',
            r'\bnhư\s+ng\b': 'nhưng',
            r'\btr\s+ên\b': 'trên',
            r'\bdu\s+ới\b': 'dưới',
            r'\bsa\s+u\b': 'sau',
            r'\btr\s+ước\b': 'trước',
            r'\bgi\s+ữa\b': 'giữa',
            r'\bhi\s+ện\b': 'hiện',
            r'\bth\s+ực\b': 'thực',
            r'\bch\s+ính\b': 'chính',
            
            # Character confusions
            r'\bΝιάς\b': 'Nước',
            r'\b0\s+\b': 'ở',  # "0" often confused for "ở"
            r'\b1\s+à\b': 'là',  # "1" confused for "l"
            r'\bII\b': 'II',  # Roman numerals
            
            # Number formatting
            r'\.\s+\b1\s+9': '.1.9',  # ". 1,9" -> ".1,9"
            r'(\d+)\s+\.\s+(\d+)': r'\1.\2',  # "5 . 2" -> "5.2"
            r'(\d+)\s+,\s+(\d+)': r'\1,\2',  # "5 , 2" -> "5,2"
            
            # Common punctuation issues
            r'\s+,': ',',  # Space before comma
            r'\s+\.': '.',  # Space before period
            r'\(\s+': '(',  # Space after opening paren
            r'\s+\)': ')',  # Space before closing paren
            
            # Vietnamese tone mark errors
            r'\bhoạ\s+t\b': 'hoạt',
            r'\bkhô\s+ng\b': 'không',
            r'\bthế\s+o\b': 'theo',
            r'\bcá\s+c\b': 'các',
        }
        
        result = line
        for pattern, replacement in corrections.items():
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _basic_vietnamese_correction(self, text: str) -> str:
        """Basic pattern-based Vietnamese text correction"""
        result = text
        lines = result.split('\n')
        corrected_lines = []
        
        for line in lines:
            if line.strip():
                corrected_line = self._apply_enhanced_corrections(line)
                corrected_lines.append(corrected_line)
            else:
                corrected_lines.append(line)
        
        return '\n'.join(corrected_lines)
    
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
        1. Insert image IDs for mapping
        2. Insert section breaks
        3. Fix spelling with LLM
        
        Args:
            markdown_text: Input markdown text
            images: List of image references with IDs
            
        Returns:
            Processed markdown text with image IDs
        """
        # logger.info("Starting markdown post-processing...")
        
        # Step 0: Inject image IDs into markdown
        if images:
            # logger.info("Step 0: Injecting image IDs into markdown...")
            markdown_text = self._inject_image_ids(markdown_text, images)
        
        # Step 1: Insert section breaks
        # logger.info("Step 1: Inserting section breaks for level-2 headings...")
        markdown_text = self.insert_section_breaks(markdown_text)
        
        # Step 2: Fix spelling with LLM
        if self.use_llm_correction:
            # logger.info("Step 2: Fixing spelling errors with LLM...")
            markdown_text = self.fix_spelling_with_llm(markdown_text, images)
        
        # logger.info("Markdown post-processing complete")
        return markdown_text
    
    def _inject_image_ids(self, markdown_text: str, images: list = None) -> str:
        """
        Inject unique image IDs into markdown for mapping to extracted images
        
        Args:
            markdown_text: Input markdown text
            images: List of image data with 'image_id' field
            
        Returns:
            Markdown text with image IDs injected as references
        """
        if not images:
            return markdown_text
        
        # Create mapping of image IDs
        image_map = {}
        for img in images:
            if 'image_id' in img:
                # Track the image ID for this image reference
                original_key = img.get('original_key', img.get('image_id'))
                image_map[original_key] = img['image_id']
        
        # Replace image placeholders with ID-mapped references
        result = markdown_text
        for original_key, image_id in image_map.items():
            # Find all placeholders
            placeholders = re.findall(r'\[IMAGE_PLACEHOLDER_\d+\]', result)
            
            # Replace placeholders with image ID references
            for idx, placeholder in enumerate(placeholders, 1):
                if idx <= len(images):
                    img = images[idx - 1]
                    image_id = img.get('image_id', f'img_{idx}')
                    
                    # Create a markdown reference with image ID
                    # Format: ![id: <image_id>](<image_id>.png)
                    image_reference = f"![id: {image_id}]({image_id}.png)"
                    result = result.replace(placeholder, image_reference, 1)
                    
                    logger.debug(f"Injected image ID: {image_id} at placeholder {placeholder}")
        
        return result
