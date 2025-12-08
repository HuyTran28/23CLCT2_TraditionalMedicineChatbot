# EasyDataset Integration Module
# Processes OCR output for use with EasyDataset framework
# Structures data for efficient post-processing and dataset creation

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


class EasyDatasetProcessor:
    """
    Process OCR results into EasyDataset-compatible format.
    
    EasyDataset is a framework for creating structured datasets from documents.
    This module transforms OCR output into a format suitable for:
    - Text segmentation and chunking
    - Question-answer pair generation
    - Training data preparation
    - Document retrieval systems
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        min_chunk_length: int = 100
    ):
        """
        Initialize EasyDataset Processor
        
        Args:
            chunk_size: Target size for text chunks (characters)
            overlap: Overlap between consecutive chunks (characters)
            min_chunk_length: Minimum length for a valid chunk
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_length = min_chunk_length
    
    def process_ocr_results(
        self,
        ocr_json_path: str | Path,
        output_path: Optional[str | Path] = None
    ) -> Dict[str, Any]:
        """
        Process OCR JSON results into EasyDataset format
        
        Args:
            ocr_json_path: Path to OCR results JSON file
            output_path: Optional path to save processed dataset
            
        Returns:
            Dictionary with processed dataset in EasyDataset format
        """
        ocr_json_path = Path(ocr_json_path)
        if not ocr_json_path.exists():
            raise FileNotFoundError(f"OCR results not found: {ocr_json_path}")
        
        # Load OCR results
        with open(ocr_json_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
        
        logger.info(f"Processing OCR results from {ocr_json_path.name}")
        
        # Extract sections using </break> markers
        sections = self._extract_sections(ocr_data)
        logger.info(f"Extracted {len(sections)} sections from document")
        
        # Create dataset entries
        dataset_entries = []
        for idx, section in enumerate(sections, 1):
            entry = self._create_dataset_entry(section, idx)
            dataset_entries.append(entry)
        
        # Create final dataset structure
        dataset = {
            "metadata": {
                "source_file": ocr_json_path.stem,
                "total_sections": len(sections),
                "total_entries": len(dataset_entries),
                "chunk_size": self.chunk_size,
                "overlap": self.overlap
            },
            "sections": dataset_entries
        }
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ Saved EasyDataset format to: {output_path}")
        
        return dataset
    
    def _extract_sections(self, ocr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract sections from OCR data using </break> markers
        
        Args:
            ocr_data: OCR results dictionary
            
        Returns:
            List of sections with text, images, and metadata
        """
        sections = []
        current_section = {
            "heading": "",
            "content": [],
            "images": [],
            "page_range": []
        }
        
        pages = ocr_data.get("pages", [])
        images = ocr_data.get("images", [])
        
        for page in pages:
            page_num = page.get("page_num", 1)
            results = page.get("results", [])
            
            for item in results:
                # Skip headers/footers
                if item.get("skip", False):
                    continue
                
                element_type = item.get("element_type", "paragraph")
                text = item.get("text", "").strip()
                
                if not text:
                    continue
                
                # Handle images
                if element_type == "image":
                    current_section["images"].append(item)
                    continue
                
                # Track page range
                if page_num not in current_section["page_range"]:
                    current_section["page_range"].append(page_num)
                
                # Check for section break
                if "</break>" in text:
                    # Remove the break marker
                    text = text.replace("</break>", "").strip()
                    
                    # Add current text
                    if element_type == "heading":
                        current_section["heading"] = text
                    else:
                        current_section["content"].append(text)
                    
                    # Save current section and start new one
                    if current_section["heading"] or current_section["content"]:
                        sections.append(current_section)
                    
                    current_section = {
                        "heading": "",
                        "content": [],
                        "images": [],
                        "page_range": []
                    }
                else:
                    # Add to current section
                    if element_type == "heading":
                        current_section["heading"] = text
                    else:
                        current_section["content"].append(text)
        
        # Add last section
        if current_section["heading"] or current_section["content"]:
            sections.append(current_section)
        
        return sections
    
    def _create_dataset_entry(
        self,
        section: Dict[str, Any],
        section_id: int
    ) -> Dict[str, Any]:
        """
        Create a dataset entry from a section
        
        Args:
            section: Section dictionary with heading, content, images
            section_id: Unique section identifier
            
        Returns:
            Dataset entry in EasyDataset format
        """
        heading = section.get("heading", "")
        content_parts = section.get("content", [])
        images = section.get("images", [])
        page_range = section.get("page_range", [])
        
        # Combine content
        full_text = " ".join(content_parts)
        
        # Create chunks if text is too long
        chunks = self._create_chunks(full_text)
        
        # Extract metadata
        metadata = {
            "section_id": section_id,
            "heading": heading,
            "page_range": page_range,
            "has_images": len(images) > 0,
            "image_count": len(images),
            "image_ids": [img.get("image_id", "") for img in images],
            "character_count": len(full_text),
            "chunk_count": len(chunks)
        }
        
        return {
            "id": f"section_{section_id:03d}",
            "heading": heading,
            "full_text": full_text,
            "chunks": chunks,
            "images": [
                {
                    "image_id": img.get("image_id", ""),
                    "file_path": img.get("file_path", ""),
                    "page_num": img.get("page_num", 0)
                }
                for img in images
            ],
            "metadata": metadata
        }
    
    def _create_chunks(self, text: str) -> List[Dict[str, str]]:
        """
        Split long text into overlapping chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk dictionaries with 'text' and 'position' keys
        """
        if len(text) <= self.chunk_size:
            return [{"text": text, "position": 0}]
        
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending punctuation
                for punct in ['. ', '! ', '? ', '。', '！', '？']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > start + self.min_chunk_length:
                        end = last_punct + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_length:
                chunks.append({
                    "text": chunk_text,
                    "position": start,
                    "chunk_id": chunk_idx
                })
                chunk_idx += 1
            
            # Move start position with overlap
            start = end - self.overlap
            if start >= len(text) - self.min_chunk_length:
                break
        
        return chunks
    
    def export_for_qa_generation(
        self,
        dataset: Dict[str, Any],
        output_path: str | Path
    ) -> Path:
        """
        Export dataset in format suitable for Q&A generation
        
        Args:
            dataset: Processed dataset
            output_path: Path to save Q&A format
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        
        qa_data = {
            "metadata": dataset.get("metadata", {}),
            "passages": []
        }
        
        for section in dataset.get("sections", []):
            for chunk_idx, chunk in enumerate(section.get("chunks", [])):
                passage = {
                    "id": f"{section['id']}_chunk_{chunk_idx}",
                    "section_heading": section.get("heading", ""),
                    "text": chunk.get("text", ""),
                    "context": {
                        "section_id": section.get("id", ""),
                        "has_images": section.get("metadata", {}).get("has_images", False),
                        "page_range": section.get("metadata", {}).get("page_range", [])
                    }
                }
                qa_data["passages"].append(passage)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Exported {len(qa_data['passages'])} passages for Q&A generation: {output_path}")
        return output_path
    
    def export_for_retrieval(
        self,
        dataset: Dict[str, Any],
        output_path: str | Path
    ) -> Path:
        """
        Export dataset in format suitable for retrieval systems (RAG)
        
        Args:
            dataset: Processed dataset
            output_path: Path to save retrieval format
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        
        retrieval_data = {
            "metadata": dataset.get("metadata", {}),
            "documents": []
        }
        
        for section in dataset.get("sections", []):
            doc = {
                "id": section.get("id", ""),
                "title": section.get("heading", "Untitled Section"),
                "content": section.get("full_text", ""),
                "chunks": [chunk.get("text", "") for chunk in section.get("chunks", [])],
                "metadata": {
                    "page_range": section.get("metadata", {}).get("page_range", []),
                    "has_images": section.get("metadata", {}).get("has_images", False),
                    "image_ids": section.get("metadata", {}).get("image_ids", [])
                }
            }
            retrieval_data["documents"].append(doc)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(retrieval_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Exported {len(retrieval_data['documents'])} documents for retrieval: {output_path}")
        return output_path


if __name__ == "__main__":
    # Example usage
    processor = EasyDatasetProcessor(chunk_size=512, overlap=50)
    
    # Process OCR results
    dataset = processor.process_ocr_results(
        "output/document_ocr_results.json",
        "output/document_easydataset.json"
    )
    
    # Export for Q&A generation
    processor.export_for_qa_generation(dataset, "output/document_qa.json")
    
    # Export for retrieval
    processor.export_for_retrieval(dataset, "output/document_retrieval.json")
