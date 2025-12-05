# Contains python-docx & tagging logic
import json
from docx import Document
from docx.shared import Pt
import re


class WordExporter:
    def __init__(self, base_spacing=1.0, font_size=12):
        self.base_spacing = base_spacing
        self.font_size = font_size

    @staticmethod
    def center_y(box):
        return sum(p[1] for p in box) / len(box)

    @staticmethod
    def inject_break_tag(text: str) -> str:
        pattern = r"^\s*\d+\.\d+"
        if re.match(pattern, text.strip()):
            return text.rstrip() + " </break>"
        return text

    def write_to_word(self, data, output_path="output.docx"):
        """
        Export OCR results to Word document
        
        Args:
            data: Can be:
                - str: path to JSON file
                - list: OCR results (assumes single page)
                - dict with 'results' key: OCR results
                - dict with 'pages' key: Multi-page OCR results
            output_path: str, path to save docx
        
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
            return self._write_multipage_to_word(data["pages"], output_path)
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
            page_items = sorted(
                page_items,
                key=lambda t: (self.center_y(t["box"]), t["box"][0][0])
            )

            last_y = None
            last_height = 12

            for it in page_items:
                text = it["text"]
                box = it["box"]

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

                p.paragraph_format.line_spacing = spacing

                last_y = y
                last_height = max(height, 8)

            if page_index < len(pages) - 1:
                doc.add_page_break()

        doc.save(output_path)
        print("Saved:", output_path)
        return output_path
    
    def _write_multipage_to_word(self, pages_data, output_path="output.docx"):
        """
        Export multi-page OCR results to Word document
        
        Args:
            pages_data: List of dicts with 'page_num' and 'results' keys
            output_path: str, path to save docx
        
        Returns:
            Path to saved document
        """
        doc = Document()
        
        # Sort pages by page number
        pages_data = sorted(pages_data, key=lambda p: p.get("page_num", 1))
        
        for page_idx, page_data in enumerate(pages_data):
            page_num = page_data.get("page_num", page_idx + 1)
            results = page_data.get("results", [])
            
            if not results:
                continue
            
            # Sort results by vertical position, then horizontal
            results_sorted = sorted(
                results,
                key=lambda t: (self.center_y(t["box"]), t["box"][0][0])
            )
            
            last_y = None
            last_height = 12
            
            for it in results_sorted:
                text = it["text"]
                box = it["box"]
                
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
                
                p.paragraph_format.line_spacing = spacing
                
                last_y = y
                last_height = max(height, 8)
            
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