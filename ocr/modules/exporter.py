# Exporter module (Person 3)
# Contains python-docx & tagging logic
import json
from docx import Document
from docx.shared import Pt
import re


def center_y(box):
    return sum(p[1] for p in box) / len(box)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def inject_break_tag(text: str) -> str:
    pattern = r"^\s*\d+\.\d+"  
    if re.match(pattern, text.strip()):
        return text.rstrip() + " </break>"
    return text


def json_to_docx(json_path, output_path, base_spacing=1.0, font_size=12):
    data = load_json(json_path)

    items = data.get("results", data)

    pages = {}
    for it in items:
        page_id = it.get("page_id", 1)
        pages.setdefault(page_id, []).append(it)

    doc = Document()

    for page_index, (page_id, page_items) in enumerate(pages.items()):
        page_items = sorted(
            page_items,
            key=lambda t: (center_y(t["box"]), t["box"][0][0])  # sort theo Y → X
        )

        last_y = None
        last_height = 12  

        for it in page_items:
            text = it["text"]
            box = it["box"]

            text = inject_break_tag(text)

            y = center_y(box)
            height = abs(box[2][1] - box[0][1]) 

            if last_y is None:
                spacing = base_spacing
            else:
                gap = y - last_y
                spacing = max(base_spacing, gap / last_height)

            p = doc.add_paragraph()
            run = p.add_run(text)
            run.font.size = Pt(font_size)

            p.paragraph_format.line_spacing = spacing

            last_y = y
            last_height = max(height, 8)

        if page_index < len(pages) - 1:
            doc.add_page_break()

    doc.save(output_path)
    print("Saved:", output_path)


if __name__ == "__main__":
    json_to_docx("vietnamese_ocr_results.json", "output.docx")