# Exporter module (Person 3)
# Contains python-docx & tagging logic
import json
from docx import Document
from docx.shared import Pt

def load_ocr_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def center_y(box):
    return sum(p[1] for p in box) / len(box)

def json_to_word(json_path, output_path, font_size=12, spacing_factor=1.05):
    data = load_ocr_data(json_path)

    pages = {}
    for item in data:
        page_id = item.get("page_id", 1)
        pages.setdefault(page_id, []).append(item)

    doc = Document()

    for idx, (page_id, items) in enumerate(pages.items()):
        items = sorted(items, key=lambda t: (center_y(t["box"]), t["box"][0][0]))

        last_y = None

        for it in items:
            text = it["text"]
            y = center_y(it["box"])

            p = doc.add_paragraph()
            run = p.add_run(text)

            run.font.size = Pt(font_size)

            p_format = p.paragraph_format
            p_format.line_spacing = spacing_factor

            last_y = y

        if idx < len(pages) - 1:
            doc.add_page_break()

    doc.save(output_path)
    print("Saved:", output_path)


if __name__ == "__main__":
    json_to_word("ocr_result.json", "ocr_output.docx")
