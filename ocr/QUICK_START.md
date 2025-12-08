# Quick Start Guide - Enhanced OCR Pipeline

## 🚀 Get Started in 3 Steps

### Step 1: Check Dependencies
```bash
# Navigate to ocr directory
cd c:\Users\Admin\Documents\23CLCT2_TraditionalMedicineChatbot\ocr

# Check if dependencies are installed
python main.py --help
```

If you see errors about missing libraries, install them:
```bash
pip install -r requirements.txt
```

### Step 2: Run OCR
```bash
# Basic OCR with all features
python main.py --input ..\input\your_document.pdf --output .\output

# With EasyDataset processing
python main.py --input ..\input\your_document.pdf --output .\output --easydataset
```

### Step 3: Check Results
```bash
# Your outputs will be in the output directory:
# - your_document.docx (Word file with images)
# - your_document_ocr_results.json (OCR data)
# - your_document_easydataset.json (if --easydataset used)
```

## 📋 What You Get

### Output Files

For input: `medical_book.pdf`

**Generated files**:
1. ✅ `medical_book.docx` - Word document with:
   - Complete text (no headers/footers)
   - Embedded images with captions `[img_001]`, `[img_002]`
   - Preserved layout and formatting
   - `</break>` markers after level 2 headings (2.1, 2.2, etc.)

2. ✅ `medical_book_ocr_results.json` - Full OCR data with:
   - Text for each page
   - Image metadata and locations
   - Element classifications (heading, paragraph, etc.)

3. ✅ `medical_book_easydataset.json` - Structured dataset (if `--easydataset` used)
4. ✅ `medical_book_qa.json` - Q&A format (if `--easydataset` used)
5. ✅ `medical_book_retrieval.json` - Retrieval format (if `--easydataset` used)
6. ✅ `temp/extracted_images/*.png` - All extracted images

## 🎯 Key Features Demonstrated

### Feature 1: Header/Footer Removal ✅
**Requirement**: "Đầy đủ nội dung đối với phần chữ (có thể bỏ header, footer)"

Open the generated Word file - you'll see:
- No page numbers at top/bottom
- No repeated headers
- Only main content

### Feature 2: Image Extraction & Embedding ✅
**Requirement**: "Đầy đủ hình ảnh, đồ thị (lưu dưới dạng file hình ảnh có đánh mã và đính kèm vào nội dung)"

Check these:
- Images saved in `temp/extracted_images/` with IDs: `img_001.png`, `img_002.png`
- Images embedded in Word file
- Captions below images: `[img_001]`, `[img_002]`

### Feature 3: Layout Preservation ✅
**Requirement**: "Giữ nguyên bố cục, cấu trúc, bố cục nội dung"

Compare PDF and Word:
- Headings are bold and larger
- Spacing matches original
- Structure preserved

### Feature 4: Section Break Markers ✅
**Requirement**: "Giữa các chỉ mục cấp độ 2 (2.1, 2.2,...) chèn chuỗi ký tự </break>"

Open Word and search for `</break>`:
```
2.1 Introduction </break>
2.2 Methodology </break>
3.1 Results </break>
```

### Feature 5: Word Output ✅
**Requirement**: "Định dạng xuất ra là Word"

- Professional `.docx` file
- Compatible with Microsoft Word
- Images embedded (not linked)

### Feature 6: EasyDataset Integration ✅
**Requirement**: "Đề xuất sử dụng EasyDataset để xử lý dữ liệu"

Run with `--easydataset` flag to get:
- Structured dataset split by `</break>` markers
- Text chunks for processing
- Q&A and Retrieval formats

## 🧪 Quick Test

### Test with Sample Document

1. Place your PDF in the input folder:
```bash
# Example path
c:\Users\Admin\Documents\23CLCT2_TraditionalMedicineChatbot\input\test.pdf
```

2. Run OCR:
```bash
python main.py --input ..\input\test.pdf --output .\output --easydataset
```

3. Check outputs:
```bash
# Should see these files:
dir output\test.*

# Expected files:
# test.docx
# test_ocr_results.json
# test_easydataset.json
# test_qa.json
# test_retrieval.json
```

4. Verify in Word:
- Open `output\test.docx`
- Check for images with captions
- Search for `</break>` (should appear after headings like 2.1, 2.2)
- Verify no headers/footers

## 📊 Understanding the Output

### Word Document Structure
```
[No header]

1. Main Heading
   Bold, large font

2. Section Heading
   Bold, medium font

2.1 Subsection Heading </break>
   Bold, medium font, with break marker

Regular paragraph text goes here...

[img_001]
(Image embedded here with caption)

2.2 Another Subsection </break>

More content...

[No footer]
```

### JSON Structure (OCR Results)
```json
{
  "pages": [
    {
      "page_num": 1,
      "results": [
        {
          "text": "2.1 Introduction </break>",
          "element_type": "heading",
          "heading_level": 2,
          "skip": false
        }
      ]
    }
  ],
  "images": [
    {
      "image_id": "img_001",
      "page_num": 1,
      "file_path": "..."
    }
  ]
}
```

## 🔧 Common Commands

### Full feature processing
```bash
python main.py --input ..\input\document.pdf --output .\output --easydataset
```

### High quality OCR
```bash
python main.py --input ..\input\document.pdf --dpi 600
```

### Fast processing (parallel)
```bash
python main.py --input ..\input\document.pdf --workers 4
```

### Text only (no images)
```bash
python main.py --input ..\input\document.pdf --no-images
```

### Simple OCR (no structure analysis)
```bash
python main.py --input ..\input\document.pdf --no-layout
```

## ✅ Verification Checklist

After running OCR, verify:

- [ ] Word file created and opens correctly
- [ ] No headers/footers in Word file
- [ ] Images are embedded (not broken links)
- [ ] Images have captions like `[img_001]`
- [ ] Headings are bold and formatted
- [ ] Search for `</break>` finds entries (should be after 2.1, 2.2, etc.)
- [ ] JSON file contains image metadata
- [ ] Extracted images exist in temp folder
- [ ] EasyDataset files created (if flag used)
- [ ] Sections in EasyDataset match level 2 headings

## 🆘 Troubleshooting

### Problem: "No input path provided"
**Solution**: Specify input with `--input` flag
```bash
python main.py --input ..\input\your_file.pdf
```

### Problem: "Missing required libraries"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Problem: No images in output
**Possible causes**:
1. PDF has no embedded images
2. Images too small (filtered out)
3. Image extraction disabled

**Solution**: 
```bash
# Check if extraction is enabled (should be by default)
python main.py --input your.pdf  # Images enabled by default

# If still no images, PDF might not have embedded images
```

### Problem: Headers/footers still present
**Solution**: Adjust margins in config.py
```python
PAGE_MARGIN_TOP = 0.15      # Increase from 0.1 to 0.15
PAGE_MARGIN_BOTTOM = 0.15
```

### Problem: No `</break>` markers
**Check**:
1. Are there level 2 headings in source? (2.1, 2.2, etc.)
2. Do headings have space after number? ("2.1 Title" not "2.1Title")
3. Check JSON to see if headings detected

## 📚 Next Steps

1. **Run on your document**: Try with your actual PDF
2. **Review Word output**: Check quality and accuracy
3. **Use EasyDataset**: Process for your chatbot/NLP tasks
4. **Customize settings**: Adjust config.py for your needs
5. **Read full docs**: Check ENHANCED_FEATURES.md for details

## 🎓 Learning Resources

- `ENHANCED_FEATURES.md` - Complete feature documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `HUONG_DAN_TIENG_VIET.md` - Vietnamese guide
- `examples.py` - Code examples
- Module docstrings - API documentation

## 🎉 Success Criteria

Your OCR is working correctly if:
✅ Word file has complete text without headers/footers
✅ Images are embedded with numbered captions
✅ Layout matches original document
✅ Level 2 headings have `</break>` markers
✅ JSON contains structured data
✅ EasyDataset splits correctly by sections

## 💬 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages in terminal
3. Examine JSON output for debugging
4. Read module docstrings for details

---

**Ready to process 3000+ sentences? Start with Step 1 above! 🚀**
