# OCR Pipeline - Vietnamese Traditional Medicine Document Converter

Complete pipeline for converting Vietnamese PDF documents to Word format with automatic mode detection.

## Features

- ✅ **Automatic Detection**: Distinguishes between digital and scanned PDFs
- ✅ **Multi-Page Support**: Processes entire PDF documents
- ✅ **Batch Processing**: Convert multiple PDFs at once
- ✅ **GPU Acceleration**: Optimized for Google Colab with CUDA support
- ✅ **Vietnamese OCR**: CRAFT text detection + VietOCR recognition
- ✅ **Image Preprocessing**: Deskewing and contrast enhancement
- ✅ **Flexible Output**: JSON intermediate format + Word export

## Quick Start

### Option 1: Command Line (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Process single file (auto-detect mode)
python main.py --input ../input/sample.pdf --output ./output

# Batch process all PDFs in input folder
python main.py --input ../input --batch --output ./output

# Force specific mode
python main.py --input ../input/scanned.pdf --mode scan
```

### Option 2: Jupyter Notebook (Recommended for Colab)

1. Open `main.ipynb` in Google Colab
2. Run cells sequentially
3. Upload your PDF files when prompted
4. Download converted Word files

## Usage Examples

### Single File Processing

```python
from modules.pipeline import OCRPipeline

# Initialize pipeline
pipeline = OCRPipeline(
    output_dir="./output",
    temp_dir="./temp",
    dpi=300,
    enable_preprocessing=True,
    auto_detect=True
)

# Process PDF (auto-detect mode)
output_path = pipeline.process_pdf("input/document.pdf")
print(f"Saved to: {output_path}")
```

### Batch Processing

```python
# Process all PDFs in directory
results = pipeline.process_batch(
    input_dir="./input",
    pattern="*.pdf",
    mode=None  # Auto-detect
)

# Print summary
for r in results:
    print(f"{r['status']}: {r['input']} -> {r['output']}")
```

### Force Specific Mode

```python
# Force digital conversion (fast)
pipeline.process_pdf("input/digital.pdf", mode="digital")

# Force OCR processing (slow but thorough)
pipeline.process_pdf("input/scanned.pdf", mode="scan")
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Input PDF file or directory | Required |
| `--output` | Output directory | `./output` |
| `--mode` | Processing mode: `auto`, `scan`, `digital` | `auto` |
| `--batch` | Process all PDFs in input directory | `False` |
| `--no-preprocess` | Disable image preprocessing | `False` |
| `--dpi` | Resolution for PDF to image conversion | `300` |

## Processing Modes

### Auto Mode (Recommended)
Automatically detects whether PDF is digital or scanned and uses appropriate method.

```bash
python main.py --input file.pdf --mode auto
```

### Digital Mode
Direct conversion using pdf2docx. Fast but only works for digital PDFs.

```bash
python main.py --input file.pdf --mode digital
```

### Scan Mode
OCR processing using CRAFT + VietOCR. Slower but works for scanned documents.

```bash
python main.py --input file.pdf --mode scan
```

## Directory Structure

```
ocr/
├── main.py                 # Command-line interface
├── main.ipynb             # Jupyter notebook for Colab
├── requirements.txt       # Python dependencies
├── modules/
│   ├── pipeline.py        # Main pipeline orchestrator
│   ├── pdf_converter.py   # PDF to image conversion
│   ├── digital_parser.py  # Digital PDF conversion
│   ├── ocr_engine.py      # CRAFT + VietOCR
│   ├── preprocessor.py    # Image preprocessing
│   └── exporter.py        # Word export
├── input/                 # Place PDF files here
├── output/                # Converted Word files
└── temp/                  # Temporary files (auto-cleaned)
```

## Google Colab Setup

1. Open `main.ipynb` in Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run setup cells to install dependencies
4. Upload PDFs when prompted
5. Download results

## Performance

| Mode | Speed (per page) | Quality |
|------|-----------------|---------|
| Digital | 1-2 seconds | Perfect |
| Scan (GPU) | 10-20 seconds | Very Good |
| Scan (CPU) | 30-60 seconds | Very Good |

## Troubleshooting

### OCR fails or produces gibberish
- Try disabling preprocessing: `--no-preprocess`
- Increase DPI for low-quality scans: `--dpi 600`

### Out of memory errors
- Process files one at a time (don't use `--batch`)
- Reduce DPI: `--dpi 150`

### Slow processing
- Check GPU availability: `torch.cuda.is_available()`
- Use digital mode for digital PDFs: `--mode digital`

### Module import errors
```bash
pip install -r requirements.txt --upgrade
```

## Dependencies

See `requirements.txt` for full list. Key dependencies:

- `torch` - Deep learning framework
- `vietocr` - Vietnamese text recognition
- `craft-text-detector` - Text detection
- `pdf2docx` - Digital PDF conversion
- `python-docx` - Word file generation
- `PyMuPDF` - PDF manipulation
- `opencv-python` - Image processing

## Output Format

### JSON Intermediate Format
```json
{
  "pages": [
    {
      "page_num": 1,
      "results": [
        {
          "id": 1,
          "text": "Extracted text",
          "box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
          "bbox": [x_min, y_min, x_max, y_max],
          "confidence": 0.95,
          "page_id": 1
        }
      ]
    }
  ]
}
```

### Word Output
- Preserves text layout and spacing
- Page breaks between PDF pages
- Maintains reading order (top to bottom, left to right)

## License

See main repository LICENSE file.

## Contributors

23CLCT2 Team - Traditional Medicine Chatbot Project
