# OCR Pipeline - Quick Reference

## Installation

### Windows
```bash
install.bat
```

### Linux/Mac
```bash
chmod +x install.sh
./install.sh
```

### Manual Installation
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify
python check_compatibility.py
```

## Usage

### Basic OCR (Scan Mode)
```bash
python main.py --mode scan --input image.png
```

### With Custom Output
```bash
python main.py --mode scan --input image.png --output result.docx
```

### Digital PDF to DOCX
```bash
python main.py --mode digital --input document.pdf --output output.docx
```

### With Preprocessing
```bash
python main.py --mode scan --input image.png --preprocess
```

## Command Line Arguments

| Argument | Required | Values | Description |
|----------|----------|--------|-------------|
| `--mode` | Yes | `scan`, `digital`, `math` | Processing mode |
| `--input` | Yes | file path | Input PDF or image file |
| `--output` | No | file path | Output file path (auto-generated if omitted) |
| `--preprocess` | No | flag | Enable image preprocessing |

## Output Files

### Scan Mode Outputs
1. `{filename}_ocr_results.json` - Raw OCR results with bounding boxes
2. `{filename}_output.docx` - Formatted Word document

### Digital Mode Output
1. `{filename}.docx` - Converted Word document

## File Structure

```
ocr/
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── check_compatibility.py       # Library checker
├── install.bat / install.sh     # Installation scripts
├── modules/
│   ├── ocr_engine.py           # Enhanced OCR (CRAFT + VietOCR)
│   ├── digital_parser.py       # Digital PDF parser
│   ├── preprocessor.py         # Image preprocessing
│   └── exporter.py             # DOCX export
└── output/                      # Output directory
```

## Supported Formats

### Input
- **Images**: PNG, JPG, JPEG, BMP, TIFF
- **PDF**: Digital PDFs (digital mode) or Scanned PDFs converted to images

### Output
- **JSON**: OCR results with coordinates
- **DOCX**: Microsoft Word document

## Pipeline Modes

### 1. Scan Mode (Default: Enhanced OCR)
- Uses CRAFT text detection
- Uses VietOCR for Vietnamese text recognition
- Best for scanned documents and images with Vietnamese text
- Outputs: JSON + DOCX

### 2. Digital Mode
- Uses pdf2docx conversion
- Best for digital PDFs with selectable text
- Preserves formatting, images, tables
- Output: DOCX

### 3. Math Mode (Future)
- LaTeX formula extraction
- Currently commented out

## Troubleshooting

### "Input file does not exist"
- Check file path is correct
- Use absolute or relative path
- Ensure file extension is correct

### "Missing required libraries"
- Run: `pip install -r requirements.txt`
- Or run installation script

### NumPy warnings on Python 3.14
- Downgrade to Python 3.11: `python3.11 -m venv venv`
- Or install specific NumPy: `pip install numpy==1.26.4`

### Slow processing
- Enable GPU: Install CUDA-enabled PyTorch
- Reduce image size
- Use lower resolution input

### VietOCR model download fails
- Check internet connection
- Retry the operation
- Models download on first run (~100MB)

## Performance

| Hardware | Speed (per page) | Recommendation |
|----------|------------------|----------------|
| CPU only | 10-30 seconds | Use for small batches |
| GPU (CUDA) | 1-3 seconds | Recommended for production |
| High-res images | Slower | Resize to 1500-2000px width |

## Tips

1. **First run**: VietOCR downloads models (~100MB)
2. **GPU**: 5-10x faster with CUDA-enabled PyTorch
3. **Batch**: Process multiple files with scripts
4. **Quality**: Higher resolution = better accuracy (but slower)
5. **Vietnamese**: This pipeline is optimized for Vietnamese text

## Support

- Compatibility issues: See `LIBRARY_COMPATIBILITY.md`
- Detailed guide: See `PIPELINE_SUMMARY.md`
- Check libraries: `python check_compatibility.py`

## Version Info

- **Python**: 3.8 - 3.11 recommended
- **OCR Engine**: CRAFT + VietOCR (default)
- **Output**: JSON + DOCX
