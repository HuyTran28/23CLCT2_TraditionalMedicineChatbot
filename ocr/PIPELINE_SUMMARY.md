# OCR Pipeline - Enhanced Vietnamese OCR (Default)

## Summary of Changes

The OCR pipeline now **always uses Enhanced Vietnamese OCR** by default, combining:
- **CRAFT** (Character Region Awareness for Text detection)
- **VietOCR** (Vietnamese Optical Character Recognition)

## Key Updates

### 1. Removed `--enhanced-ocr` Flag
- Enhanced OCR is now the **default and only mode** for scan processing
- No need to specify additional flags
- Simplified command-line interface

### 2. Library Compatibility
- Added NumPy version constraint: `numpy<2.0.0`
- Prevents compatibility issues with Python 3.12+
- Recommended Python version: **3.8 - 3.11**

### 3. Dependency Checking
- Automatic library verification on startup
- Clear error messages for missing dependencies
- Graceful handling of import errors

### 4. Enhanced Error Handling
- Better exception messages
- Exit codes for automation
- Visual feedback with ✓ and ✗ symbols

## Usage

### Scan Mode (Enhanced OCR - Default)
```bash
python main.py --mode scan --input image.png --output result.docx
```

### Digital PDF Mode
```bash
python main.py --mode digital --input document.pdf --output result.docx
```

### With Preprocessing
```bash
python main.py --mode scan --input image.png --preprocess --output result.docx
```

## Library Compatibility Matrix

| Library | Version | Python 3.8-3.11 | Python 3.12+ | Python 3.14 |
|---------|---------|-----------------|--------------|-------------|
| torch | 2.x | ✅ Stable | ✅ Stable | ⚠️ Check compatibility |
| numpy | <2.0.0 | ✅ Stable | ✅ Stable | ⚠️ Experimental warnings |
| opencv-python | 4.x | ✅ Stable | ✅ Stable | ✅ Stable |
| craft-text-detector | latest | ✅ Stable | ⚠️ May have issues | ❌ Not tested |
| vietocr | latest | ✅ Stable | ✅ Stable | ⚠️ NumPy warnings |
| python-docx | latest | ✅ Stable | ✅ Stable | ✅ Stable |

### Legend
- ✅ Fully compatible, no issues
- ⚠️ Works but may show warnings
- ❌ Known compatibility issues

## Verification

### Check Library Compatibility
```bash
python check_compatibility.py
```

### Expected Output
```
✓ PyTorch: 2.x.x
  CUDA available: NVIDIA GeForce RTX 3080
✓ OpenCV: 4.x.x
✓ NumPy: 1.26.x
✓ Pillow: 10.x.x
✓ CRAFT Text Detector: installed
✓ VietOCR: installed
✓ python-docx: installed
✓ PyMuPDF: x.x.x
✓ pdf2docx: installed
✓ Matplotlib: 3.x.x

✅ All critical libraries are installed and compatible!
```

## Troubleshooting

### NumPy Warnings on Python 3.14
If you see warnings like:
```
Warning: Numpy built with MINGW-W64 on Windows 64 bits is experimental
```

**Solutions:**
1. Use Python 3.11 (recommended)
2. Or install specific NumPy version: `pip install numpy==1.26.4`

### Missing Libraries
If you get import errors:
```bash
pip install -r requirements.txt
```

### CUDA Not Available
OCR will work on CPU but slower. To enable GPU:
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Performance Tips

1. **Use GPU**: 5-10x faster with CUDA-enabled PyTorch
2. **Batch Processing**: Process multiple images together
3. **Image Resolution**: Resize very large images for faster processing
4. **Disable Visualization**: Set `visualize=False` in production

## Files Created/Modified

### Modified
- `main.py` - Simplified to always use enhanced OCR
- `ocr_engine.py` - Better error handling and documentation
- `requirements.txt` - Added NumPy version constraint
- `exporter.py` - Already compatible with OCR results

### Created
- `check_compatibility.py` - Library compatibility checker
- `LIBRARY_COMPATIBILITY.md` - Detailed compatibility guide
- `PIPELINE_SUMMARY.md` - This file

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Verify compatibility**: `python check_compatibility.py`
3. **Test with sample**: `python main.py --mode scan --input sample.png`
4. **Review output**: Check JSON and DOCX files

## Architecture

```
Input Image/PDF
    ↓
[CRAFT Text Detector]
    ↓
Text Regions (bounding boxes)
    ↓
[VietOCR Recognition]
    ↓
Vietnamese Text
    ↓
[WordExporter]
    ↓
JSON + DOCX Output
```

## Support

For library compatibility issues:
- Check `LIBRARY_COMPATIBILITY.md`
- Run `python check_compatibility.py`
- Ensure Python 3.8-3.11 for best compatibility
