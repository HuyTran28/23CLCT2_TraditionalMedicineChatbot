# Library Compatibility Guide

## Overview
This OCR pipeline uses **Enhanced Vietnamese OCR** by default, combining CRAFT text detection with VietOCR recognition.

## Required Libraries

### Core Dependencies
- **PyTorch** (`torch`) - Deep learning framework
- **OpenCV** (`opencv-python`) - Image processing
- **NumPy** (`<2.0.0`) - Numerical operations
- **Pillow** - Image handling
- **CRAFT Text Detector** (`craft-text-detector`) - Text region detection
- **VietOCR** (`vietocr`) - Vietnamese text recognition

### Additional Dependencies
- **python-docx** - DOCX file generation
- **PyMuPDF** (`fitz`) - PDF handling (digital mode)
- **pdf2docx** - PDF to DOCX conversion (digital mode)
- **matplotlib** - Visualization (optional)

## Python Version Compatibility

### Recommended: Python 3.8 - 3.11
- All libraries are fully tested and stable
- NumPy works without warnings

### Python 3.12+
- **Warning**: Some libraries may have compatibility issues
- NumPy 2.x has breaking changes with older packages
- **Solution**: Pin NumPy to `<2.0.0` in requirements.txt

### Python 3.14
- **Not Recommended**: Experimental NumPy builds may cause runtime warnings
- Use Python 3.11 or earlier for production

## Known Compatibility Issues

### 1. NumPy Version Conflicts
**Problem**: VietOCR and CRAFT may require NumPy 1.x, but NumPy 2.x is incompatible

**Solution**:
```bash
pip install "numpy<2.0.0"
```

### 2. PyTorch + CUDA
**Problem**: CUDA support requires specific PyTorch versions matching your CUDA version

**Solution**:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision
```

### 3. CRAFT Text Detector
**Problem**: May fail on newer Python versions (3.12+)

**Solution**: Use Python 3.11 or earlier

### 4. VietOCR Model Download
**Problem**: First run downloads models (~100MB), may timeout

**Solution**: Ensure stable internet connection on first run

## Installation Steps

### 1. Create Virtual Environment (Recommended)
```bash
# Python 3.11 recommended
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python check_compatibility.py
```

## Testing Library Compatibility

Run the compatibility checker:
```bash
python check_compatibility.py
```

Expected output:
```
✓ PyTorch: 2.x.x
✓ OpenCV: 4.x.x
✓ NumPy: 1.x.x
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

### Import Errors
```python
ImportError: No module named 'craft_text_detector'
```
**Solution**: `pip install craft-text-detector`

### CUDA Not Available
```
CUDA not available - will use CPU (slower)
```
**Solution**: This is a warning, not an error. OCR will work but slower.
To enable GPU: Install CUDA-enabled PyTorch

### NumPy Warnings on Python 3.14
```
Warning: Numpy built with MINGW-W64 on Windows 64 bits is experimental
```
**Solution**: Downgrade to Python 3.11 or use NumPy 1.26.x:
```bash
pip install numpy==1.26.4
```

### VietOCR Model Download Fails
```
Error downloading model
```
**Solution**: 
1. Check internet connection
2. Manually download from VietOCR GitHub
3. Use a VPN if in restricted regions

## Performance Optimization

### GPU Acceleration
- Install CUDA-enabled PyTorch for 5-10x speedup
- Requires NVIDIA GPU with CUDA support

### CPU Optimization
- Use multi-threading for batch processing
- Reduce image resolution for faster processing
- Disable visualization (`visualize=False`)

## Recommended Setup

```bash
# Use Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies with version pinning
pip install torch torchvision
pip install "numpy<2.0.0"
pip install -r requirements.txt

# Verify
python check_compatibility.py
```

## Quick Start Test

```bash
# Test with a sample image
python main.py --mode scan --input sample.png --output result.docx
```

This should:
1. Detect text regions with CRAFT
2. Recognize Vietnamese text with VietOCR
3. Export to both JSON and DOCX formats
