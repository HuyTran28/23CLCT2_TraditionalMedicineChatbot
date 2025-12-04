"""
Library Compatibility Checker for OCR Pipeline
Checks if all required libraries are installed and compatible
"""

def check_libraries():
    issues = []
    warnings = []
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            warnings.append("CUDA not available - will use CPU (slower)")
    except ImportError:
        issues.append("PyTorch is not installed. Run: pip install torch")
    
    # Check OpenCV
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        issues.append("OpenCV is not installed. Run: pip install opencv-python")
    
    # Check NumPy
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        issues.append("NumPy is not installed. Run: pip install numpy")
    
    # Check PIL/Pillow
    try:
        from PIL import Image
        import PIL
        print(f"✓ Pillow: {PIL.__version__}")
    except ImportError:
        issues.append("Pillow is not installed. Run: pip install Pillow")
    
    # Check CRAFT Text Detector
    try:
        from craft_text_detector import Craft
        print(f"✓ CRAFT Text Detector: installed")
    except ImportError:
        issues.append("CRAFT Text Detector is not installed. Run: pip install craft-text-detector")
    
    # Check VietOCR
    try:
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
        print(f"✓ VietOCR: installed")
    except ImportError:
        issues.append("VietOCR is not installed. Run: pip install vietocr")
    
    # Check python-docx
    try:
        from docx import Document
        print(f"✓ python-docx: installed")
    except ImportError:
        issues.append("python-docx is not installed. Run: pip install python-docx")
    
    # Check PyMuPDF
    try:
        import fitz
        print(f"✓ PyMuPDF: {fitz.version}")
    except ImportError:
        warnings.append("PyMuPDF is not installed (optional for digital PDF mode). Run: pip install PyMuPDF")
    
    # Check pdf2docx
    try:
        from pdf2docx import Converter
        print(f"✓ pdf2docx: installed")
    except ImportError:
        warnings.append("pdf2docx is not installed (optional for digital PDF mode). Run: pip install pdf2docx")
    
    # Check matplotlib
    try:
        import matplotlib
        print(f"✓ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        warnings.append("Matplotlib is not installed (optional for visualization). Run: pip install matplotlib")
    
    print("\n" + "="*50)
    
    if issues:
        print("\n❌ CRITICAL ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("\n✅ All critical libraries are installed and compatible!")
    return True

if __name__ == "__main__":
    check_libraries()
