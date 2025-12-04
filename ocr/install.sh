#!/bin/bash
# Installation script for OCR Pipeline (Linux/Mac)
# Run this to set up the environment

echo "========================================"
echo "OCR Pipeline Installation"
echo "========================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    echo "Please install Python 3.8-3.11"
    exit 1
fi

echo "Step 1: Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "Step 2: Activating virtual environment..."
source venv/bin/activate

echo "Step 3: Upgrading pip..."
python -m pip install --upgrade pip

echo "Step 4: Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    echo "Try installing manually: pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "Step 5: Verifying installation..."
python check_compatibility.py

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "To activate the environment in the future:"
echo "    source venv/bin/activate"
echo ""
echo "To run OCR:"
echo "    python main.py --mode scan --input your_image.png"
echo ""
