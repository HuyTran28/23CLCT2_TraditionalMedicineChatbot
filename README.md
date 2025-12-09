# Document OCR & Conversion Pipeline

This repository provides a modern pipeline for converting PDF documents to Word format using state-of-the-art OCR techniques. It supports both scanned and digital PDFs, leveraging marker-pdf for Vietnamese text recognition with superior accuracy.

## Features
- Automatic detection of PDF type (scanned vs. digital)
- OCR for scanned documents using **marker-pdf** (modern, Vietnamese-optimized)
- Direct conversion for digital PDFs
- Automatic extraction of images and formulas to separate directory
- Image placeholders in output documents
- Preserves tables and maintains reading order
- Treats formulas as images (no complex math rendering)
- Outputs clean markdown and Word documents

## Prerequisites
- **Operating System:** Windows (recommended)
- **Python Version:** Python 3.10+ (64-bit, via Conda)
- **Hardware:** CPU (GPU auto-detected if available for faster processing)
- **Conda:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)

## Setup Instructions

### 1. Install Conda
Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) for Windows.

### 2. Create a Conda Environment (Recommended)
Using a Conda environment with Python 3.10+ ensures better compatibility with modern libraries.

Open PowerShell and run:

```powershell
# Navigate to the project directory
cd "C:\Users\Admin\Documents\23CLCT2_TraditionalMedicineChatbot\ocr"

# Create a new conda environment named 'ocr-env' with Python 3.10
conda create -n ocr-env python=3.10 -y

# Activate the environment
conda activate ocr-env
```

### 3. Install Required Packages
With the conda environment activated, install all dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** marker-pdf will automatically download required models on first use. This may take a few minutes.

### 4. Configuration (Optional)
You can customize input/output directories and other settings in `config.py`.

Example `config.py`:
```python
INPUT_DIR = "./input"
OUTPUT_DIR = "./output"
EXTRACT_IMAGES = True  # Extract images/formulas to temp/extracted_images/
EXTRACT_TABLES = True  # Preserve table structure
```

### 5. Running the Pipeline
Run the pipeline from the `ocr` directory with the conda environment activated:

```powershell
python main.py --input "path/to/your.pdf"
```

#### Common Arguments
- `--input`: Path to a single PDF file (required unless set in `config.py`)
- `--output`: Output directory (default: `./output`)
- `--mode`: `auto` (default), `scan`, or `digital`
- `--batch`: Process all PDFs in input directory

**Example:**
```powershell
python main.py --input "input/sample.pdf" --output "output" --mode auto
```

**Batch Processing:**
```powershell
python main.py --input "input" --batch
```

### 6. Output Structure
After processing, you'll find:
- **Word document** (`.docx`): In the output directory
- **Markdown file** (`.md`): In the output directory (intermediate format)
- **Extracted images**: In `temp/extracted_images/` directory
- **Image placeholders**: In the Word document showing where images were located

### 7. Troubleshooting
- If you see missing library errors, run:
  ```powershell
  pip install -r requirements.txt
  ```
- Ensure your conda environment is activated before running the script.
- For GPU acceleration, marker-pdf will automatically detect and use CUDA if available.
- **First run may be slow** as marker-pdf downloads models (~1-2GB).

## Key Improvements
✅ **Better Vietnamese Support**: marker-pdf is optimized for Vietnamese text  
✅ **Formula Handling**: Automatically detects and saves formulas as images  
✅ **Cleaner Output**: Maintains reading order without complex layout analysis  
✅ **Table Preservation**: Keeps table structure intact  
✅ **Image Management**: Extracts images to separate directory with placeholders  
✅ **No Heavy Dependencies**: Removed PyTorch, PaddleOCR, viet-ocr bloat  

## Acknowledgements
- [marker-pdf](https://github.com/VikParuchuri/marker) - Modern PDF to Markdown converter
- [python-docx](https://python-docx.readthedocs.io/) - Word document generation

---