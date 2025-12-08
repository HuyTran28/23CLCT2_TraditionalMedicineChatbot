# Document OCR & Conversion Pipeline

This repository provides a pipeline for converting PDF documents to Word format using advanced OCR techniques. It supports both scanned and digital PDFs, leveraging deep learning models for Vietnamese text recognition.

## Features
- Automatic detection of PDF type (scanned vs. digital)
- OCR for scanned documents using PaddleOCR and VietOCR
- Direct conversion for digital PDFs
- Configurable preprocessing and DPI settings
- Single-file processing (batch mode disabled)

## Prerequisites
- **Operating System:** Windows (recommended)
- **Python Version:** Python 3.10 (64-bit, via Conda)
- **Hardware:** CPU (GPU recommended for faster OCR)
- **Conda:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)

## Setup Instructions

### 1. Install Conda
Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) for Windows.

### 2. Create a Conda Environment (Recommended)
Using a Conda environment with Python 3.10 ensures better compatibility with deep learning libraries.

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

> If you encounter errors related to `torch` or `paddlepaddle`, ensure you are using a compatible Python version and have a supported CPU/GPU.

### 4. Configuration (Optional)
You can customize input/output directories and other settings in `config.py`.

Example `config.py`:
```python
INPUT_DIR = "./input"
OUTPUT_DIR = "./output"
DPI = 300
ENABLE_PREPROCESSING = True
AUTO_DETECT = True
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
- `--dpi`: DPI for image conversion (default: 300)
- `--no-preprocess`: Disable preprocessing
- `--workers`: Number of worker threads (default: CPU count)

**Example:**
```powershell
python main.py --input "input/sample.pdf" --output "output" --mode auto --dpi 300
```

> **Batch processing is disabled.** To process multiple PDFs, run the script separately for each file.

### 6. Troubleshooting
- If you see missing library errors, run:
  ```powershell
  pip install -r requirements.txt
  ```
- Ensure your conda environment is activated before running the script.
- For GPU acceleration, install the appropriate version of `torch` and `paddlepaddle` for your hardware.

## Acknowledgements
- [VietOCR](https://github.com/quantra/VietOCR)
- [paddleocr](https://github.com/PaddlePaddle/PaddleOCR)

---