# OCR Pipeline Configuration
# Copy / edit values for your environment

import logging
from pathlib import Path

_OCR_DIR = Path(__file__).resolve().parent

# Directory Configuration (resolved relative to this file)
INPUT_DIR = str(_OCR_DIR / "input")
OUTPUT_DIR = str(_OCR_DIR / "output")

# Processing Configuration
AUTO_DETECT = True               # Auto-detect digital vs scanned PDFs
ENABLE_PREPROCESSING = False     # Disabled - marker-pdf handles preprocessing internally
DPI = 300                        # Resolution for PDF to image conversion (150-600)

# OCR Configuration
# marker-pdf automatically detects and uses GPU if available, otherwise uses CPU
MARKER_EXTRACT_IMAGES = True    # Extract images and formulas to separate directory
MARKER_OUTPUT_FORMAT = "markdown"  # Output format: markdown

# Export Configuration
BASE_SPACING = 1.0               # Base line spacing in Word output
FONT_SIZE = 12                   # Font size in Word output (points)
SAVE_JSON = False                # Save intermediate JSON results (markdown is saved instead)

# Logging Configuration
LOG_LEVEL = logging.INFO         # Logging level
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Batch Processing
BATCH_PATTERN = "*.pdf"          # File pattern for batch processing
CONTINUE_ON_ERROR = True         # Continue batch processing if one file fails

# Performance
CLEANUP_TEMP = True              # Automatically cleanup temp files after processing
PARALLEL_PAGES = False           # marker-pdf handles parallelization internally
