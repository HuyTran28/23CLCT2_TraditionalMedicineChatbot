# OCR Pipeline Configuration
# Copy / edit values for your environment

import logging

# Directory Configuration (relative paths)
INPUT_DIR = "../input"           # Input folder containing PDFs
OUTPUT_DIR = "./ocr/output"          # Output folder for Word files

# Processing Configuration
AUTO_DETECT = True               # Auto-detect digital vs scanned PDFs
ENABLE_PREPROCESSING = True      # Enable image enhancement for scanned docs
DPI = 300                        # Resolution for PDF to image conversion (150-600)

# OCR Configuration
# Set to 'cpu' for local Windows environment; change to 'cuda' on Colab with GPU
OCR_DEVICE = "cpu"
CRAFT_TEXT_THRESHOLD = 0.7       # CRAFT text detection threshold (0.0-1.0)
CRAFT_LINK_THRESHOLD = 0.8       # CRAFT link detection threshold (0.0-1.0)
Y_THRESHOLD = 10                 # Line grouping threshold (pixels)

# Preprocessing Configuration
MAX_DESKEW_ANGLE = 15.0          # Maximum angle for deskewing (degrees)
CLIP_LIMIT = 3.0                 # CLAHE clip limit for contrast enhancement
TILE_GRID_SIZE = 8               # CLAHE tile grid size
SHARPEN_STRENGTH = 1.5           # Sharpening strength

# Export Configuration
BASE_SPACING = 1.0               # Base line spacing in Word output
FONT_SIZE = 12                   # Font size in Word output (points)
SAVE_JSON = True                 # Save intermediate JSON results

# Logging Configuration
LOG_LEVEL = logging.INFO         # Logging level
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Batch Processing
BATCH_PATTERN = "*.pdf"          # File pattern for batch processing
CONTINUE_ON_ERROR = True         # Continue batch processing if one file fails

# Performance
CLEANUP_TEMP = True              # Automatically cleanup temp files after processing
PARALLEL_PAGES = False           # Process pages in parallel (experimental)