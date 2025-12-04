@echo off
REM Installation script for OCR Pipeline
REM Run this to set up the environment

echo ========================================
echo OCR Pipeline Installation
echo ========================================
echo.

REM Check Python version
python --version 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8-3.11 from https://www.python.org/
    pause
    exit /b 1
)

echo Step 1: Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo Step 4: Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    echo Try installing manually: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo Step 5: Verifying installation...
python check_compatibility.py

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To activate the environment in the future:
echo    venv\Scripts\activate
echo.
echo To run OCR:
echo    python main.py --mode scan --input your_image.png
echo.
pause
