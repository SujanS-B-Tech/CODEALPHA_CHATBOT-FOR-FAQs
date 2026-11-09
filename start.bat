@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title SmartFAQ - Intelligent FAQ Assistant
color 0B

echo ======================================================================
echo           SmartFAQ - Intelligent FAQ Assistant (NLP Engine)
echo ======================================================================
echo.

:: 1. Detect Python or Py launcher
set "PY_CMD=python"
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=py"
    ) else (
        echo [ERROR] Python is not detected in your system PATH.
        echo.
        echo Please ensure Python 3.8+ is installed from https://www.python.org/
        echo and that "Add Python to PATH" was checked during setup.
        echo.
        pause
        exit /b 1
    )
)

echo [INFO] Using Python interpreter: %PY_CMD%
%PY_CMD% --version
echo.

:: 2. Check or create virtual environment
if exist "venv\Scripts\python.exe" goto :CHECK_DEPS

echo [INFO] Virtual environment not found. Creating 'venv'...
%PY_CMD% -m venv venv
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to create virtual environment.
    echo.
    pause
    exit /b 1
)
echo [INFO] Virtual environment created successfully.
echo.

:CHECK_DEPS
:: 3. Check or install dependencies
if exist "venv\Scripts\streamlit.exe" goto :LAUNCH_APP

echo [INFO] Installing required dependencies (streamlit, pandas, scikit-learn, pytest)...
echo This may take a couple of minutes on first run. Please wait...
echo.

"venv\Scripts\python.exe" -m pip install --upgrade pip
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install required packages from requirements.txt.
    echo.
    pause
    exit /b 1
)
echo [INFO] Dependencies installed successfully!
echo.

:LAUNCH_APP
:: 4. Launch Streamlit Application
echo ======================================================================
echo [INFO] Starting SmartFAQ Web Application...
echo [INFO] URL: http://localhost:8501
echo [INFO] Press Ctrl+C in this terminal window to stop the chatbot.
echo ======================================================================
echo.

"venv\Scripts\streamlit.exe" run app.py

if errorlevel 1 (
    echo.
    echo [WARNING] Streamlit exited with an error code: %errorlevel%
)

echo.
echo Application stopped.
pause
