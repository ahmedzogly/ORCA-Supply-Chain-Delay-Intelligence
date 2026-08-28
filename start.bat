@echo off
setlocal enabledelayedexpansion
title ORCA - Supply Chain Delay Intelligence Launcher
color 0B

echo.
echo  ======================================================================
echo   ORCA: Operational Risk ^& Cost Analytics
echo   Pharmaceutical Supply Chain Delay Intelligence ^& Digital Twin
echo  ======================================================================
echo.

cd /d "%~dp0"

:: 1. Check Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    color 0C
    echo [ERROR] Python is not found in your system PATH!
    echo Please install Python 3.10+ and make sure to check "Add Python to PATH".
    pause
    exit /b 1
)

:: 2. Check Virtual Environment (.venv)
if exist ".venv\Scripts\python.exe" (
    echo [OK] Virtual environment found. Activating .venv ...
    call .venv\Scripts\activate
) else if exist "venv\Scripts\python.exe" (
    echo [OK] Virtual environment found. Activating venv ...
    call venv\Scripts\activate
) else (
    echo [*] Creating virtual environment (.venv) ...
    python -m venv .venv
    call .venv\Scripts\activate
    echo [*] Installing dependencies (this may take a few moments) ...
    python -m pip install --upgrade pip
    pip install -e .
    pip install -r requirements.txt
)

:: 3. Launch the Streamlit Control Tower
echo.
echo [*] Launching ORCA Streamlit Control Tower ...
echo [*] Server is starting at http://localhost:8501
echo [*] Press Ctrl+C in this window anytime to stop the server.
echo.

streamlit run streamlit_app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [*] Falling back to direct app launcher ...
    streamlit run src/delay_intelligence/dashboard/app.py
)

pause
