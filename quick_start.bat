@echo off
REM Quick start batch file for Windows users
REM This file launches the quick start script

echo.
echo ================================================================
echo 🤖 Handwritten Digit Predictor - Quick Start (Windows)
echo ================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo.
    pause
    exit /b 1
)

REM Run the quick start script
echo 🚀 Starting quick start guide...
python quick_start.py

echo.
echo Press any key to exit...
pause >nul
