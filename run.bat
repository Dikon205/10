@echo off
setlocal enabledelayedexpansion
title Fluid Simulation Launcher
cd /d "%~dp0"

echo ======================================================
echo           Fluid Dynamics Simulator Launcher
echo ======================================================
echo.

:: 1. Check if virtual environment already exists
if exist ".venv\Scripts\python.exe" (
    echo [OK] Virtual environment found.
    goto RUN_APP
)

:: 2. Check if Python is installed on the system
set "PY_CMD="
where py >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=py"
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PY_CMD=python"
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python was not detected on your computer!
    echo.
    echo To run this application:
    echo   1. Install Python 3.10, 3.11, or 3.12 from:
    echo      https://www.python.org/downloads/
    echo      (Make sure to check "Add Python to PATH" during installation)
    echo   OR
    echo   2. Download the standalone .exe directly from GitHub Releases.
    echo.
    pause
    exit /b 1
)

:: 3. Create virtual environment and install requirements
echo [INFO] First-time setup: Creating virtual environment...
%PY_CMD% -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [INFO] Installing required libraries (pygame-ce, numpy, numba, scipy)...
echo [INFO] This might take a couple of minutes on the first run...
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [OK] Setup completed successfully!
echo.

:RUN_APP
echo [INFO] Launching Fluid Dynamics Simulation...
.\.venv\Scripts\python.exe fluid_dynamics_simulation.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
    exit /b %errorlevel%
)
