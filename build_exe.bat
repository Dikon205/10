@echo off
setlocal
title Fluid Simulator Executable Builder
cd /d "%~dp0"

echo ======================================================
echo       Building Standalone FluidLab Executable
echo ======================================================
echo.

if not exist ".venv\Scripts\pyinstaller.exe" (
    echo [INFO] Installing PyInstaller...
    .\.venv\Scripts\python.exe -m pip install pyinstaller
)

echo [INFO] Compiling into standalone application (folder mode)...
echo [INFO] Bundling Python, SDL2/pygame-ce, Numba JIT, NumPy, and SciPy...
.\.venv\Scripts\pyinstaller.exe --noconfirm --onedir --windowed --name "FluidLab" ^
    --collect-all numba ^
    --collect-all llvmlite ^
    --collect-all scipy ^
    fluid_dynamics_simulation.py

if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    pause
    exit /b %errorlevel%
)

echo.
echo ======================================================
echo [SUCCESS] Build complete!
echo Standalone folder: %~dp0dist\FluidLab\
echo Executable file:   %~dp0dist\FluidLab\FluidLab.exe
echo.
echo You can zip the "FluidLab" folder and send it to anyone!
echo They do NOT need Python or VS Code to run it.
echo ======================================================
echo.
pause
