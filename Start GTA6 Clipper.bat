@echo off
setlocal
cd /d "%~dp0"
title GTA 6 Clip Engine
echo ================================
echo   GTA 6 Clip Engine
echo ================================

python -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>nul
if errorlevel 1 (
  echo.
  echo Python 3.11 or newer is required and was not found.
  echo Install it from https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^), then double-click this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo First-time setup: installing everything the app needs. This takes a few minutes, once.
  python -m venv .venv
  if errorlevel 1 goto :fail
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -e .
  if errorlevel 1 goto :fail
  where nvidia-smi >nul 2>nul
  if not errorlevel 1 (
    echo NVIDIA graphics card found: adding GPU speech libraries for faster transcription...
    ".venv\Scripts\python.exe" -m pip install -e ".[gpu]"
  )
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo.
  echo FFmpeg is missing. Installing it now...
  winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
  where ffmpeg >nul 2>nul
  if errorlevel 1 (
    echo.
    echo FFmpeg was installed but this window cannot see it yet.
    echo Close this window and double-click "Start GTA6 Clipper.bat" again.
    pause
    exit /b 0
  )
)

echo.
echo Starting the app...
start "" ".venv\Scripts\pythonw.exe" -m engine.gui
exit /b 0

:fail
echo.
echo Setup did not finish. Read the messages above ^(usually a network problem^), then run this file again.
pause
exit /b 1
