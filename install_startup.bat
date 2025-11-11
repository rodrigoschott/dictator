@echo off
REM Install Dictator to run at Windows startup using Task Scheduler
REM This avoids password issues and works perfectly with system tray
setlocal enabledelayedexpansion

echo ========================================
echo    Dictator Startup Installer
echo ========================================
echo.

REM Check for admin rights
net session >nul 2>&1
if !errorLevel! neq 0 (
    echo ERROR: This script requires administrator privileges!
    echo Right-click and select "Run as Administrator"
    pause
    exit /b 1
)

echo Checking requirements...

REM Check if Python is installed
python --version >nul 2>&1
if !errorLevel! neq 0 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)
echo [OK] Python found

REM Check if Poetry is installed
poetry --version >nul 2>&1
if !errorLevel! neq 0 (
    echo ERROR: Poetry not found!
    pause
    exit /b 1
)
echo [OK] Poetry found

REM Get current directory
set "CURRENT_DIR=%~dp0"
cd /d "%CURRENT_DIR%"

echo.
echo Installing dependencies with Poetry...
poetry install

echo.
echo Detecting Poetry virtualenv location...

REM Get Poetry virtualenv path
for /f "delims=" %%i in ('poetry env info --path 2^>nul') do set "VENV_PATH=%%i"

if "!VENV_PATH!"=="" (
    echo ERROR: Could not detect Poetry virtualenv!
    pause
    exit /b 1
)

if not exist "!VENV_PATH!\Scripts\pythonw.exe" (
    echo ERROR: Could not find pythonw.exe in virtualenv!
    pause
    exit /b 1
)

echo [OK] Found virtualenv: !VENV_PATH!

echo.
echo Creating startup task...

REM Create logs directory
if not exist "%CURRENT_DIR%logs" mkdir "%CURRENT_DIR%logs"

REM Remove old task if exists
schtasks /Delete /TN "Dictator" /F >nul 2>&1

REM Create scheduled task to run at startup
schtasks /Create /TN "Dictator" /TR "\"!VENV_PATH!\Scripts\pythonw.exe\" \"%CURRENT_DIR%src\dictator\tray.py\" \"%CURRENT_DIR%config.yaml\"" /SC ONLOGON /RL HIGHEST /F

if !errorLevel! neq 0 (
    echo ERROR: Failed to create startup task!
    pause
    exit /b 1
)

echo [OK] Startup task created!

echo.
echo ========================================
echo    Installation Complete!
echo ========================================
echo.
echo Task Name: Dictator
echo Trigger: At Windows Logon
echo Run as: %USERNAME% (highest privileges)
echo Trigger: Mouse Side Button
echo.
echo The service will start automatically when you log in.
echo.

choice /C YN /M "Start Dictator now?"
if !errorLevel!==1 (
    echo.
    echo Starting Dictator...
    start "" "!VENV_PATH!\Scripts\pythonw.exe" "%CURRENT_DIR%src\dictator\tray.py" "%CURRENT_DIR%config.yaml"
    echo.
    echo [OK] Dictator started!
    echo Check system tray for microphone icon.
    echo.
)

echo.
echo Useful commands:
echo   - Start manually: schtasks /Run /TN "Dictator"
echo   - Disable autostart: schtasks /Change /TN "Dictator" /DISABLE
echo   - Enable autostart: schtasks /Change /TN "Dictator" /ENABLE
echo   - Remove: schtasks /Delete /TN "Dictator" /F
echo.
pause
