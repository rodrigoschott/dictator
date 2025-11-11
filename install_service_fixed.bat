@echo off
REM Dictator Service Installer - Fixed Version
setlocal enabledelayedexpansion

echo ========================================
echo    Dictator Service Installer
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
    echo ERROR: Python not found! Please install Python first.
    pause
    exit /b 1
)
echo [OK] Python found

REM Check if Poetry is installed
poetry --version >nul 2>&1
if !errorLevel! neq 0 (
    echo.
    echo Poetry not found! Installing Poetry...
    python -m pip install poetry
    if !errorLevel! neq 0 (
        echo ERROR: Failed to install Poetry
        pause
        exit /b 1
    )
    echo [OK] Poetry installed
) else (
    echo [OK] Poetry found
)

REM Get current directory
set "CURRENT_DIR=%~dp0"
cd /d "%CURRENT_DIR%"

REM Check if NSSM is installed
set "NSSM_PATH="
where nssm >nul 2>&1
if !errorLevel! equ 0 (
    echo [OK] NSSM found
    set "NSSM_PATH=nssm"
) else (
    if exist "%WINDIR%\System32\nssm.exe" (
        echo [OK] NSSM found in System32
        set "NSSM_PATH=%WINDIR%\System32\nssm.exe"
    ) else (
        echo.
        echo NSSM not found! Downloading NSSM...
        echo.

        REM Download NSSM using PowerShell
        powershell -Command "Invoke-WebRequest -Uri 'https://nssm.cc/release/nssm-2.24.zip' -OutFile '%TEMP%\nssm.zip'"

        echo Extracting NSSM...
        powershell -Command "Expand-Archive -Path '%TEMP%\nssm.zip' -DestinationPath '%TEMP%\nssm' -Force"

        REM Copy NSSM to Windows directory
        copy "%TEMP%\nssm\nssm-2.24\win64\nssm.exe" "%WINDIR%\System32\" >nul

        if !errorLevel! neq 0 (
            echo ERROR: Failed to install NSSM
            pause
            exit /b 1
        )

        echo [OK] NSSM installed
        set "NSSM_PATH=%WINDIR%\System32\nssm.exe"
    )
)

echo.
echo Installing dependencies with Poetry...
poetry install

echo.
echo Detecting Poetry virtualenv location...

REM Get Poetry virtualenv path
for /f "delims=" %%i in ('poetry env info --path 2^>nul') do set "VENV_PATH=%%i"

REM Check if VENV_PATH was set
if "!VENV_PATH!"=="" (
    echo ERROR: Could not detect Poetry virtualenv!
    echo Please run: poetry env info --path
    pause
    exit /b 1
)

REM Check if Python exists in virtualenv
if not exist "!VENV_PATH!\Scripts\python.exe" (
    echo ERROR: Could not find Poetry virtualenv Python executable!
    echo Expected: !VENV_PATH!\Scripts\python.exe
    pause
    exit /b 1
)

echo [OK] Found virtualenv: !VENV_PATH!

echo.
echo Installing Dictator Service...

REM Create logs directory if it doesn't exist
if not exist "%CURRENT_DIR%logs" (
    mkdir "%CURRENT_DIR%logs"
    echo [OK] Created logs directory
)

REM Stop service if already exists
"!NSSM_PATH!" stop Dictator >nul 2>&1
"!NSSM_PATH!" remove Dictator confirm >nul 2>&1

REM Install service
"!NSSM_PATH!" install Dictator "!VENV_PATH!\Scripts\pythonw.exe" "%CURRENT_DIR%src\dictator\tray.py" "%CURRENT_DIR%config.yaml"

REM Configure service
"!NSSM_PATH!" set Dictator AppDirectory "%CURRENT_DIR%"
"!NSSM_PATH!" set Dictator DisplayName "Dictator - Voice to Text"
"!NSSM_PATH!" set Dictator Description "Background voice-to-text service with global hotkey support"
"!NSSM_PATH!" set Dictator Start SERVICE_AUTO_START

REM Configure to run as current user (required for system tray and audio)
echo.
echo Configuring service to run as current user: %USERNAME%
echo This is required for system tray icon and audio access.
echo.
"!NSSM_PATH!" set Dictator ObjectName ".\%USERNAME%" ""

REM Set stdout/stderr redirects
"!NSSM_PATH!" set Dictator AppStdout "%CURRENT_DIR%logs\service.log"
"!NSSM_PATH!" set Dictator AppStderr "%CURRENT_DIR%logs\service_error.log"

REM Set environment
"!NSSM_PATH!" set Dictator AppEnvironmentExtra PYTHONUNBUFFERED=1

echo.
echo Service installed successfully!
echo.
echo Service Name: Dictator
echo Trigger: Mouse Side Button (configurable via system tray menu)
echo.

choice /C YN /M "Start service now?"
if !errorLevel!==1 (
    "!NSSM_PATH!" start Dictator
    if !errorLevel!==0 (
        echo.
        echo [OK] Service started!
        echo Check system tray for Dictator icon.
    ) else (
        echo.
        echo ERROR: Failed to start service
        echo Check logs\service_error.log for details
    )
)

echo.
echo Installation complete!
echo.
echo Useful commands:
echo   - Start:   nssm start Dictator
echo   - Stop:    nssm stop Dictator
echo   - Restart: nssm restart Dictator
echo   - Status:  sc query Dictator
echo.
pause
