@echo off
REM Dictator Service Installer
REM Installs Dictator as a Windows Service using NSSM

echo ========================================
echo    Dictator Service Installer
echo ========================================
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script requires administrator privileges!
    echo Right-click and select "Run as Administrator"
    pause
    exit /b 1
)

echo Checking requirements...

REM Check if Python is installed
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Python not found! Please install Python first.
    pause
    exit /b 1
)
echo [OK] Python found

REM Check if Poetry is installed
poetry --version >nul 2>&1
if %errorLevel% neq 0 (
    echo.
    echo Poetry not found! Installing Poetry...
    python -m pip install poetry
    if %errorLevel% neq 0 (
        echo ERROR: Failed to install Poetry
        pause
        exit /b 1
    )
    echo [OK] Poetry installed
) else (
    echo [OK] Poetry found
)

REM Check if NSSM is installed
where nssm >nul 2>&1
if %errorLevel% neq 0 (
    echo.
    echo NSSM ^(Non-Sucking Service Manager^) not found!
    echo.
    echo Downloading NSSM...

    REM Download NSSM using PowerShell
    powershell -Command "Invoke-WebRequest -Uri 'https://nssm.cc/release/nssm-2.24.zip' -OutFile '%TEMP%\nssm.zip'"

    echo Extracting NSSM...
    powershell -Command "Expand-Archive -Path '%TEMP%\nssm.zip' -DestinationPath '%TEMP%\nssm' -Force"

    REM Copy NSSM to Windows directory
    copy "%TEMP%\nssm\nssm-2.24\win64\nssm.exe" "%WINDIR%\System32\" >nul

    if %errorLevel% neq 0 (
        echo ERROR: Failed to install NSSM
        pause
        exit /b 1
    )

    echo [OK] NSSM installed
) else (
    echo [OK] NSSM found
)

REM Get current directory
set "CURRENT_DIR=%~dp0"
cd /d "%CURRENT_DIR%"

echo.
echo Installing dependencies with Poetry (faster-whisper)...
poetry install

echo.
echo Detecting Poetry virtualenv location...

REM Get Poetry virtualenv path
for /f "delims=" %%i in ('poetry env info --path 2^>nul') do set "VENV_PATH=%%i"

REM Check if VENV_PATH was set
if "%VENV_PATH%"=="" (
    echo ERROR: Could not detect Poetry virtualenv!
    echo Run: poetry env info --path
    pause
    exit /b 1
)

REM Check if Python exists in virtualenv
if not exist "%VENV_PATH%\Scripts\python.exe" (
    echo ERROR: Could not find Poetry virtualenv Python executable!
    echo Expected: %VENV_PATH%\Scripts\python.exe
    pause
    exit /b 1
)

echo [OK] Found virtualenv: %VENV_PATH%

echo.
echo Installing Dictator Service...

REM Create logs directory if it doesn't exist
if not exist "%CURRENT_DIR%logs" (
    mkdir "%CURRENT_DIR%logs"
    echo [OK] Created logs directory
)

REM Stop service if already exists
nssm stop Dictator >nul 2>&1
nssm remove Dictator confirm >nul 2>&1

REM Install service (using Poetry environment)
nssm install Dictator "%VENV_PATH%\Scripts\python.exe" "%CURRENT_DIR%src\dictator\tray.py" "%CURRENT_DIR%config.yaml"

REM Configure service
nssm set Dictator AppDirectory "%CURRENT_DIR%"
nssm set Dictator DisplayName "Dictator - Voice to Text"
nssm set Dictator Description "Background voice-to-text service with global hotkey support"
nssm set Dictator Start SERVICE_AUTO_START

REM Set stdout/stderr redirects
nssm set Dictator AppStdout "%CURRENT_DIR%logs\service.log"
nssm set Dictator AppStderr "%CURRENT_DIR%logs\service_error.log"

REM Set environment
nssm set Dictator AppEnvironmentExtra PYTHONUNBUFFERED=1

echo.
echo Service installed successfully!
echo.
echo Service Name: Dictator
echo Trigger: Mouse Side Button (configurable via system tray menu)
echo.

choice /C YN /M "Start service now?"
if %errorLevel%==1 (
    nssm start Dictator
    if %errorLevel%==0 (
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
