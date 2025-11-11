@echo off
REM Restart Dictator with updated code
echo ========================================
echo    Restarting Dictator
echo ========================================
echo.

REM Kill any running instances
taskkill /F /IM pythonw.exe /FI "WINDOWTITLE eq Dictator*" >nul 2>&1
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Dictator*" >nul 2>&1

echo Stopping any running instances...
timeout /t 2 /nobreak >nul

REM Check if running as scheduled task
schtasks /Query /TN "Dictator" >nul 2>&1
if %errorLevel% equ 0 (
    echo Starting via Task Scheduler...
    schtasks /Run /TN "Dictator"
) else (
    echo Task not found. Starting manually...
    cd /d "%~dp0"

    REM Get virtualenv path
    for /f "delims=" %%i in ('poetry env info --path 2^>nul') do set "VENV_PATH=%%i"

    if "%VENV_PATH%"=="" (
        echo ERROR: Could not find Poetry virtualenv
        pause
        exit /b 1
    )

    start "" "%VENV_PATH%\Scripts\pythonw.exe" "%~dp0src\dictator\tray.py" "%~dp0config.yaml"
)

echo.
echo [OK] Dictator restarted!
echo Check system tray for microphone icon.
echo.
timeout /t 2 /nobreak
