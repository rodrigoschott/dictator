@echo off
REM Run Dictator locally with admin privileges
REM This allows capturing input from elevated applications

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo This script must run as Administrator!
    echo Right-click and select "Run as Administrator"
    pause
    exit /b 1
)

echo ========================================
echo    Dictator - Local Admin Mode
echo ========================================
echo.
echo Running with admin privileges
echo This allows capturing input from elevated apps
echo.

cd /d "%~dp0"

echo Starting Dictator with system tray...
echo Press mouse button to record
echo Right-click tray icon to exit
echo.

poetry run python src\dictator\tray.py config.yaml

pause
