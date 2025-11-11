@echo off
REM Run Dictator locally (not as service) for testing

echo ========================================
echo    Dictator - Local Test Mode
echo ========================================
echo.

cd /d "%~dp0"

echo Starting Dictator with system tray...
echo Press mouse side button to record
echo Right-click tray icon to exit
echo.
echo NOTE: Won't work in elevated apps (use run_local_admin.bat instead)
echo.

poetry run python src\dictator\tray.py config.yaml

pause
