@echo off
REM Stop all Dictator processes

echo Stopping Dictator...
taskkill /IM Dictator.exe /F 2>nul

if errorlevel 1 (
    echo Dictator not running
) else (
    echo [OK] Dictator stopped
)

pause
