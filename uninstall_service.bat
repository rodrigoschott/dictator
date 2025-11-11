@echo off
REM Dictator Service Uninstaller

echo ========================================
echo   Dictator Service Uninstaller
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

echo Stopping Dictator service...
nssm stop Dictator

timeout /t 2 /nobreak >nul

echo Removing Dictator service...
nssm remove Dictator confirm

if %errorLevel%==0 (
    echo.
    echo [OK] Dictator service removed successfully!
) else (
    echo.
    echo ERROR: Failed to remove service
)

echo.
choice /C YN /M "Remove configuration and logs?"
if %errorLevel%==1 (
    echo Removing config and logs...
    if exist "%~dp0logs" rd /s /q "%~dp0logs"
    echo [OK] Cleaned up
)

echo.
echo Uninstallation complete!
echo.
pause
