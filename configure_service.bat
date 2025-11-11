@echo off
REM Configure Dictator service to run as current user
setlocal enabledelayedexpansion

echo ========================================
echo    Configure Dictator Service
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

REM Get NSSM path
set "NSSM_PATH="
where nssm >nul 2>&1
if !errorLevel! equ 0 (
    set "NSSM_PATH=nssm"
) else if exist "%WINDIR%\System32\nssm.exe" (
    set "NSSM_PATH=%WINDIR%\System32\nssm.exe"
) else (
    echo ERROR: NSSM not found!
    echo Please run install_service_fixed.bat first
    pause
    exit /b 1
)

echo Current user: %USERNAME%
echo Domain: %USERDOMAIN%
echo.

REM Stop service if running
"!NSSM_PATH!" stop Dictator >nul 2>&1
timeout /t 2 /nobreak >nul

REM Configure service to run as current user with desktop interaction
"!NSSM_PATH!" set Dictator ObjectName ".\%USERNAME%" ""

REM Enable desktop interaction
"!NSSM_PATH!" set Dictator Type SERVICE_INTERACTIVE_PROCESS

echo.
echo Service configured to run as: %USERNAME%
echo Desktop interaction: Enabled
echo.

choice /C YN /M "Start service now?"
if !errorLevel!==1 (
    "!NSSM_PATH!" start Dictator
    if !errorLevel!==0 (
        echo.
        echo [OK] Service started!
        echo Check system tray for Dictator icon.
        echo.
        timeout /t 3 /nobreak >nul
    ) else (
        echo.
        echo ERROR: Failed to start service
        echo.
        echo This might happen because:
        echo 1. Service needs your password to run as your user
        echo 2. Run services.msc and configure manually
        echo.
    )
)

echo.
pause
