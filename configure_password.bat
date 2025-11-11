@echo off
REM Configure Dictator service password
setlocal enabledelayedexpansion

echo ========================================
echo    Configure Dictator Password
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
    pause
    exit /b 1
)

echo Opening NSSM service editor...
echo.
echo INSTRUCTIONS:
echo 1. Go to the "Log on" tab
echo 2. Enter your Windows password in "Password" field
echo 3. Confirm password in "Confirm" field
echo 4. Click "Edit service"
echo 5. Close the window
echo.
echo Press any key to open editor...
pause >nul

"!NSSM_PATH!" edit Dictator

echo.
echo Configuration updated!
echo.

choice /C YN /M "Start service now?"
if !errorLevel!==1 (
    "!NSSM_PATH!" start Dictator
    if !errorLevel!==0 (
        echo.
        echo [OK] Service started!
        echo Check system tray for Dictator icon.
        echo.
    ) else (
        echo.
        echo ERROR: Failed to start service
        echo Check if you entered the correct password.
        echo.
    )
)

echo.
pause
