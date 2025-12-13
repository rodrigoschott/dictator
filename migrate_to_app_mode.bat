@echo off
REM Migrate from Windows Service to normal application mode
REM Run as Administrator

net session >nul 2>&1
if errorlevel 1 (
    echo This script requires Administrator privileges
    echo Right-click and select "Run as Administrator"
    pause
    exit /b 1
)

echo ========================================
echo Dictator Service to App Migration
echo ========================================
echo.

REM Check if service exists
sc query Dictator >nul 2>&1
if errorlevel 1 (
    echo Service not found - nothing to migrate
    pause
    exit /b 0
)

echo Step 1: Stopping service...
nssm stop Dictator
timeout /t 3 /nobreak >nul

echo Step 2: Removing service...
nssm remove Dictator confirm

echo Step 3: Creating startup shortcut...
call add_to_startup.bat

echo.
echo ========================================
echo Migration completed successfully
echo ========================================
echo.
echo Dictator will now start automatically with Windows
echo as a normal application (not a service)
echo.
echo IMPORTANT: Hotkeys will NOT work in elevated apps
echo (Admin CMD, Task Manager, etc.)
echo.
pause
