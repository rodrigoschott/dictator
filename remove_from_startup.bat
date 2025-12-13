@echo off
REM Remove Dictator from Windows Startup folder

setlocal
set STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup

echo Removing startup shortcut...

del "%STARTUP%\Dictator.lnk" 2>nul

if errorlevel 1 (
    echo Shortcut not found
) else (
    echo [OK] Startup shortcut removed
)

pause
