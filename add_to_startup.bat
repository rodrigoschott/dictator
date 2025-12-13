@echo off
REM Add Dictator to Windows Startup folder

setlocal
set STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set INSTALL_DIR=D:\Programas\Dictator

echo Creating startup shortcut...

powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%STARTUP%\Dictator.lnk'); $s.TargetPath = '%INSTALL_DIR%\Dictator.exe'; $s.Arguments = '\"%INSTALL_DIR%\config.yaml\"'; $s.WorkingDirectory = '%INSTALL_DIR%'; $s.Description = 'Dictator Voice-to-Text'; $s.Save()"

if errorlevel 1 (
    echo [ERROR] Failed to create startup shortcut
) else (
    echo [OK] Startup shortcut created
)

pause
