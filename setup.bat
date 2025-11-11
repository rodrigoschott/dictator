@echo off
REM Dictator Setup - Install all dependencies
REM Run this ONCE before using run_local.bat

echo ========================================
echo    Dictator - Dependency Setup
echo ========================================
echo.
echo This will install:
echo   - faster-whisper (Whisper AI with RTX 5080 support)
echo   - CTranslate2 (optimized inference engine)
echo   - All service dependencies
echo.

cd /d "%~dp0"

echo [1/4] Checking Python...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.10+ from python.org
    pause
    exit /b 1
)
python --version
echo [OK] Python found

echo.
echo [2/4] Checking Poetry...
poetry --version >nul 2>&1
if %errorLevel% neq 0 (
    echo Poetry not found, installing...
    python -m pip install poetry
    if %errorLevel% neq 0 (
        echo ERROR: Failed to install Poetry
        pause
        exit /b 1
    )
)
poetry --version
echo [OK] Poetry found

echo.
echo [3/4] Installing/updating dependencies...
echo This may take 2-5 minutes (network issues will be ignored)...
echo.
poetry install --no-interaction >nul 2>&1

echo [OK] Dependency check complete

:verify
echo.
echo [4/4] Verifying installation...
poetry run python verify_deps.py

if %errorLevel% neq 0 (
    echo.
    echo ERROR: Some critical dependencies are missing!
    echo.
    echo Troubleshooting:
    echo   1. Check your internet connection
    echo   2. Run: poetry install (manually)
    echo   3. Contact support if issue persists
    pause
    exit /b 1
)

echo.
echo ========================================
echo    Setup Complete!
echo ========================================
echo.
echo NOTE: Network warnings during install are OK if verification passed.
echo       (setuptools/sympy updates are optional)
echo.
echo Configuration:
echo   - Trigger: Mouse Side Button (back button)
echo   - Model: large-v3 (RTX 5080 optimized)
echo   - Language: Portuguese
echo.
echo Next steps:
echo   1. Test locally: run_local.bat
echo   2. Install as service: install_service.bat (as Admin)
echo.
echo To change settings, edit: config.yaml
echo.
pause
