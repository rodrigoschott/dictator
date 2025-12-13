@echo off
REM Start Dictator as normal application (development mode)

echo Starting Dictator...
poetry run python src/dictator/tray.py config.yaml

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start Dictator
    pause
)
