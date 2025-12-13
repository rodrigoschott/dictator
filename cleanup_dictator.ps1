# Dictator Cleanup Script
# Removes all Dictator shortcuts and checks for duplicates

Write-Host "=== DICTATOR CLEANUP SCRIPT ===" -ForegroundColor Cyan
Write-Host ""

# 1. Check Startup folder
Write-Host "[1/4] Checking Startup folder..." -ForegroundColor Yellow
$startupPath = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup"
$startupShortcuts = Get-ChildItem -Path $startupPath -Filter "Dictator*.lnk" -ErrorAction SilentlyContinue

if ($startupShortcuts) {
    Write-Host "Found in Startup:" -ForegroundColor Green
    $startupShortcuts | ForEach-Object {
        Write-Host "  - $($_.FullName)" -ForegroundColor White
    }
} else {
    Write-Host "  No Dictator shortcuts in Startup" -ForegroundColor Gray
}

# 2. Check Desktop
Write-Host "`n[2/4] Checking Desktop..." -ForegroundColor Yellow
$desktopPath = [Environment]::GetFolderPath("Desktop")
$desktopShortcuts = Get-ChildItem -Path $desktopPath -Filter "Dictator*.lnk" -ErrorAction SilentlyContinue

if ($desktopShortcuts) {
    Write-Host "Found on Desktop:" -ForegroundColor Green
    $desktopShortcuts | ForEach-Object {
        Write-Host "  - $($_.FullName)" -ForegroundColor White
    }
} else {
    Write-Host "  No Dictator shortcuts on Desktop" -ForegroundColor Gray
}

# 3. Check Start Menu
Write-Host "`n[3/4] Checking Start Menu..." -ForegroundColor Yellow
$startMenuPath = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs"
$startMenuShortcuts = Get-ChildItem -Path $startMenuPath -Filter "Dictator*" -Recurse -ErrorAction SilentlyContinue

if ($startMenuShortcuts) {
    Write-Host "Found in Start Menu:" -ForegroundColor Green
    $startMenuShortcuts | ForEach-Object {
        Write-Host "  - $($_.FullName)" -ForegroundColor White
    }
} else {
    Write-Host "  No Dictator items in Start Menu" -ForegroundColor Gray
}

# 4. Check running processes
Write-Host "`n[4/4] Checking running processes..." -ForegroundColor Yellow
$processes = Get-Process -Name "Dictator" -ErrorAction SilentlyContinue

if ($processes) {
    Write-Host "Found running processes:" -ForegroundColor Red
    $processes | ForEach-Object {
        Write-Host "  - PID: $($_.Id) | Memory: $([math]::Round($_.WorkingSet64/1MB, 2)) MB" -ForegroundColor White
    }
} else {
    Write-Host "  No Dictator processes running" -ForegroundColor Gray
}

# Summary
Write-Host "`n=== SUMMARY ===" -ForegroundColor Cyan
$totalShortcuts = ($startupShortcuts.Count + $desktopShortcuts.Count + $startMenuShortcuts.Count)
Write-Host "Total shortcuts found: $totalShortcuts" -ForegroundColor White
Write-Host "Total processes running: $($processes.Count)" -ForegroundColor White

# Offer cleanup
Write-Host "`n=== CLEANUP OPTIONS ===" -ForegroundColor Cyan
Write-Host "To remove Startup shortcut: Remove-Item '$startupPath\Dictator.lnk' -Force" -ForegroundColor Yellow
Write-Host "To remove Desktop shortcut: Remove-Item '$desktopPath\Dictator.lnk' -Force" -ForegroundColor Yellow
Write-Host "To kill processes: Stop-Process -Name Dictator -Force" -ForegroundColor Yellow
