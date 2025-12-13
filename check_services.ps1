# Check for Dictator-related services
Write-Host "=== Searching for Dictator-related services ===" -ForegroundColor Cyan

$services = Get-Service | Where-Object {
    $_.DisplayName -like '*Dictator*' -or
    $_.Name -like '*Dictator*' -or
    $_.DisplayName -like '*dictator*' -or
    $_.Name -like '*dictator*'
}

if ($services) {
    $services | Format-Table Name, DisplayName, Status, StartType -AutoSize
} else {
    Write-Host "No Dictator services found." -ForegroundColor Green
}

Write-Host "`n=== Checking for orphaned NSSM services ===" -ForegroundColor Cyan
$allServices = Get-WmiObject Win32_Service | Where-Object {
    $_.PathName -like '*nssm*' -or
    $_.PathName -like '*Dictator*' -or
    $_.PathName -like '*dictator*'
}

if ($allServices) {
    $allServices | Format-Table Name, DisplayName, State, PathName -AutoSize
} else {
    Write-Host "No NSSM or Dictator services found in WMI." -ForegroundColor Green
}
