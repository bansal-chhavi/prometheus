param(
    [string]$PythonPath = "C:\\Python\\Python39\\python.exe",
    [string]$ServiceName = "TrustSignal",
    [string]$WorkingDir = "C:\\path\\to\\repo",
    [string]$Host = "127.0.0.1",
    [int]$Port = 8000
)

# Example: run as admin
# .\install_windows_service.ps1 -PythonPath "C:\\Users\\you\\AppData\\Local\\Programs\\Python\\Python39\\python.exe" -WorkingDir "C:\\path\\to\\repo"

$uvicornCmd = "-m uvicorn src.trustsignal.api:app --host $Host --port $Port --log-level info"
$binPath = "\"$PythonPath\" $uvicornCmd"

Write-Host "Creating Windows service '$ServiceName' that runs: $binPath"

# Create the service (requires admin)
New-Service -Name $ServiceName -BinaryPathName $binPath -DisplayName $ServiceName -Description "TrustSignal FastAPI service" -StartupType Automatic

Write-Host "Service created. Start it with: Start-Service -Name $ServiceName"
Write-Host "To remove: Stop-Service -Name $ServiceName ; Remove-Service -Name $ServiceName"