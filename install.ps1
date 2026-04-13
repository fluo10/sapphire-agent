$ErrorActionPreference = 'Stop'

$Repo = 'fluo10/sapphire-agent'
$InstallDir = Join-Path $HOME '.local\bin'
$Binary = 'sapphire-agent'

$Release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest"
$Version = $Release.tag_name
if (-not $Version) {
    Write-Error 'Failed to fetch latest version.'
    exit 1
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

$Asset = "$Binary-windows-x86_64.exe"
$Url = "https://github.com/$Repo/releases/download/$Version/$Asset"
$Dest = Join-Path $InstallDir "$Binary.exe"

Write-Host "Installing $Binary $Version to $InstallDir..."
Invoke-WebRequest -Uri $Url -OutFile $Dest
Write-Host "Done! $Dest installed."

$UserPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable('PATH', "$InstallDir;$UserPath", 'User')
    Write-Host ""
    Write-Host "Added $InstallDir to your PATH. Restart your terminal to apply."
}
