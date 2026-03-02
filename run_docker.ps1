# PowerShell script to build and run the IISC-HACK Docker image
# Save this file as run_docker.ps1 in the project root (c:/Hackahon project/IISC/IISC-HACK)

# Optional: load HF_TOKEN from a .env file if present
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$") {
            $name = $matches[1]
            $value = $matches[2]
            [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

# Build the Docker image (tag: iisc-hack:latest)
Write-Host "Building Docker image..."
$buildCmd = "docker build -t iisc-hack:latest `"$PSScriptRoot`""
Invoke-Expression $buildCmd
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed. Exiting."
    exit $LASTEXITCODE
}

# Run the container with GPU support and expose port 8000
Write-Host "Running Docker container..."
$runCmd = "docker run --gpus all -p 8000:8000 --rm iisc-hack:latest"
Invoke-Expression $runCmd
