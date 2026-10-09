<#
.SYNOPSIS
    Configures Windows Task Scheduler for the Autonomous AI + Data Development Agent.

.DESCRIPTION
    Registers or unregisters an unattended scheduled task in Windows Task Scheduler
    to run daily at a configurable time. Uses single-instance concurrency lock
    guarantees so parallel tasks cannot interfere.
    Does NOT execute the task automatically during registration.

.PARAMETER Time
    The daily execution time in 24-hour HH:mm format (default: "22:00").

.PARAMETER Action
    Action to perform: 'Register', 'Unregister', 'Status', or 'ShowCommand'.

.PARAMETER TaskName
    Name of the scheduled task (default: 'GitHub-Development-Agent').

.EXAMPLE
    .\scripts\setup_scheduler.ps1 -Action Register -Time "23:00"
    .\scripts\setup_scheduler.ps1 -Action Status
    .\scripts\setup_scheduler.ps1 -Action Unregister
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [ValidateSet("Register", "Unregister", "Status", "ShowCommand")]
    [string]$Action = "Status",

    [Parameter(Mandatory = $false)]
    [string]$Time = "22:00",

    [Parameter(Mandatory = $false)]
    [string]$TaskName = "GitHub-Development-Agent"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$PythonExe = (Get-Command python.exe).Source

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  AUTONOMOUS DEVELOPMENT AGENT - WINDOWS SCHEDULER SETUP" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Project Root: $ProjectRoot"
Write-Host " Python Executable: $PythonExe"
Write-Host " Task Name: $TaskName"
Write-Host " Scheduled Time: $Time (Daily)"
Write-Host "----------------------------------------------------------------------"

switch ($Action) {
    "ShowCommand" {
        $CmdArgs = "/create /tn `"$TaskName`" /tr `"\`"$PythonExe\`" -m agent.cli run --scheduled`" /sc DAILY /st $Time /f"
        Write-Host "Command to be registered:" -ForegroundColor Yellow
        Write-Host "schtasks $CmdArgs"
    }

    "Status" {
        Write-Host "Checking current scheduled task status..." -ForegroundColor Yellow
        try {
            $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
            if ($null -ne $Task) {
                Write-Host " [PASS] Task '$TaskName' is currently registered." -ForegroundColor Green
                Write-Host " State: $($Task.State)"
                Write-Host " Actions: $($Task.Actions.Execute) $($Task.Actions.Arguments)"
            } else {
                Write-Host " [INFO] Task '$TaskName' is not registered in Windows Task Scheduler." -ForegroundColor Yellow
            }
        } catch {
            Write-Host " [INFO] Task '$TaskName' is not registered." -ForegroundColor Yellow
        }
    }

    "Register" {
        Write-Host "Registering task '$TaskName' to run daily at $Time..." -ForegroundColor Yellow
        
        $FullAction = "`"$PythonExe`" -m agent.cli run --scheduled"
        
        # Execute schtasks to register the task
        $Result = schtasks.exe /create /tn "$TaskName" /tr "$FullAction" /sc DAILY /st "$Time" /f
        if ($LASTEXITCODE -eq 0) {
            Write-Host " [SUCCESS] Scheduled task '$TaskName' successfully registered!" -ForegroundColor Green
            Write-Host " Note: The task was NOT executed. It will run automatically at $Time." -ForegroundColor Cyan
        } else {
            Write-Host " [FAIL] Failed to register scheduled task: $Result" -ForegroundColor Red
        }
    }

    "Unregister" {
        Write-Host "Unregistering task '$TaskName'..." -ForegroundColor Yellow
        $Result = schtasks.exe /delete /tn "$TaskName" /f
        if ($LASTEXITCODE -eq 0) {
            Write-Host " [SUCCESS] Scheduled task '$TaskName' successfully deleted." -ForegroundColor Green
        } else {
            Write-Host " [FAIL] Could not delete task '$TaskName': $Result" -ForegroundColor Red
        }
    }
}

Write-Host "======================================================================" -ForegroundColor Cyan
