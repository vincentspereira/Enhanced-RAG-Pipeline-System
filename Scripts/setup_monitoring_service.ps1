Import-Module PSScheduledJob

$trigger = New-JobTrigger -AtStartup -RandomDelay 00:00:30
$options = New-ScheduledJobOption -RunElevated -StartIfOnBattery -ContinueIfGoingOnBattery

$scriptBlock = {
    Set-Location "C:\Users\Vincent_Pereira\Qdrant\Scripts"
    # Activate virtual environment
    & "C:\Users\Vincent_Pereira\Qdrant\venv_Qdrant\Scripts\Activate.ps1"
    # Start the document monitor
    python document_monitor.py
}

Register-ScheduledJob -Name "QdrantDocumentMonitor" -Trigger $trigger -ScheduledJobOption $options -ScriptBlock $scriptBlock
