# PowerShell profile to add python alias, avoiding Windows App Execution Alias
Set-Alias -Name python -Value 'C:\Program Files\Python314\python.exe' -Force
Set-Alias -Name pip -Value 'C:\Program Files\Python314\Scripts\pip.exe' -Force

Write-Host "Python aliases configured: python -> Python 3.14.2"
