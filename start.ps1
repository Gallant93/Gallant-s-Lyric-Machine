# start.ps1
# Dieses Skript startet den Flask-Entwicklungsserver

Write-Host "Aktiviere virtuelle Umgebung..."
.\.venv\Scripts\Activate.ps1

Write-Host "Setze FLASK_APP Variable..."
$env:FLASK_APP = "backend/main.py"

Write-Host "Starte Flask-Server..."
flask run --debug --port=8080