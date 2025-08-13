# Wähle ein offizielles Python-Image als Basis. Python 3.9 ist stabil.
FROM python:3.9-slim-buster

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere die requirements.txt und installiere die Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Füge den möglichen Installationspfad für Python-Binärdateien zum PATH hinzu
ENV PATH="/usr/local/bin:${PATH}"

# Kopiere den gesamten Rest des Anwendungscodes in das Arbeitsverzeichnis
COPY . .

# Cloud Run erwartet, dass deine Anwendung auf dem durch die Umgebungsvariable PORT
# angegebenen Port lauscht. Standardmäßig ist 8080.
ENV PORT 8080

# Startbefehl für die Anwendung mit Gunicorn
# WICHTIG: Verwende 'sh -c', um sicherzustellen, dass die Shell-Variablenexpansion stattfindet.
CMD sh -c "gunicorn --bind 0.0.0.0:${PORT} main:app"