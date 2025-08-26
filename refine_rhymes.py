# test_models.py (Version nur für Claude)
import os
import time
from dotenv import load_dotenv
import anthropic

# Lädt den API-Schlüssel aus deiner .env-Datei
load_dotenv()

# ---- KONFIGURATION ----
try:
    # Wir brauchen nur noch den Anthropic-Client
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
except Exception as e:
    print(f"Fehler bei der Initialisierung des Anthropic-Clients. Ist der Key in der .env-Datei korrekt? Fehler: {e}")
    exit()

# Der "Master-Prompt" für maximale Präzision
prompt = """
## AUFGABE
Du bist ein deutscher Linguistik-Professor, der auf Metrik und Phonetik spezialisiert ist. Deine Aufgabe ist es, eine Liste perfekter Reime zu erstellen, indem du einen strengen, dreistufigen Prozess befolgst.

---

### SCHRITT 1: INTERNES BRAINSTORMING
Generiere eine lange, interne Liste von 30-40 potenziellen Reim-Kandidaten für die unten stehenden Testfälle. Diese Liste wird nicht ausgegeben.

---

### SCHRITT 2: SYSTEMATISCHE ANALYSE & BEWERTUNG
Analysiere JEDEN Kandidaten aus deiner Brainstorming-Liste einzeln und extrem penibel. Erstelle für jeden Kandidaten eine interne Bewertung basierend auf den Regeln des jeweiligen Testfalls.

---

### SCHRITT 3: FINALE, GEFILTERTE AUSGABE
Gib als Ergebnis NUR eine Liste der Kandidaten zurück, die deine strenge Analyse zu 100% bestanden haben. Halte dich exakt an die Regeln.

---

## TESTFALL 1
- **Zielwort:** "fabelhaft"
- **Regeln:**
    1. Der Reim MUSS exakt 3 Silben haben.
    2. Die Vokalfolge des Reims MUSS exakt "a-e-a" sein.
    3. Die erste Silbe darf phonetisch nicht identisch sein (z.B. "fabel-" vs. "Gabel-").
    4. Die letzte Silbe darf phonetisch nicht identisch sein (z.B. "-haft" vs. "-schaft").
- **Ergebnis:** Liste die gefundenen Reime, die ALLE Regeln erfüllen, als Aufzählung auf.

## TESTFALL 2
- **Zielwort:** "Sommerregen"
- **Regeln:**
    1. Der Reim MUSS exakt 4 Silben haben.
    2. Die Vokalfolge des Reims MUSS exakt "o-e-e-e" sein.
    3. Mehrwortphrasen sind erlaubt.
    4. Die erste Silbe darf phonetisch nicht identisch sein.
    5. Die letzte Silbe darf phonetisch nicht identisch sein.
- **Ergebnis:** Liste die gefundenen Reime, die ALLE Regeln erfüllen, als Aufzählung auf.
"""


# ---- TEST-DURCHFÜHRUNG ----

def run_claude_test():
    model_to_use = "claude-opus-4-1-20250805" # Das beste Modell für diese Aufgabe

    print(f"\n========================================")
    print(f"▶️  Teste Modell: Anthropic Claude 3 Opus")
    print(f"========================================")
    
    start_time = time.time()
    
    try:
        response = anthropic_client.messages.create(
            model=model_to_use,
            max_tokens=2048, # Erhöht, um Platz für den Denkprozess zu schaffen
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2 # Leicht kreativ, aber sehr fokussiert
        )
        output_text = response.content[0].text
        
        duration = time.time() - start_time
        print(f"✅ Antwort erhalten nach {duration:.2f} Sekunden.")
        print("---")
        print(output_text)

    except Exception as e:
        print(f"❌ FEHLER bei Claude: {e}")


# Führt den Test aus, wenn das Skript gestartet wird
if __name__ == "__main__":
    run_claude_test()