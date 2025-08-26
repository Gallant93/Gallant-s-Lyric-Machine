# test_models.py
import os
import time
from dotenv import load_dotenv
import anthropic

# Lädt die API-Schlüssel aus Ihrer .env-Datei
load_dotenv()

# ---- KONFIGURATION ----
try:
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
except Exception as e:
    print(f"Fehler bei der Initialisierung eines Clients. Stellen Sie sicher, dass die API-Keys in der .env-Datei korrekt sind. Fehler: {e}")
    exit()

# Neuer, präziser Prompt
prompt = """
## AUFGABE
Du bist ein Reim-Experte. Finde perfekte, mehrsilbige Reime für die folgenden Wörter unter strikter Einhaltung der vorgegebenen Regeln.


## TESTFALL 1
- **Zielwort:** "fabelhaft"
- **Regeln:**
    1. Der Reim MUSS exakt 3 Silben haben.
    2. Die Vokalfolge des Reims MUSS exakt "a-e-a" sein.
    3. Die erste Silbe darf nicht exakt gleich sein (Beispiel:"Oberschenkel" auf "Oberkellner")
    4. Die letzte Silbe darf nicht exakt gleich sein (Beispiel: "fabelhaft" auf "sagenhaft")
- **Ergebnis:** Liste die gefundenen Reime als Aufzählung auf.

## TESTFALL 2
- **Zielwort:** "Sommerregen"
- **Regeln:**
    1. Der Reim MUSS exakt 4 Silben haben.
    2. Die Vokalfolge des Reims MUSS exakt "o-e-e-e" sein.
    3. Mehrwortphrasen, die eine logische Einheit bilden (z.B. "oben schweben"), sind erlaubt.
    4. Die erste Silbe darf nicht exakt gleich sein (Beispiel:"Oberschenkel" auf "Oberkellner")
    5. Die letzte Silbe darf nicht exakt gleich sein (Beispiel: "fabelhaft" auf "sagenhaft")
- **Ergebnis:** Liste die gefundenen Reime als Aufzählung auf.

## QUALITÄTSSICHERUNG (Sehr Wichtig)
Nachdem du eine erste Liste an Reimen für einen Testfall erstellt hast, führe die folgenden Schritte aus:
1.  **Überprüfung:** Gehe deine gefundene Liste Wort für Wort durch. Prüfe jeden einzelnen Kandidaten erneut und extrem streng gegen die Regeln (exakte Silbenzahl und exakte Vokalfolge).
2.  **Filterung:** Wenn ein Wort auch nur minimal von den Regeln abweicht, entferne es sofort aus der Liste.
3.  **Iteration & Verbesserung:** Ersetze jedes entfernte Wort durch einen neuen, besseren Kandidaten. Überprüfe diesen neuen Kandidaten ebenfalls sofort streng nach den Regeln.
4.  **Ziel:** Wiederhole diesen Prozess, bis du eine finale Liste mit mindestens 10 Reimen pro Testfall hast, die zu 100% allen aufgestellten Regeln entsprechen.
"""

validator_prompt_template = """
## AUFGABE: Strenge Qualitätsprüfung
Du bist ein extrem genauer und unbestechlicher Reim-Validator. Deine einzige Aufgabe ist es, die folgende "Kandidatenliste" zu überprüfen und eine finale, perfekte Liste zu erstellen.

## KANDIDATENLISTE ZUR ÜBERPRÜFUNG:
{claude_candidates}

---

## DEINE ANWEISUNGEN:
1.  **Analysiere die Kandidatenliste** für beide Testfälle ("fabelhaft" und "Sommerregen").
2.  **Prüfe jeden einzelnen Kandidaten** gegen die folgenden, nicht verhandelbaren Regeln:
    - **Für "fabelhaft":** MUSS exakt 3 Silben haben UND die Vokalfolge MUSS exakt "a-e-a" sein.
    - **Für "Sommerregen":** MUSS exakt 4 Silben haben UND die Vokalfolge MUSS exakt "o-e-e-e" sein.
3.  **Erstelle eine finale Liste**, die NUR die Kandidaten enthält, die die Regeln zu 100% perfekt erfüllen. Gib keine Erklärungen, nur die finalen, gefilterten Listen für jeden Testfall.
"""

# Liste der Modelle, die wir testen wollen
models_to_test = [
    {"name": "Anthropic Claude 4 Opus", "type": "anthropic", "id": "claude-opus-4-1-20250805"},
]


# ---- AUSFÜHRUNG DES ZWEISTUFIGEN PROZESSES ----

def generate_and_validate():
    # -- SCHRITT 1: GENERIEREN mit Claude --
    print("========================================")
    print("▶️  Schritt 1: Claude generiert kreative Kandidaten...")
    print("========================================")
    
    start_time_claude_gen = time.time()
    try:
        response_claude_gen = anthropic_client.messages.create(
            model="claude-opus-4-1-20250805",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        claude_output = response_claude_gen.content[0].text
        duration_claude_gen = time.time() - start_time_claude_gen
        print(f"✅ Claude hat eine Kandidatenliste in {duration_claude_gen:.2f}s erstellt.")
        print("---\n" + claude_output + "\n---")

    except Exception as e:
        print(f"❌ FEHLER bei Claude (Generierung): {e}")
        return

    # -- SCHRITT 2: VALIDIEREN mit Claude --
    print("\n========================================")
    print("▶️  Schritt 2: Claude validiert die Kandidaten...")
    print("========================================")
    
    # Fülle die Vorlage mit der Antwort von Claude
    validation_prompt = validator_prompt_template.format(claude_candidates=claude_output)
    
    start_time_claude_val = time.time()
    try:
        response_claude_val = anthropic_client.messages.create(
            model="claude-opus-4-1-20250805",
            max_tokens=1024,
            messages=[{"role": "user", "content": validation_prompt}]
        )
        claude_validation_output = response_claude_val.content[0].text
        duration_claude_val = time.time() - start_time_claude_val
        
        print(f"✅ Claude hat die Liste in {duration_claude_val:.2f}s validiert.")
        print("\n========================================")
        print("🏆 DAS FINALE, ULTRA-RAFFINIERTE ERGEBNIS:")
        print("========================================")
        print(claude_validation_output)

    except Exception as e:
        print(f"❌ FEHLER bei Claude (Validierung): {e}")
        return


# Führt den Prozess aus
if __name__ == "__main__":
    generate_and_validate()