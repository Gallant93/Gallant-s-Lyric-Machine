import os
import json
import logging
import sys
import time
import random
import uuid
from typing import Union
import re

# --- NEU: Silbenzählung mit Pyphen (de_DE) ---
try:
    import pyphen
    _pyphen_de = pyphen.Pyphen(lang="de_DE")
except Exception:
    _pyphen_de = None  # Fallback greift dann automatisch

import anthropic
from flask import Flask, jsonify, request
from flask_cors import CORS
import google.generativeai as genai

# GPT optional nutzen
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # Falls Paket nicht installiert ist

from dotenv import load_dotenv

load_dotenv()

print(f"!!! Server läuft aus dem Verzeichnis: {os.getcwd()}")

def parse_json_safely(text: str):
    """
    Entfernt Markdown-Zäune und schneidet alles vor der ersten { und nach der letzten } ab.
    Gibt (obj, error) zurück – error ist None, wenn alles ok.
    """
    try:
        if not text:
            return None, "Leere Antwort der KI."

        cleaned = text.strip()
        # Häufige Markdown-Reste raus
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        # Nur das „echte“ JSON heraus schneiden
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None, "Kein gültiger JSON-Block gefunden."

        sliced = cleaned[start:end+1]
        return json.loads(sliced), None
    except Exception as e:
        return None, f"JSON-Parsing fehlgeschlagen: {e}"


# === ZENTRALES SPRACHVERZEICHNIS ===
# Enthält grundlegende Definitionen für linguistische Konzepte, auf die in Prompts verwiesen wird.

SPRACHVERZEICHNIS = {
    # HINZUFÜGEN ZU SPRACHVERZEICHNIS in main.py

    "Sinnhaftigkeit & Kreativität": "Die oberste Direktive. Jede generierte Zeile muss ein grammatikalisch korrekter, logisch nachvollziehbarer und thematisch passender Satz sein. Das Erfinden von Nicht-Wörtern oder das Bilden von unsinnigen Satzkonstrukten zur reinen Erfüllung phonetischer Regeln ist strengstens verboten. Kreativität entsteht durch clevere Wortwahl, nicht durch die Missachtung von Logik.",
    "Ganzzeilige Phonetische Resonanz": "Ein fortgeschrittenes Reimkonzept, bei dem eine Textzeile nicht nur durch einen Endreim, sondern über ihre gesamte Länge eine klangliche Ähnlichkeit zu einer anderen Zeile aufweist. Dies wird erreicht durch die Spiegelung von Vokal-Rhythmus, Konsonanten-Fluss und Betonungsmuster.",
    "Vokal-Rhythmus": "Die exakte Sequenz von Vokalen und Diphthongen (z.B. 'au', 'ei', 'ie') in einer Textzeile. Beispiel für 'Hallo Welt': a-o-e. Die Spiegelung dieses Rhythmus ist entscheidend für die phonetische Resonanz.",
    "Konsonanten-Fluss": "Das Muster von harten (k, t, p, ch) und weichen (s, f, w, sch) Konsonanten in einer Zeile. Eine gute phonetische Resonanz versucht, ein ähnliches Gefühl durch eine ähnliche Verteilung dieser Konsonanten-Typen zu erzeugen.",
    "Betonungsmuster": "Die natürliche Betonung oder der 'Stress' auf bestimmten Silben oder Wörtern innerhalb einer Zeile, die den rhythmischen Takt (Kadenz) vorgibt. Beispiel: '**KICK**-en **AUFM** **BOL**-zer **BIS** zum **SON**-nen-**UN**-ter-**GANG**'.",
    "Prinzip der Reim-Varianz": "Diese Regel verbietet die Wiederholung des exakten Reimwortes oder seines Kern-Wortstammes in einer Reimzeile. Ziel ist die semantische und lexikalische Vielfalt. Beispiel: 'Sonnenuntergang' auf 'Morgenuntergang' ist verboten. 'Sonnenuntergang' auf 'Wolkenhimmelrand' ist erwünscht."
}

# Um das Verzeichnis einfach in einen Prompt einzufügen
SPRACHVERZEICHNIS_TEXT = "\n".join([f"- **{key}:** {value}" for key, value in SPRACHVERZEICHNIS.items()])

# === Logging Konfiguration ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# === Initialisierung der Dienste ===
CLAUDE_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
if not CLAUDE_API_KEY: raise ValueError("FATAL: ANTHROPIC_API_KEY environment variable not set.")
anthropic_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# Client für das Analyse-Modell (Gemini)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY: raise ValueError("FATAL: GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Optionaler GPT-Client (nur wenn Key vorhanden und Paket verfügbar)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None

# Umschalter für den Reim-Provider („claude" | „gpt")
RHYME_PROVIDER = os.environ.get('RHYME_PROVIDER', 'claude').lower()

# Optional eigenes Modell setzen, sonst solides Default
OPENAI_RHYMES_MODEL = os.environ.get('OPENAI_RHYMES_MODEL', 'gpt-4o-mini')

# Optionales Häufigkeitslexikon (läuft offline)
try:
    from wordfreq import zipf_frequency as _zipf
except Exception:
    _zipf = None

# === ZENTRALE KI-AUFRUF-FUNKTION ===
def call_ai_model(task_type: str, prompt: Union[str, list], schema: dict = None, is_json_output: bool = True):
    """
    Zentrale Funktion für KI-Aufrufe, die als intelligente Weiche zwischen Gemini und Claude fungiert.
    Implementiert einen Retry-Mechanismus für mehr Stabilität.
    """
    # NEU: Konfiguration für Wiederholungsversuche
    max_retries = 3
    last_exception = None

    # Aufgaben, die an das kosteneffiziente Analyse-Modell gehen
    gemini_tasks = ["SYNTHESIZE_STYLES"]

    # NEU: Schleife für Wiederholungsversuche
    for attempt in range(max_retries):
        try:
            # Die gesamte Logik von vorher befindet sich jetzt in diesem try-Block
            if task_type in gemini_tasks:
                logger.info(f"Routing task '{task_type}' to Gemini 1.5 Pro (Attempt {attempt + 1}/{max_retries})...")
                
                # ... (restliche Gemini-Logik bleibt unverändert) ...
                if isinstance(prompt, list):
                    if len(prompt) == 2 and isinstance(prompt[0], dict) and 'inline_data' in prompt[0]:
                        audio_part = prompt[0]
                        text_prompt = prompt[1]
                        response = gemini_model.generate_content([text_prompt, audio_part])
                    else:
                        text_prompt = " ".join([str(p) for p in prompt])
                        response = gemini_model.generate_content(text_prompt)
                else:
                    response = gemini_model.generate_content(prompt)
                
                result = response.text
                
                if is_json_output and schema:
                    try:
                        clean_text = result.strip().replace("```json", "").replace("```", "").replace("```JSON", "")
                        if "{" in clean_text and "}" in clean_text:
                            start = clean_text.find("{")
                            end = clean_text.rfind("}") + 1
                            clean_text = clean_text[start:end]
                        return json.loads(clean_text) # ERFOLG: Funktion wird hier beendet
                    except json.JSONDecodeError as e:
                        logger.warning(f"Could not parse Gemini response as JSON: {e}, returning raw text")
                        return {"candidates": [result]} # ERFOLG
                
                return result # ERFOLG
                
            else:
                # Alle anderen, kreativen Aufgaben gehen an das Top-Modell (Claude)
                logger.info(f"Routing task '{task_type}' to Claude Opus (Attempt {attempt + 1}/{max_retries})...")
                
                # ... (restliche Claude-Logik bleibt unverändert) ...
                if isinstance(prompt, list):
                    if len(prompt) == 2 and isinstance(prompt[0], dict) and 'inline_data' in prompt[0]:
                        audio_part = prompt[0]
                        text_prompt = prompt[1]
                        messages = [{"role": "user", "content": [audio_part, text_prompt]}]
                    else:
                        text_prompt = " ".join([str(p) for p in prompt])
                        messages = [{"role": "user", "content": text_prompt}]
                else:
                    messages = [{"role": "user", "content": prompt}]
                
                if schema:
                    schema_instructions = f"\n\nWICHTIG: Gib deine Antwort AUSSCHLIESSLICH als gültiges JSON aus, das diesem Schema entspricht. Keine zusätzlichen Texte, keine Erklärungen, nur das JSON:\n{json.dumps(schema, indent=2)}"
                    enhanced_prompt = f"{messages[0]['content']}{schema_instructions}"
                    response = anthropic_client.messages.create(model="claude-opus-4-1-20250805", max_tokens=2048, temperature=0.7, messages=[{"role": "user", "content": enhanced_prompt}])
                    try:
                        clean_text = response.content[0].text.strip().replace("```json", "").replace("```", "").replace("```JSON", "")
                        if "{" in clean_text and "}" in clean_text:
                            start = clean_text.find("{")
                            end = clean_text.rfind("}") + 1
                            clean_text = clean_text[start:end]
                        return json.loads(clean_text) # ERFOLG
                    except json.JSONDecodeError as e:
                        logger.warning(f"Could not parse Claude response as JSON: {e}, returning raw text")
                        return {"candidates": [response.content[0].text]} # ERFOLG
                else:
                    response = anthropic_client.messages.create(model="claude-opus-4-1-20250805", max_tokens=2048, temperature=0.7, messages=messages)
                    if is_json_output:
                        try:
                            clean_text = response.content[0].text.strip().replace("```json", "").replace("```", "")
                            return json.loads(clean_text) # ERFOLG
                        except json.JSONDecodeError:
                            return response.content[0].text # ERFOLG
                    return response.content[0].text # ERFOLG
                    
        except Exception as e:
            # NEU: Logik bei Fehlschlag eines Versuchs
            last_exception = e
            logger.warning(f"AI call attempt {attempt + 1}/{max_retries} failed for task '{task_type}': {e}")
            if attempt < max_retries - 1:
                # Wartezeit vor dem nächsten Versuch (z.B. 2 Sekunden)
                sleep_time = 2 * (attempt + 1)
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                # Letzter Versuch ist fehlgeschlagen, also wird die Schleife beendet
                logger.error(f"All {max_retries} attempts failed for task '{task_type}'.")

    # NEU: Wird nur ausgeführt, wenn alle Versuche fehlschlagen
    raise ValueError(f"Error calling AI model for task '{task_type}' after {max_retries} attempts. Last error: {str(last_exception)}")

app = Flask(__name__)
# Die CORS-Konfiguration erlaubt Anfragen vom Frontend, sowohl lokal als auch deployed.
CORS(app, origins=[r"https://.*\.run\.app", "http://localhost:3000", "http://localhost:5173"],
     supports_credentials=True)


# === Helper-Funktionen ===
def make_audio_part(base64_audio: str, mime_type: str):
    """Erstellt den inlineData-Teil für multimodale Prompts."""
    return {"inline_data": {"mime_type": mime_type, "data": base64_audio}}

def call_claude_creative(prompt, temperature=0.7):
    """Speziell für kreative Aufgaben ohne Schema-Zwang"""
    response = anthropic_client.messages.create(
        model="claude-opus-1-20250805",
        max_tokens=2048,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text    


def call_gpt_candidates(prompt: str, schema: dict):
    """
    Ruft GPT auf und erwartet ausschließlich ein JSON-Objekt:
    {"candidates": ["...", "...", ...]}
    """
    if openai_client is None:
        raise RuntimeError("GPT ist nicht initialisiert (OPENAI_API_KEY oder Paket fehlt).")

    system_msg = (
        "Du bist ein Reim-Experte. Antworte AUSSCHLIESSLICH mit einem JSON-Objekt, "
        "das GENAU dem folgenden Schema entspricht. KEINE Erklärungen, KEIN Text außerhalb des JSON."
    )
    # Wir hängen das Schema als Anweisung an; robust gegen kleinere Formatierungsfehler.
    messages = [
        {"role": "system", "content": system_msg + "\n" + json.dumps(schema, ensure_ascii=False)},
        {"role": "user", "content": prompt}
    ]

    resp = openai_client.chat.completions.create(
        model=OPENAI_RHYMES_MODEL,
        messages=messages,
        temperature=0.6,            # etwas strenger für präzisere Kandidaten
        max_tokens=800
    )
    text = resp.choices[0].message.content or ""

    # JSON sicher herauslösen (falls GPT Code-Fences o.ä. setzt)
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        # Fallback: JSON-Kern herausschneiden
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end+1])
            except Exception:
                pass
        # Harte Notbremse: mache aus Zeilen eine Candidates-Liste
        lines = [ln.strip("-*• 0123456789.").strip() for ln in cleaned.splitlines() if ln.strip()]
        return {"candidates": lines[:20]}


def call_claude(prompt: Union[str, list], schema: dict = None, is_json_output: bool = True):
    """
    Ruft die Claude API auf. Wenn ein Schema vorhanden ist, wird JSON-Output erzwungen.
    Verwendet ausgewogene Parameter für natürliche, aber präzise Antworten.
    """
    try:
        # Ausgewogene Parameter für natürlichere, aber präzise Antworten
        generation_config = {
            "temperature": 0.7,  # Höhere Temperatur für natürlichere Antworten
            "max_tokens": 2048,  # Begrenzt die Antwortlänge
        }
        
        # Wenn ein Prompt-Array übergeben wird (für multimodale Eingaben)
        if isinstance(prompt, list):
            # Für Audio + Text Kombinationen
            if len(prompt) == 2 and isinstance(prompt[0], dict) and 'inline_data' in prompt[0]:
                # Audio + Text Format
                audio_part = prompt[0]
                text_prompt = prompt[1]
                messages = [
                    {"role": "user", "content": [audio_part, text_prompt]}
                ]
            else:
                # Fallback für andere Listen
                text_prompt = " ".join([str(p) for p in prompt])
                messages = [{"role": "user", "content": text_prompt}]
        else:
            # Einfacher Text-Prompt
            messages = [{"role": "user", "content": prompt}]
        
        if schema:
            # Verwende Claude mit Schema-Anweisungen im Prompt
            schema_instructions = f"\n\nWICHTIG: Gib deine Antwort AUSSCHLIESSLICH als gültiges JSON aus, das diesem Schema entspricht. Keine zusätzlichen Texte, keine Erklärungen, nur das JSON:\n{json.dumps(schema, indent=2)}"
            enhanced_prompt = f"{messages[0]['content']}{schema_instructions}"
            
            response = anthropic_client.messages.create(
                model="claude-opus-4-1-20250805",
                max_tokens=generation_config["max_tokens"],
                temperature=generation_config["temperature"],
                messages=[{"role": "user", "content": enhanced_prompt}]
            )
            # Versuche, die Antwort als JSON zu parsen
            try:
                clean_text = response.content[0].text.strip()
                # Entferne alle Markdown-Formatierung
                clean_text = clean_text.replace("```json", "").replace("```", "").replace("```JSON", "").replace("```JSON", "")
                # Entferne mögliche Einleitungen oder Erklärungen
                if "{" in clean_text and "}" in clean_text:
                    start = clean_text.find("{")
                    end = clean_text.rfind("}") + 1
                    clean_text = clean_text[start:end]
                
                return json.loads(clean_text)
            except json.JSONDecodeError as e:
                # Falls das Parsen fehlschlägt, gib den rohen Text zurück
                logger.warning(f"Could not parse response as JSON: {e}, returning raw text")
                return {"candidates": [response.content[0].text]}
        else:
            response = anthropic_client.messages.create(
                model="claude-opus-4-1-20250805",
                max_tokens=generation_config["max_tokens"],
                temperature=generation_config["temperature"],
                messages=messages
            )
            # Falls trotzdem JSON erwartet wird (z.B. bei komplexen Anweisungen ohne striktes Schema)
            if is_json_output:
                # Bereinigen des Textes, um Markdown-Formatierung zu entfernen
                clean_text = response.content[0].text.strip().replace("```json", "").replace("```", "")
                try:
                    return json.loads(clean_text)
                except json.JSONDecodeError:
                    return response.content[0].text
            return response.content[0].text
    except Exception as e:
        logger.error(f"CLAUDE_API_ERROR: {e}\nPrompt: {prompt}", exc_info=True)
        # Versuche, eine genauere Fehlermeldung aus der Claude-Antwort zu extrahieren, falls vorhanden
        error_details = str(e)
        raise ValueError(f"Error calling Claude API. Details: {error_details}")


def get_phonetic_breakdown_py(word: str) -> dict:
    """Eine verbesserte Python-Implementierung der phonetischen Analyse."""
    temp_word = word.lower().strip()

    # Schritt 0: Spezielle Behandlung für "er"-Endungen
    if temp_word.endswith('er'):
        temp_word = temp_word[:-2] + 'a'

    # Schritt 1: Deutsche Eigenheiten normalisieren
    temp_word = re.sub(r'([aouäöü])h', r'\1', temp_word)  # Dehnungs-h entfernen
    temp_word = temp_word.replace('ch', 'X').replace('sch', 'Y').replace('ck', 'k')
    
    # Schritt 2: Diphthonge und Vokale in der Reihenfolge ihrer Priorität suchen
    # WICHTIG: Diphthonge zuerst, damit sie nicht als separate Vokale interpretiert werden
    # "ie" muss vor "i" und "e" stehen!
    vowels_and_diphthongs = ['au', 'eu', 'äu', 'ei', 'ai', 'ey', 'ay', 'ie', 'a', 'e', 'i', 'o', 'u', 'ä', 'ö', 'ü']
    vowel_sequence_list = []
    i = 0
    
    while i < len(temp_word):
        found = False
        for v_or_d in vowels_and_diphthongs:
            if temp_word[i:].startswith(v_or_d):
                vowel_sequence_list.append(v_or_d)
                i += len(v_or_d)
                found = True
                break
        if not found:
            i += 1

    # Schritt 3: Qualitätskontrolle
    if len(vowel_sequence_list) == 0:
        # Fallback: Suche nach einzelnen Vokalen, falls keine Diphthonge gefunden wurden
        single_vowels = ['a', 'e', 'i', 'o', 'u', 'ä', 'ö', 'ü']
        for char in temp_word:
            if char in single_vowels:
                vowel_sequence_list.append(char)

    return {
        "syllableCount": len(vowel_sequence_list),
        "vowelSequence": "-".join(vowel_sequence_list),
        "originalWord": word,
        "normalizedWord": temp_word
    }


# === NEU: Basistools (nur einfügen, wenn noch nicht vorhanden) ===
try:
    _VOWEL_CLUSTER_RE
except NameError:
    import re as _re  # falls re oben nicht importiert
    _VOWEL_CLUSTER_RE = _re.compile(
        r'(aa|ee|oo|uu|ää|öö|üü|ie|ei|ai|au|eu|äu|oi|ui|[aeiouäöüy])',
        _re.IGNORECASE
    )
try:
    _DEF_CONS
except NameError:
    _DEF_CONS = "bcdfghjklmnpqrstvwxyzß"

def _last_vowel_cluster(word: str):
    w = (word or "").strip().lower()
    if not w:
        return None, None
    clusters = _VOWEL_CLUSTER_RE.findall(w)
    if not clusters:
        return None, None
    cluster = clusters[-1].lower()
    idx = w.rfind(cluster)
    after = w[idx + len(cluster):]
    return cluster, after

def _length_class(cluster: str, after: str):
    # 'long' | 'short' | 'diph'
    if cluster in ('ei','ai','au','eu','äu','oi','ui'):
        return 'diph'
    if cluster in ('aa','ee','oo','uu','ää','öö','üü') or cluster == 'ie':
        return 'long'
    if (after or '').startswith('h'):
        return 'long'
    if len(after or '') >= 2 and after[0] in _DEF_CONS and after[1] in _DEF_CONS:
        return 'short'
    return 'long' if (after or '') == '' else 'short'

def _vowel_family(cluster: str):
    if cluster in ('ei','ai','au','eu','äu','oi','ui'):
        return cluster  # Diphthonge bleiben eigenständig
    m = {'ä': 'e', 'ö': 'o', 'ü': 'u'}
    base = cluster[0]
    return m.get(base, base)

# === NEU: erster Vokal + Coda-Normalisierung ===
def _first_vowel_cluster(word: str):
    if not word:
        return None
    m = _VOWEL_CLUSTER_RE.search(word.lower())
    return m.group(0) if m else None

# Kons.-Normalisierung für Varianz-Signatur
def _normalize_coda(coda: str) -> str:
    s = (coda or "").lower()
    s = re.sub(r'(.)\1+', r'\1', s)  # Doppelkonsonanten reduzieren
    for a, b in (("sch","sh"),("tsch","ch"),("ck","k"),("tz","z"),("ph","f")):
        s = s.replace(a, b)
    return s

def first_vowel_family_match(base_word: str, candidate: str) -> bool:
    """Erster Vokal (Familie) muss übereinstimmen (ä~e, ö~o, ü~u; Diphthonge 1:1)."""
    b = _first_vowel_cluster(base_word)
    c = _first_vowel_cluster(candidate)
    if not b or not c:
        return True
    return _vowel_family(b) == _vowel_family(c)

# === NEU: Schwa behandeln + Kern bestimmen ===
_SCHWA_SUFFIXES = ("er", "en", "el", "em", "es", "e")  # Reihenfolge wichtig

def get_schwa_suffix(word: str) -> str:
    """Gibt sichtbares Schwa-Suffix zurück (er/en/el/em/es/e) oder ''. Orthografie-Regel."""
    w = (word or "").strip().lower()
    for suf in _SCHWA_SUFFIXES:
        if w.endswith(suf):
            return suf
    return ""

def strip_schwa(word: str) -> str:
    suf = get_schwa_suffix(word)
    return (word or "")[:-len(suf)] if suf else (word or "")

def last_stressed_vowel_cluster(word: str):
    """Letzter betonter Vokal-Cluster (nachdem Schwa entfernt wurde)."""
    base = strip_schwa(word)
    return _last_vowel_cluster(base)

def vowel_core_compatible(base_word: str, candidate: str) -> bool:
    """
    Kernvergleich (betonter Endvokal): Vokalfamilie + Längenklasse/Diphthong müssen passen.
    """
    b_cluster, b_after = last_stressed_vowel_cluster(base_word)
    c_cluster, c_after = last_stressed_vowel_cluster(candidate)
    if not b_cluster or not c_cluster:
        return True
    b_len = _length_class(b_cluster, b_after or '')
    c_len = _length_class(c_cluster, c_after or '')
    if b_len == 'diph':
        return c_len == 'diph' and c_cluster == b_cluster
    return (_vowel_family(b_cluster) == _vowel_family(c_cluster)) and (b_len == c_len)

# === NEU: Varianz-Signatur (mit ignoriertem Schwa) ===
def rhyme_signature_core(word: str) -> str:
    """
    Signatur für Deduplizierung/Varianz: (Familie des Kernvokals | normalisierte Koda).
    Z.B. 'Mantaplatte' -> 'a|te' (Schwa ignoriert).
    """
    cl, after = last_stressed_vowel_cluster(word or "")
    if not cl:
        return (word or "").lower()
    fam = _vowel_family(cl)
    coda = _normalize_coda(after or "")
    tail = coda[-3:] if len(coda) > 3 else coda
    return f"{fam}|{tail}"
# === ENDE NEU (Phonetik) ===

# === NEU: feine Varianz-Signatur (Onset + letzte Silbe) ===
def rhyme_signature_fine(word: str) -> str:
    """
    Zählt Varianz nach der *ganzen letzten Silbe* (Onset+Vokal+Coda),
    z.B. '...schwanz' vs. '...kranz' vs. '...tanz' getrennt.
    """
    w = (word or "").strip().lower()
    if not w:
        return ""
    
    # NEU: letzte Vokalgruppe + Endkoda sauber ziehen
    m = re.search(r'([^aeiouäöüy]*)([aeiouäöüy]+)([^aeiouäöüy]*)$', w)
    if not m:
        return ""
    onset  = m.group(1) or ""
    nucleus = m.group(2)
    coda    = m.group(3) or ""
    
    # Vokalfamilie normalisieren
    fam = _vowel_family(nucleus)
    
    # leichte Normalisierung
    segment = (onset + coda).replace("ß", "ss")
    return f"{fam}|{segment}"
# === ENDE NEU ===








def vowel_length_compatible(base_word: str, candidate: str) -> bool:
    """
    True, wenn Reimkern (Vokalfamilie + Längenklasse) kompatibel ist.
    """
    b_cluster, b_after = _last_vowel_cluster(base_word)
    c_cluster, c_after = _last_vowel_cluster(candidate)
    if not b_cluster or not c_cluster:
        return True  # keine harte Aussage möglich -> nicht blocken
    b_len = _length_class(b_cluster, b_after or '')
    c_len = _length_class(c_cluster, c_after or '')
    # Diphthonge: nur identischer Diphthong erlaubt
    if b_len == 'diph':
        return c_len == 'diph' and c_cluster == b_cluster
    # Sonst: gleiche Vokalfamilie UND gleiche Längenklasse
    return (_vowel_family(b_cluster) == _vowel_family(c_cluster)) and (b_len == c_len)
# === ENDE NEU ===







# === NEU: Silbenzähler ===
def count_syllables(text: str) -> int:
    """
    Zählt Silben über Vokal-Cluster (aa, ee, oo, uu, ää, öö, üü, ie, ei, ai, au, eu/äu, oi, ui,
    sowie a e i o u ä ö ü y). Schwa-Endungen zählen mit. Bei Phrasen wird summiert.
    """
    if not text:
        return 0
    total = 0
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß'']+", text)
    for tok in tokens:
        w = tok.lower().replace("'", "'")
        clusters = _VOWEL_CLUSTER_RE.findall(w)  # nutzt deine bestehende Regex
        total += len(clusters)
    return total
# === ENDE NEU ===

# --- NEU: strenger Silbenzähler für Wort & Phrase ---
_VOWELS_DE = "aeiouäöüy"

def _count_syllables_word_pyphen(word: str) -> int:
    """Zähle Silben für EIN Wort per Pyphen; Fallback: Heuristik."""
    w = word.lower().replace("'", "'").strip("'")
    if not w:
        return 0
    # Klein-Normalisierung für Kontraktionen, damit Zählung stabil bleibt
    if w in {"'n", "n'", "'n"}:
        w = "nen"
    if w in {"'ne", "'ne"}:
        w = "ne"
    # Pyphen-Zählung
    if _pyphen_de:
        inserted = _pyphen_de.inserted(w)  # z.B. "pol-ter-ab-end"
        if inserted:
            return inserted.count("-") + 1
        # Wenn nichts getrennt wurde, ist es mind. 1 Silbe
        return 1
    # Fallback: sehr simple Heuristik (Vokalgruppen)
    import re
    vgroups = re.findall(rf"[{_VOWELS_DE}]+", w)
    return max(1, len(vgroups))

def count_syllables_strict(text: str) -> int:
    """
    Zähle Silben in EINEM Wort ODER in einer MEHRWORTPHRASE (Summe).
    Nicht-alphabetische Tokens werden ignoriert.
    """
    import re
    # Worte/Token extrahieren (inkl. Umlauten & ' )
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß'']+", text)
    total = 0
    for t in tokens:
        # Bindestriche in zusammengesetzten Wörtern nicht doppelt werten
        t_clean = t.replace("-", "")
        # nur alphabetisch?
        if not re.search(r"[A-Za-zÄÖÜäöüß]", t_clean):
            continue
        total += _count_syllables_word_pyphen(t_clean)
    return total




_CONS_NORM = [
    ("sch", "sh"), ("tsch", "ch"), ("ck", "k"), ("tz", "z"), ("ph", "f"),
]


def rhyme_signature(word: str) -> str:
    """
    Signatur aus Vokalfamilie des LETZTEN Vokals + normalisierter Koda.
    Beispiel: 'Mantaplatte' -> ('a', 'tte' -> 'te') => 'a|te'
    Verhindert Massen an *-ratte/-matte/-latte* (alles ~ 'a|te').
    """
    cl, after = _last_vowel_cluster(word or "")
    if not cl:
        return word.lower()
    fam = _vowel_family(cl)
    coda = _normalize_coda(after or "")
    # nur die letzten 2–3 coda-Zeichen reichen für Varianz
    tail = coda[-3:] if len(coda) > 3 else coda
    return f"{fam}|{tail}"
# === ENDE NEU ===


# === NEU: Wörterbuch-Gate & Phrasen-Validierung ===
_STOPWORDS = {
    "der","die","das","den","dem","des","ein","eine","einen","einem","eines",
    "und","oder","mit","ohne","am","im","an","in","auf","vom","zum","zur",
    "hat","hab","hast","hatte","haben","'nen","nen","nen'","n'","mal","noch","ganz","sehr"
}

def _freq_de(word: str) -> float:
    if not _zipf:
        return 0.0
    try:
        return _zipf(word, "de")
    except Exception:
        return 0.0

# NEU (direkt nach _freq_de):
def _is_prob_name(word: str) -> bool:
    """Grobe Heuristik: Großform deutlich häufiger als Kleinform ⇒ Eigenname."""
    try:
        up = _freq_de(word.title())
        lo = _freq_de(word.lower())
        return (up - lo) >= 0.5
    except Exception:
        return False

def _is_prob_verb_head(stem: str) -> bool:
    """Grobe Heuristik: Infinitiv deutlich häufiger ⇒ eher Verb-Stamm (z.B. 'lauf'→'laufen')."""
    try:
        base = _freq_de(stem)
        inf  = _freq_de(stem + "en")
        return (inf - base) >= 0.7
    except Exception:
        return False

# --- NEU: Heuristiken für Eigennamen/Verbköpfe + Score ---
def _is_probable_proper_name(word: str) -> bool:
    """Groß-/Kleinschreibung stark unterschiedlich häufig => wohl Eigenname."""
    w = (word or "").strip()
    if not w:
        return False
    cap = w[0].upper() + w[1:]
    return (_freq_de(cap) - _freq_de(w)) >= 0.5 and _freq_de(cap) >= 3.0

def _is_probable_verb_head(word: str) -> bool:
    """Infinitiv viel häufiger als Grundform => wohl Verb (z.B. 'plan' vs 'planen')."""
    w = (word or "").strip().lower()
    if not w:
        return False
    return (_freq_de(w + "en") - _freq_de(w)) >= 0.7 and _freq_de(w + "en") >= 2.5

def _lexicon_score_word(word: str) -> float:
    """Score für Ranking/Kappung: Zipf des Gesamtworts; bei Phrase nehmen wir später das letzte Wort."""
    return _freq_de((word or "").strip().lower())
# --- ENDE NEU ---

def is_german_word_or_compound(token: str, freq_thresh: float = 2.5) -> bool:
    """
    True, wenn 'token' (ein Wort, ggf. mit Bindestrich) im Deutschen plausibel ist.
    1) Direktfrequenz via wordfreq.zipf_frequency >= Schwelle
    2)Kompositum-Heuristik: split in zwei Teile (inkl. Fugen-s/-n/-en/-er/-e),
       beide Teile jeweils häufig genug.
    """
    if not token:
        return False
    w = token.lower().strip()

    # 1) Full-word zuerst (strenger): sehr seltene Gesamtwörter raus
    FULL_T = 2.7   # strenger als vorher
    PART_T = 3.0   # Teile müssen wirklich geläufig sein
    if _freq_de(w) >= FULL_T:
        return True

    # Bindestrich-Komposita: alle Teile müssen ok sein
    if "-" in w:
        parts = [p for p in w.split("-") if p]
        return all(is_german_word_or_compound(part, freq_thresh) for part in parts)

    # Zahlen/rohe Abk. verbieten
    if re.fullmatch(r"[0-9]+", w):
        return False

    # 2) Kompositum-Heuristik (zwei Teile)
    for i in range(3, max(3, len(w) - 2)):
        left, right = w[:i], w[i:]

        # Fugen behandeln: linker Stamm evtl. ohne s/n/en/er/e prüfen
        left_variants = {left}
        for f in ("s","n","en","er","e"):
            if left.endswith(f) and len(left) - len(f) >= 3:
                left_variants.add(left[:-len(f)])

        for lv in left_variants:
            left_ok  = (_freq_de(lv)    >= PART_T)
            right_ok = (_freq_de(right) >= PART_T)

            # rechter Kopf darf kein Eigenname und kein Verb-Stamm sein
            if right_ok and (_is_prob_name(right) or _is_prob_verb_head(right)):
                right_ok = False

            if left_ok and right_ok:
                return True

    return False

def is_valid_phrase(text: str, freq_thresh: float = 2.5) -> bool:
    """Mehrwort-Phrase ok, wenn alle Inhaltswörter im Lexikon sind (oder valide Komposita)."""
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß\-]+", text)
    content = [t for t in tokens if t.lower() not in _STOPWORDS]
    if not content:
        return False
    return all(is_german_word_or_compound(t, freq_thresh) for t in content)
# === ENDE NEU (Wörterbuch) ===


def is_phonetically_similar(vowel_seq1: str, vowel_seq2: str) -> bool:
    """
    Prüft, ob zwei Vokalfolgen klangähnlich sind.
    Erlaubt leichte Abweichungen für natürlichere Reime.
    """
    if vowel_seq1 == vowel_seq2:
        return True
    
    # Teile die Vokalfolgen in einzelne Vokale/Diphthonge
    vowels1 = vowel_seq1.split('-')
    vowels2 = vowel_seq2.split('-')
    
    # Wenn die Längen zu unterschiedlich sind, sind sie nicht ähnlich
    if abs(len(vowels1) - len(vowels2)) > 2:
        return False
    
    # Prüfe, ob die Vokale ähnlich klingen
    similar_count = 0
    min_length = min(len(vowels1), len(vowels2))
    
    for i in range(min_length):
        if i < len(vowels1) and i < len(vowels2):
            v1, v2 = vowels1[i], vowels2[i]
            
            # Exakte Übereinstimmung
            if v1 == v2:
                similar_count += 1
            # Ähnliche Vokale (a/ä, o/ö, u/ü, e/i)
            elif (v1 in ['a', 'ä'] and v2 in ['a', 'ä']) or \
                 (v1 in ['o', 'ö'] and v2 in ['o', 'ö']) or \
                 (v1 in ['u', 'ü'] and v2 in ['u', 'ü']) or \
                 (v1 in ['e', 'i'] and v2 in ['e', 'i']):
                similar_count += 0.8  # Teilweise ähnlich
            # Diphthonge mit ähnlichen Komponenten
            elif (v1 == 'au' and v2 in ['a', 'u']) or \
                 (v1 in ['a', 'u'] and v2 == 'au') or \
                 (v1 == 'ei' and v2 in ['e', 'i']) or \
                 (v1 in ['e', 'i'] and v2 == 'ei'):
                similar_count += 0.6  # Leicht ähnlich
    
    # Berechne Ähnlichkeitsgrad
    similarity = similar_count / min_length if min_length > 0 else 0
    
    # Erlaubt Abweichungen bis zu 40%
    return similarity >= 0.6




















def detect_suspicious_response(response: dict) -> bool:
    """
    Erkennt verdächtige Antworten des KI-Trainers.
    Gibt True zurück, wenn die Antwort verdächtig erscheint.
    """
    try:
        # Sehr lockere Validierung - nur offensichtliche Probleme
        reply = response.get('reply', '')
        learning = response.get('learningObject', {})
        
        # Nur bei extremen Fällen warnen
        if not reply or len(reply.strip()) < 5:
            logger.warning("Trainer response too short")
            return True
            
        # Nur bei sehr offensichtlichen Test-Regeln warnen
        if learning and learning.get('rules'):
            rules = learning.get('rules', [])
            for rule in rules:
                title = rule.get('title', '').lower()
                if any(word in title for word in ['test', 'dummy', 'example', 'placeholder', 'lorem']):
                    logger.warning("Trainer rule contains obvious test words")
                    return True
                    
        # Standardmäßig als nicht verdächtig annehmen
        return False
        
    except Exception as e:
        logger.error(f"Error detecting suspicious response: {e}")
        return False


def validate_trainer_response(response: dict, original_messages: list) -> dict:
    """
    Validiert die Antwort des KI-Trainers auf Plausibilität.
    Sehr lockere Validierung für normale Chat-Funktionalität.
    """
    try:
        # Nur bei extrem verdächtigen Antworten eingreifen
        if detect_suspicious_response(response):
            logger.warning("Trainer response flagged as suspicious - minimal intervention")
            # Nur bei extremen Fällen das learningObject entfernen
            if 'learningObject' in response:
                # Prüfe, ob es offensichtlich eine Test-Regel ist
                learning = response.get('learningObject', {})
                if learning and learning.get('rules'):
                    rules = learning.get('rules', [])
                    for rule in rules:
                        title = rule.get('title', '').lower()
                        if any(word in title for word in ['test', 'dummy', 'example', 'placeholder']):
                            response.pop('learningObject')
                            break
        
        # Ansonsten: Antwort unverändert zurückgeben
        return response
        
    except Exception as e:
        logger.error(f"Error validating trainer response: {e}")
        return response


# === API Endpunkte ===

@app.route('/')
def index():
    return "Gallant's Lyric Machine Backend is running!"


@app.route('/api/analyze-audio', methods=['POST'])
def analyze_audio_endpoint():
    """
    Schritt 1 der Analyse: Transkribiert Audio und führt eine schnelle Voranalyse durch.
    Verwendet die korrekte Multimodalität durch Datei-Upload auf Google-Server.
    """
    import tempfile
    import os
    import base64
    
    temp_audio_path = None  # Initialisieren für den finally-Block
    
    try:
        data = request.get_json()
        base64_audio = data.get('base64Audio')
        mime_type = data.get('mimeType')

        if not base64_audio or not mime_type:
            return jsonify({"error": "base64Audio und mimeType sind erforderlich."}), 400

        # 1. Base64 dekodieren und in eine temporäre Datei schreiben
        audio_bytes = base64.b64decode(base64_audio)
        
        # Erstellt eine temporäre Datei mit passender Endung
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            temp_audio_path = temp_audio_file.name

        # 2. Datei auf den Google-Server hochladen
        logger.info(f"Uploading audio file '{temp_audio_path}' for song analysis...")
        audio_file = genai.upload_file(path=temp_audio_path, mime_type=mime_type)
        logger.info("File uploaded successfully.")

        # 3. Der spezifische Prompt für die Song-Analyse
        analysis_prompt = """
        Analysiere die angehängte Audiodatei. Führe die folgenden drei Aufgaben aus und gib das Ergebnis als sauberes JSON-Objekt mit exakt diesen Schlüsseln zurück: "lyrics", "performanceStyle", "title".

        1.  **"lyrics"**: Transkribiere den gesamten gesprochenen oder gesungenen Text wortgetreu.
        2.  **"performanceStyle"**: Klassifiziere den Stil der Darbietung als "sung", "rapped", oder "unknown".
        3.  **"title"**: Schlage einen kurzen, passenden Titel für den transkribierten Text vor.

        WICHTIG: Gib deine Antwort AUSSCHLIESSLICH als gültiges JSON aus, das diesem Schema entspricht. Keine zusätzlichen Texte, keine Erklärungen, nur das JSON:
        {
            "lyrics": "Der transkribierte Text...",
            "performanceStyle": "sung",
            "title": "Vorgeschlagener Titel"
        }
        """
        
        # 4. Gemini mit Text-Prompt UND Datei-Referenz aufrufen
        response = gemini_model.generate_content([analysis_prompt, audio_file])

        # 5. Antwort bereinigen und parsen
        parsed_result, parse_err = parse_json_safely(response.text)
        if parse_err:
            logger.error(f"/api/analyze-audio JSON-Fehler: {parse_err}\nRoh: {response.text[:4000]}")
            return jsonify({"error": "Song-Analyse: JSON nicht lesbar.", "detail": parse_err}), 500

        # Schicke die extrahierten Daten an das Frontend zurück
        return jsonify(parsed_result), 200

    except Exception as e:
        logger.error(f"Fehler bei der Gemini Song-Analyse: {e}", exc_info=True)
        return jsonify({"error": "Die KI konnte den Song nicht analysieren."}), 500
    
    finally:
        # 6. Temporäre Datei nach Gebrauch immer löschen
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            logger.info(f"Cleaned up temporary file: {temp_audio_path}")


# === ZU ERSETZENDER BLOCK ===

@app.route('/api/deep-analyze', methods=['POST'])
def deep_analyze_endpoint():
    """
    Schritt 3 der Analyse: Führt die zentrale Analyse aus, um Stil- und Technik-Merkmale zu extrahieren.
    """
    try:
        data = request.get_json()
        lyrics = data['lyrics']
        # knowledge_base = data['knowledgeBase'] # ENTFERNT
        title = data.get('title', 'Unbenannte Analyse')

        prompt = f"""
        **System Instruction: You are a world-class music analyst. Your task is to perform a deep analysis of the provided lyrics.**

        **LYRICS TO ANALYZE:**
        ---
        {lyrics}
        ---

        **Perform the following analyses and return a single JSON object with the specified keys:**

        1.  **`formattedLyrics`**: Structure the lyrics by adding `[Verse]`, `[Chorus]`, etc., tags where appropriate.
        2.  **`characterTraits`**: Derive an array of 3-4 key themes, moods, or character traits as concise strings.
        3.  **`technicalSkills`**: Identify an array of 3-4 specific songwriting techniques (e.g., "Metaphor usage", "AABB rhyme scheme", "Narrative perspective") as concise strings.
        4.  **`emphasisPattern`**: Provide a detailed descriptive text analyzing the phonetic patterns, stressed syllables, and rhythmic emphasis throughout the lyrics. Focus on how words are emphasized and how this creates the song's rhythm.
        5.  **`rhymeFlowPattern`**: Provide a detailed descriptive text analyzing the rhyme scheme's effect on the song's flow, including the pattern of rhymes, how they connect verses and choruses, and the overall musical structure created by the rhyming.
        
        **IMPORTANT:** Both `emphasisPattern` and `rhymeFlowPattern` should be comprehensive, descriptive analyses that can be used to understand and replicate a unique style in future songwriting.
        """

        schema = {
            "type": "object",
            "properties": {
                "formattedLyrics": {"type": "string"},
                "characterTraits": {"type": "array", "items": {"type": "string"}},
                "technicalSkills": {"type": "array", "items": {"type": "string"}},
                "emphasisPattern": {"type": "string"},
                "rhymeFlowPattern": {"type": "string"}
            },
            "required": ["formattedLyrics", "characterTraits", "technicalSkills", "emphasisPattern", "rhymeFlowPattern"]
        }

        analysis_result = call_claude(prompt, schema=schema)
        
        # Erstelle die neue Antwortstruktur
        response_data = {
            "id": f"da-{uuid.uuid4().hex}",
            "timestamp": time.time(),
            "type": "deep_analysis_result",
            "sourceTitle": title,
            "data": analysis_result
        }
        
        return jsonify(response_data), 200
    except Exception as e:
        logger.error(f"Error in /api/deep-analyze: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Das ist ein Test
@app.route('/api/trainer-chat', methods=['POST'])
def trainer_chat_endpoint():
    """
    Vereinfachter Trainer: Lernt nur noch allgemeine, textliche Regeln
    und speichert sie als 'rule_category'.
    """
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        knowledge_base = data.get('knowledgeBase', '')

        chat_history_text = ""
        for msg in messages:
            role = "Benutzer" if msg.get('isUser') else "KI"
            chat_history_text += f"{role}: {msg.get('text')}\n"

        prompt = f"""
        **Systemanweisung: Du bist die 'Lyric Machine' KI, ein präziser Trainer. Deine Aufgabe ist es, aus dem Gespräch mit dem Künstler zu lernen.**
        **ANTWORTE AUSSCHLIESSLICH AUF DEUTSCH.**
        **DEINE ANTWORT MUSS EIN EINZELNES JSON-OBJEKT SEIN.**
        
        **WICHTIGE REGELN:**
        - Reagiere NUR auf die letzte Nachricht des Benutzers
        - Erfinde KEINE Details, die nicht explizit erwähnt wurden
        - Spekuliere NICHT über Hintergründe oder Motivationen
        - Bleibe bei den konkreten, geäußerten Inhalten
        - Wenn du unsicher bist, frage nach oder gib eine vorsichtige Antwort
        - Antworte direkt auf das, was der Benutzer gesagt hat

        **CHAT-HISTORIE:**
        {chat_history_text}

        **AUFGABE:**
        1. **ANALYSIERE** die letzte Nachricht des Benutzers (die unterste Nachricht in der Chat-Historie)
        2. **ANTWORTE** direkt auf diese letzte Nachricht im "reply"-Feld
        3. **LERNEN** nur, wenn der Benutzer EXPLIZIT einen Lerninhalt beibringt

        **DREI ARTEN VON BEFEHLEN, DIE DU ERKENNEN KANNST:**

        **1. KOMPLEXE REGELN (rule_category):**
        - Wenn der Benutzer eine komplexe Regel definiert (z.B. "NEUE REGEL: Bei Dreisilbigen Reimen...")
        - Erstelle ein learningObject mit type: "rule_category"
        - Beispiel: "NEUE REGEL: Bei Dreisilbigen Reimen darf sich der mittlere Vokal unterscheiden, wenn der erste und letzte Vokal identisch sind."

        **2. NEUE STILE (style):**
        - Wenn der Benutzer einen neuen Stil definiert (z.B. "Definiere neuen Stil: Bildliche Vergleiche")
        - Erstelle ein learningObject mit type: "style"
        - Beispiel: "Definiere neuen Stil: Bildliche Vergleiche"

        **3. NEUE TECHNIKEN (technique):**
        - Wenn der Benutzer eine neue Technik definiert (z.B. "Neue Technik: Tempowechsel im Flow")
        - Erstelle ein learningObject mit type: "technique"
        - Beispiel: "Neue Technik: Tempowechsel im Flow"

        **WICHTIG:** 
        - Wenn der Benutzer nur eine Frage stellt oder etwas erzählt, antworte normal und lerne KEINE Regel
        - Nur wenn der Benutzer explizit einen der obigen Befehle verwendet, erstelle ein learningObject
        - Antworte immer direkt auf die letzte Nachricht, nicht auf frühere Nachrichten

        **BEISPIEL FÜR VERSCHIEDENE LERN-TYPEN:**
        - Benutzer: "Bist du da?" → Antwort: "Ja, ich bin hier! Wie kann ich dir helfen?"
        - Benutzer: "NEUE REGEL: Bei Dreisilbigen Reimen..." → learningObject mit type: "rule_category"
        - Benutzer: "Definiere neuen Stil: Bildliche Vergleiche" → learningObject mit type: "style"
        - Benutzer: "Neue Technik: Tempowechsel im Flow" → learningObject mit type: "technique"

        **STRUKTUR DES LEARNINGOBJECTS:**
        Je nach Typ des Lerninhalts erstelle ein learningObject mit der entsprechenden Struktur:
        
        **Für rule_category:**
        - type: "rule_category"
        - categoryTitle: Ein kurzer, beschreibender Titel für die Regelkategorie
        - rules: Ein Array mit mindestens einer Regel, jede Regel hat:
          * title: Kurzer Titel der Regel
          * definition: Vollständige Definition der Regel
        
        **Für style:**
        - type: "style"
        - content: Der Inhalt des neuen Stils
        
        **Für technique:**
        - type: "technique"
        - content: Der Inhalt der neuen Technik
        """

        # Erweitertes Schema, das alle drei Lern-Typen unterstützt
        trainer_response_schema = {
            "type": "object",
            "properties": {
                "reply": {
                    "type": "string",
                    "description": "Deine textliche Antwort an den Benutzer für den Chat."
                },
                "learningObject": {
                    "type": "object",
                    "description": "Ein Lernobjekt. NUR ausgeben, wenn ein Lerninhalt beigebracht wurde.",
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["rule_category"]},
                                "categoryTitle": {"type": "string"},
                                "rules": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                            "definition": {"type": "string"}
                                        }, "required": ["title", "definition"]
                                    }
                                }
                            }, "required": ["type", "categoryTitle", "rules"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["style"]},
                                "content": {"type": "string"}
                            }, "required": ["type", "content"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["technique"]},
                                "content": {"type": "string"}
                            }, "required": ["type", "content"]
                        }
                    ]
                }
            }, "required": ["reply"]
        }

        result = call_claude(prompt, schema=trainer_response_schema)

        # Validiere die Antwort auf Plausibilität
        validated_result = validate_trainer_response(result, messages)

        learning_object = validated_result.get('learningObject')
        if learning_object:
            # Füge die IDs basierend auf dem Typ hinzu
            if learning_object.get('type') == 'rule_category':
                learning_object['id'] = f"rcat-{int(time.time() * 1000)}-{random.randint(100, 999)}"
                for rule in learning_object.get('rules', []):
                    rule['id'] = f"rule-{int(time.time() * 1000)}-{random.randint(100, 999)}"
            elif learning_object.get('type') == 'style':
                learning_object['id'] = f"style-{int(time.time() * 1000)}-{random.randint(100, 999)}"
            elif learning_object.get('type') == 'technique':
                learning_object['id'] = f"technique-{int(time.time() * 1000)}-{random.randint(100, 999)}"
            
            # Sende nur das learning Feld, wenn ein Lerninhalt beigebracht wurde
            return jsonify({"reply": validated_result.get('reply'), "learning": learning_object}), 200
        else:
            # Keine Regel gelernt - sende kein learning Feld
            return jsonify({"reply": validated_result.get('reply')}), 200

    except Exception as e:
        logger.error(f"Error in /api/trainer-chat: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/rhymes', methods=['POST'])
def find_rhymes_endpoint():
    """
    Korrigierte Version: Verwendet den exakten zweistufigen Prozess aus test_models.py
    MIT den neuen sprachlichen Verbesserungen im Prompt
    """
    logger.info(">>> /api/rhymes LOADED (patch v1)")
    try:
        data = request.get_json()
        input_word = data.get('input_word') or data.get('input')
        knowledge_base = data.get('knowledge_base', 'Keine DNA vorhanden.')
        max_words = int(data.get('max_words', 4))
        N_MIN = int(data.get('min_results', 12))  # NEU: Mindestanzahl gewünschter Treffer
        DEBUG_VALIDATION = bool(data.get('debug', False))  # NEU: Detail-Logs (ablehnungsgründe)
        phrase_only_needed = bool(data.get('phrase_only', False))  # NEU: Nur Phrasen generieren

        # Nutzer-Parameter / Defaults
        TARGET_PHRASE_RATIO = float(data.get('target_phrase_ratio', 0.30))
        MAX_PER_FAMILY     = int(data.get('max_per_family', 2))
        MAX_RESULTS        = int(data.get('max_results', 12))

        # Basis-Schwa & Silben
        base_syllables = count_syllables_strict(input_word)
        base_schwa     = get_schwa_suffix(input_word)

        # NEU: Start des Loggings für diese Anfrage
        logger.info("="*50)
        logger.info(f" NEUE REIM-ANFRAGE FÜR: '{input_word}' ")
        logger.info("="*50)

        logger.info(f"--- Reim-Anfrage gestartet für: '{input_word}' ---")
        if not input_word:
            logger.error("Fehler: Kein 'input_word' im Request gefunden.")
            return jsonify({"error": "input_word is missing"}), 400

        # === PRÄ-ANALYSE FÜR EINEN PRÄZISEN PROMPT ===
        target_analysis = get_phonetic_breakdown_py(input_word)
        target_syllables = target_analysis.get("syllableCount")
        target_vowels = target_analysis.get("vowelSequence")
        
        # NEU: Logging der Voranalyse
        logger.info(f"--> Schritt 0: Phonetik-Voranalyse (Python-Skript)")
        logger.info(f"    - Berechnete Silben: {target_syllables}")
        logger.info(f"    - Berechnete Vokalfolge: '{target_vowels}'")
        
        stress_pattern = target_analysis.get("stressPattern")
        stressed_vowels = target_analysis.get("stressedVowelSequence")
        logger.info(f"Vor-Analyse für '{input_word}': Silben={target_syllables}, Vokale='{target_vowels}', Betonung='{stress_pattern}'")

        # === SCHRITT 1: KREATIVE GENERIERUNG (OHNE SCHEMA-ZWANG) ===
        # NEU: Intelligente, gelockerte Regel mit weicherer Formulierung
        if target_syllables >= 4:
            syllable_rule = f"""4. **Hohe klangliche Ähnlichkeit (Fokus auf Betonung):**
   - Der Reim muss klanglich sehr nah am Original sein. **Orientiere dich stark** an der Vokalsequenz der betonten Silben ('{stressed_vowels}') und dem Rhythmus ('{stress_pattern}').
   - **Leichte Abweichungen sind erlaubt**, wenn der Reim dadurch kreativer und natürlicher wird. Die Priorität liegt auf einem gut klingenden, sinnvollen Ergebnis."""
        else:
            syllable_rule = f"4. **Exakte Vokalfolge:** Der Reim muss exakt die Vokalfolge '{target_vowels}' haben."



        # --- Kurz-Prompt überschreibt den langen Prompt ---
        fam_cluster, fam_after = last_stressed_vowel_cluster(input_word)
        len_class = _length_class(fam_cluster, fam_after or '') if fam_cluster else ''
        first_vowel_family = _vowel_family(_first_vowel_cluster(input_word) or '') if _first_vowel_cluster(input_word) else ''

        # Dynamische Phrasenquote basierend auf TARGET_PHRASE_RATIO
        target_phrase_pct = int(TARGET_PHRASE_RATIO * 100)
        target_phrase_ratio = float(TARGET_PHRASE_RATIO)
        phrase_rule = (f"6) Mindestens {target_phrase_pct}% Mehrwortphrasen (2–5 Wörter); Rest echte Einwort-Wörter/Komposita."
                       if TARGET_PHRASE_RATIO < 0.5
                       else "6) Bevorzuge Mehrwortphrasen (2–5 Wörter).")

        # --- NEU: Prompt-Variablen aus deiner Phonetik-Analyse (alles generisch) ---
        # Falls ein Feld in target_analysis mal fehlt, greifen sinnvolle Defaults.
        first_vowel_family  = target_analysis.get("first_vowel_family", "")
        first_vowel_length  = target_analysis.get("first_vowel_length", len_class)  # z.B. "short"/"long"
        last_vowel_family   = target_analysis.get("last_vowel_family", "")
        last_vowel_length   = target_analysis.get("last_vowel_length", len_class)
        last_coda_family_hint = target_analysis.get(
            "last_coda_family_hint",
            "ähnliche Koda-Struktur (gleiche Artikulationsstelle; stimmhaft/stimmlos Variante ok)"
        )

        core_vowel_family   = target_analysis.get("core_vowel_family")
        core_vowel_length   = target_analysis.get("core_vowel_length")

        if core_vowel_family:
            core_rule       = f"Zwischen erstem und letztem Vokal muss **genau einmal** ein betonter Vokal der Familie {core_vowel_family} (Länge={core_vowel_length}) auftreten."
            core_rule_hint  = "Dieser Kern liegt syllabisch vor der letzten Silbe."
            core_selfcheck  = f"den inneren betonten {core_vowel_family}-Kern nicht enthält"
        else:
            core_rule       = "Kein zusätzlicher innerer Kern erforderlich."
            core_rule_hint  = ""
            core_selfcheck  = "False"   # Platzhalter -> fällt im Prompt als Bedingung faktisch weg

        creative_prompt = f"""
Du bist ein deutscher Reim-Coach. Erzeuge starke, natürliche Reimkandidaten für „{input_word}".

REGELN (unbedingt einhalten):
1) **Ausgabeform**: Gib **nur JSON** zurück: {{"candidates":["...","..."]}}. Keine Erklärungen, keine Gedanken, nichts davor oder danach.
2) **Silbenzahl**: Jeder Kandidat hat **genau {base_syllables} Silben**.
3) **Vokal-Anker (Familie/Längenklasse)**:
   - **Erster Vokal**: Familie={first_vowel_family}, Länge={first_vowel_length} – muss übereinstimmen.
   - **Letzter Vokal**: Familie={last_vowel_family}, Länge={last_vowel_length} – muss übereinstimmen.
   - **Innerer betonter Vokal**: {core_rule}
     {core_rule_hint}
4) **Letzte Silbe**: Behalte den **Nukleus** (Vokalfamilie wie oben). Die **Koda** soll in einer **ähnlichen Koda-Familie** liegen ({last_coda_family_hint}). Kleine stimmhaft/stimmlos-Wechsel in **derselben Artikulationsstelle** sind ok (z. B. d/t, s/z). **Kein Wechsel der Vokalfamilie.**
5) **Deutsch**: Nur echte deutsche Wörter oder sehr plausible Komposita/Mehrwortphrasen.
6) **Vielfalt**: Keine Serien gleicher Präfixe (max. 2 Kandidaten mit identischem Anfang). Keine Fantasieendungen.
7) **Phrasenanteil**: **{target_phrase_pct}% Mehrwortphrasen (2–5 Wörter)**, der Rest **Einwortkandidaten**.
8) **Selbstcheck vor Ausgabe**: Entferne jeden Kandidaten, der
   - nicht exakt {base_syllables} Silben hat,
   - beim **ersten** oder **letzten** Vokal die **Vokalfamilie/Längenklasse** ändert,
   - {core_selfcheck},
   - oder offensichtlich kein deutsches Wort/keine plausible Phrase ist.

AUSGABE:
- Liefere **bis zu {max_results}** hochwertige, unterschiedliche Kandidaten.
- **Nur** dieses JSON-Objekt: {{"candidates":["...","..."]}}
"""



        logger.info(f"Rhyme-Gen: Schritt 1 - Kreative Generierung für '{input_word}'...")
        
        # NEU: Logging des Kreativ-Prompts
        logger.info("--> Schritt 1: Sende KREATIV-PROMPT an die KI...")
        logger.info(f"===== START KREATIV-PROMPT =====\n{creative_prompt}\n===== ENDE KREATIV-PROMPT =====")
        
        try:
            # NEU: Schema + Provider-Umschalter (JSON-only)
            creative_schema = {
                "type": "object",
                "properties": { "candidates": { "type": "array", "items": { "type": "string" }, "minItems": 10, "maxItems": 30 } },
                "required": ["candidates"]
            }

            if RHYME_PROVIDER == 'gpt' and openai_client:
                gpt_obj = call_gpt_candidates(creative_prompt, schema=creative_schema)
                creative_output = "\n".join([c for c in gpt_obj.get("candidates", []) if isinstance(c, str)])
            else:
                # Fallback/Default: Claude (funktioniert wie zuvor)
                creative_response = anthropic_client.messages.create(
                    model="claude-opus-4-1-20250805",
                    max_tokens=2048,
                    temperature=0.7,
                    messages=[{"role": "user", "content": creative_prompt + "\n\nANTWORTFORMAT:\n" + json.dumps(creative_schema, ensure_ascii=False)}]
                )
                # Hinweis: Falls Claude doch Text drumherum setzt, nutzen wir den Roh-Text;
                # euer bestehendes Post-Processing filtert ihn ohnehin weg.
                creative_output = creative_response.content[0].text
            
            # Logging der Kreativ-Antwort
            logger.info("<-- Schritt 1: KREATIVE ANTWORT von KI erhalten.")
            logger.info(f"===== START KREATIV-ANTWORT (ROH) =====\n{creative_output}\n===== ENDE KREATIV-ANTWORT (ROH) =====")

            # Post-Processing direkt hier durchführen
            valid_candidates = []
            # NEU: Erst versuchen, JSON direkt zu lesen
            obj, perr = parse_json_safely(creative_output)
            if not perr and isinstance(obj, dict) and isinstance(obj.get('candidates'), list):
                lines = [s for s in obj['candidates'] if isinstance(s, str)]
            else:
                # Fallback: auf Zeilenebene weiterarbeiten
                lines = creative_output.splitlines()
            
            # NEU: Robuste Schleife, die "Junk"-Wörter UND KI-Kommentare ignoriert
            valid_candidates = []
            seen_signatures = {}  # NEU: Varianz-Zähler je Reimfamilie
            base_schwa = get_schwa_suffix(input_word)  # NEU: sichtbares Suffix des Basisworts
            ignore_keywords = [
                'analyse', 'analysier', 'denkschritt', 'mehrwort', 'finale liste',
                'silben', 'vokal', 'ergebnis', 'reimsammlung', 'qualitäts', 'iteration',
                'regel', 'prüfung', 'schema', 'format', 'json', 'beispiel', 'beschreibung', 'erklärung',
                'alle sind deutsche wörter', 'keine beginnt mit gegen'
            ]

            for line in lines:
                line = line.strip()
                
                if not line or any(keyword in line.lower() for keyword in ignore_keywords):
                    continue

                # Überschriften/Meta (##, #, Aufzählungs-Linien) überspringen
                if line.startswith('#') or line.startswith('##') or re.match(r'^(?:[-=]{3,}|>{1,}|\[.+\]:)', line):
                    continue
                # Brackets/Links/Code-Reste überspringen
                if any(ch in line for ch in ['{','}','[',']','`']) or 'http' in line.lower():
                    continue

                cleaned_line = re.sub(r'^[-*•]\s*', '', line)
                cleaned_line = re.sub(r'^\d+\.\s*', '', cleaned_line)
                cleaned_line = cleaned_line.strip().replace('*', '')

                # Hängende Satzzeichen am Ende entfernen
                cleaned_line = re.sub(r'[,:;–—-]+$', '', cleaned_line).strip()

                # NEU: Trenne den Reim von der KI-Beschreibung (alles nach dem Bindestrich)
                if " - " in cleaned_line:
                    cleaned_line = cleaned_line.split(" - ")[0].strip()
                
                # Finale Prüfung mit der jetzt sauberen Phrase
                if not cleaned_line or ':' in cleaned_line:
                    continue

                # Phonetik
                if get_schwa_suffix(cleaned_line) != base_schwa:
                    continue
                if not first_vowel_family_match(input_word, cleaned_line):
                    continue
                if not vowel_core_compatible(input_word, cleaned_line):
                    continue
                if count_syllables(cleaned_line) != base_syllables:
                    continue

                # Varianz (feine Familie, z.B. -schwanz/-kranz getrennt zählen)
                sig = rhyme_signature_fine(cleaned_line)
                if seen_signatures.get(sig, 0) >= MAX_PER_FAMILY:
                    continue
                seen_signatures[sig] = seen_signatures.get(sig, 0) + 1

                # Lexikon-Gate
                is_phrase = bool(re.search(r"\s", cleaned_line))
                if is_phrase:
                    if not is_valid_phrase(cleaned_line):
                        continue
                else:
                    if not is_german_word_or_compound(cleaned_line):
                        continue

                valid_candidates.append(cleaned_line)

            valid_candidates = list(dict.fromkeys(valid_candidates))[:20]

            logger.info(f"--> Schritt 2: Post-Processing")
            logger.info(f"    - {len(valid_candidates)} valide Kandidaten aus der Antwort extrahiert.")
            logger.info(f"    - Extrahierte Kandidaten: {valid_candidates}")

            # --- NEU: strenges Silben-Gate für Run 1 ---
            raw_count = len(valid_candidates)
            syllable_ok = []
            if DEBUG_VALIDATION:
                rejects = []

            for c in valid_candidates:
                syl = count_syllables_strict(c)
                if syl == base_syllables:
                    syllable_ok.append(c)
                else:
                    if DEBUG_VALIDATION:
                        rejects.append((c, syl))

            if DEBUG_VALIDATION and rejects:
                for c, syl in rejects:
                    logger.info(f"/api/rhymes Pass1 reject (syll): '{c}' -> {syl} vs base {base_syllables}")

            logger.info(f"/api/rhymes Pass1 (Silben): base_syll={base_syllables}, after_syll_eq={len(syllable_ok)} / raw={raw_count}")

            # Ab hier NUR noch mit den Silben-korrekten Kandidaten weiterarbeiten
            valid_candidates = syllable_ok

            # NEU: Pass-1 Kurzinfo
            fam_cluster, fam_after = last_stressed_vowel_cluster(input_word)
            len_class = _length_class(fam_cluster, fam_after or '') if fam_cluster else ''
            phrase_count = sum(1 for vc in valid_candidates if ' ' in vc)
            logger.info(
                f"/api/rhymes Pass1: valid={len(valid_candidates)}, phrases={phrase_count}, "
                f"ratio={(phrase_count / max(1, len(valid_candidates))):.2f}, "
                f"base_syll={base_syllables}, schwa='{base_schwa}', len_class='{len_class}', "
                f"families={sorted(seen_signatures.keys())}"
            )
            
            before_set = set(valid_candidates)  # NEU: zum Vergleichen nach dem Second Pass
            
            # === NEU: Reject-Sampling, wenn zu wenige Treffer ===
            if len(valid_candidates) < N_MIN:
                logger.info(f"/api/rhymes: Nur {len(valid_candidates)} Treffer – starte Second Pass (Reject-Sampling).")

                # --- NEU: Falls zu wenig Phrasen, Pass 2 auf "nur Phrasen" stellen ---
                need_phrases = (not phrase_only_needed) and (TARGET_PHRASE_RATIO > 0.0)
                current_phrases = sum(1 for x in valid_candidates if ' ' in x)  # 'valid_candidates' = Liste nach Pass 1
                need_phrase_only_mode = need_phrases and (current_phrases < int(round(TARGET_PHRASE_RATIO * MAX_RESULTS)))
                logger.info(f"/api/rhymes Second Pass mode: {'phrases-only' if need_phrase_only_mode else 'mixed'} (current_phrases={current_phrases})")

                # Bisherige Reimfamilien (für Blacklist) + benutzte Wörter
                banned_families = sorted(seen_signatures.keys())  # z.B. ['a|te', 'i|nd', ...]
                banned_words = sorted({vc.lower() for vc in valid_candidates})

                # Basis-Merkmale für den Prompt (rein informativ/leitend – harte Prüfung macht weiterhin der Server)
                fam_cluster, fam_after = last_stressed_vowel_cluster(input_word)
                fam_name = _vowel_family(fam_cluster) if fam_cluster else ''
                len_class = _length_class(fam_cluster, fam_after or '') if fam_cluster else ''  # 'short'/'long'/'diph'

                # Phrasenquote-Check für Second Pass
                phrases = [c for c in valid_candidates if re.search(r"\s", c)]
                phrase_only_needed = (len(phrases) < TARGET_PHRASE_RATIO * max(1, len(valid_candidates)))

                # Wir nutzen dasselbe Schema wie im ersten Lauf (JSON-only mit candidates[])
                second_prompt = (
                    f"{creative_prompt}\n\n"
                    "ERGÄNZUNG (zweiter Lauf / Reject-Sampling):\n"
                    f"- Erzeuge 40 NEUE Kandidaten.\n"
                    f"- Vermeide strikt diese Reimfamilien (Kern|Koda): {', '.join(banned_families) if banned_families else '—'}\n"
                    f"- Schließe diese Wörter strikt aus: {', '.join(banned_words) if banned_words else '—'}\n"
                    f"- Halte die Regeln exakt ein (erste Vokalfamilie = '{_vowel_family(_first_vowel_cluster(input_word) or '')}', "
                    f"Kern-Länge/Diphthong = '{len_class}', Schwa-Suffix = '{get_schwa_suffix(input_word)}').\n"
                    f"- EXAKT {base_syllables} Silben pro Kandidat.\n"
                    'ANTWORTFORMAT: Nur JSON-Objekt {"candidates": ["..."]}\n'
                )

                # Dynamische Phrasen-Anweisung basierend auf need_phrase_only_mode
                if need_phrase_only_mode:
                    second_prompt += (
                        "\n- GIB AUSSCHLIESSLICH MEHRWORTPHRASEN (2–5 Wörter).\n"
                        "- KEINE Einwort-Kandidaten.\n"
                        "- Antworte NUR als JSON {\"candidates\": [\"...\"]}.\n"
                        "- Keine Fantasieformen, keine Eigennamen.\n"
                    )
                else:
                    target_phrase_pct = int(TARGET_PHRASE_RATIO * 100)
                    second_prompt += (
                        f"\n- Mindestens {target_phrase_pct}% Mehrwortphrasen (2–5 Wörter).\n"
                        "- Antworte NUR als JSON {\"candidates\": [\"...\"]}.\n"
                    )

                # Provider-Aufruf wie im ersten Lauf, aber mit dem Second-Prompt
                try:
                    if RHYME_PROVIDER == 'gpt' and openai_client:
                        second_obj = call_gpt_candidates(second_prompt, schema=creative_schema)
                        second_lines = [s for s in (second_obj.get("candidates") or []) if isinstance(s, str)]
                    else:
                        second_obj = call_claude(second_prompt, schema=creative_schema)
                        if isinstance(second_obj, dict) and isinstance(second_obj.get("candidates"), list):
                            second_lines = [s for s in second_obj["candidates"] if isinstance(s, str)]
                        else:
                            second_lines = str(second_obj).splitlines()
                except Exception as e:
                    logger.error(f"/api/rhymes Second Pass: Provider-Fehler: {e}")
                    second_lines = []

                # NEU: Second-Pass Stufen-Statistik
                stats = {
                    "raw": len(second_lines),
                    "after_phrase": 0,
                    "after_schwa": 0,
                    "after_first": 0,
                    "after_core": 0,
                    "after_syll": 0,
                    "after_variance": 0,
                    "after_lexicon": 0,  # NEU: Nach Lexikon-Gate
                    "accepted": 0,
                }

                # NEU: Annahmeliste + optionale Rejektgründe
                debug_mode = bool(data.get('debug_validation', False))
                accepted_second = []
                rejected_reasons = []  # nur wenn debug_mode True

                # Gleiche Post-Processing-Pipeline wie im ersten Lauf (nur Zeilenquelle ist jetzt 'second_lines')
                for raw in second_lines:
                    line = (raw or "").strip()
                    reason = None  # NEU: Ablehnungsgrund (nur für Debug-Log)
                    
                    # === NEU: "Nur Phrasen" hart (wenn need_phrase_only_mode=True) ===
                    if need_phrase_only_mode and not re.search(r"\s", line):
                        if debug_mode: rejected_reasons.append((line, "phrase_only"))
                        continue
                    # === ENDE NEU ===
                    
                    if not line or any(keyword in line.lower() for keyword in ignore_keywords):
                        reason = "phrase_only"
                        if DEBUG_VALIDATION: logger.info(f"SP REJECT ({reason}): {line}")
                        if debug_mode: rejected_reasons.append((line, "phrase_only"))
                        continue
                    if line.startswith('#') or line.startswith('##') or re.match(r'^(?:[-=]{3,}|>{1,}|\[.+\]:)', line):
                        reason = "phrase_only"
                        if DEBUG_VALIDATION: logger.info(f"SP REJECT ({reason}): {line}")
                        if debug_mode: rejected_reasons.append((line, "phrase_only"))
                        continue
                    if any(ch in line for ch in ['{','}','[',']','`']) or 'http' in line.lower():
                        reason = "phrase_only"
                        if DEBUG_VALIDATION: logger.info(f"SP REJECT ({reason}): {line}")
                        if debug_mode: rejected_reasons.append((line, "phrase_only"))
                        continue

                    stats["after_phrase"] += 1

                    cleaned_line = re.sub(r'^\s*[-*•]\s*', '', line)
                    if ' - ' in cleaned_line and cleaned_line.count(' ') >= 1:
                        parts = [p.strip() for p in cleaned_line.split(' - ') if p.strip()]
                        if parts and len(parts[0].split()) <= max_words:
                            cleaned_line = parts[0]
                    cleaned_line = cleaned_line.strip().replace('*', '')
                    cleaned_line = re.sub(r'[,:;–—-]+$', '', cleaned_line).strip()

                    if cleaned_line and len(cleaned_line.split()) <= max_words and ':' not in cleaned_line:
                        # a) sichtbares Schwa-Suffix muss identisch sein
                        if get_schwa_suffix(cleaned_line) != base_schwa:
                            reason = "schwa_mismatch"
                            if DEBUG_VALIDATION: logger.info(f"SP REJECT ({reason}): {cleaned_line}")
                            if debug_mode: rejected_reasons.append((cleaned_line, "schwa_mismatch"))
                            continue
                        stats["after_schwa"] += 1
                        # b) erster Vokal (Familie) muss passen
                        if not first_vowel_family_match(input_word, cleaned_line):
                            reason = "first_vowel"
                            if DEBUG_VALIDATION: logger.info(f"SP REJECT ({reason}): {cleaned_line}")
                            if debug_mode: rejected_reasons.append((cleaned_line, "first_vowel"))
                            continue
                        stats["after_first"] += 1
                        # c) Kern (betonter Endvokal): Familie + Länge/Diphthong
                        if not vowel_core_compatible(input_word, cleaned_line):
                            reason = "core"
                            if DEBUG_VALIDATION: logger.info(f"SP REJECT ({reason}): {cleaned_line}")
                            if debug_mode: rejected_reasons.append((cleaned_line, "core"))
                            continue
                        stats["after_core"] += 1
                        # c.5) NEU: Silbenzahl muss exakt der des Basiswortes entsprechen
                        if count_syllables(cleaned_line) != base_syllables:
                            reason = "syllables"
                            if DEBUG_VALIDATION: logger.info(f"SP REJECT ({reason}): {cleaned_line}")
                            if debug_mode: rejected_reasons.append((cleaned_line, "syllables"))
                            continue
                        stats["after_syll"] += 1
                        # d) Varianzlimit: max. 2 pro Reimfamilie (feine Signatur: ganze letzte Silbe)
                        sig = rhyme_signature_fine(cleaned_line)
                        if seen_signatures.get(sig, 0) >= MAX_PER_FAMILY:
                            reason = "variance"
                            if DEBUG_VALIDATION: logger.info(f"SP REJECT ({reason}): {cleaned_line}")
                            if debug_mode: rejected_reasons.append((cleaned_line, "variance"))
                            continue
                        seen_signatures[sig] = seen_signatures.get(sig, 0) + 1

                        # === LEXIKON-GATE: Direkte Validierung vor dem append() ===
                        is_phrase = bool(re.search(r"\s", cleaned_line))
                        if is_phrase:
                            if not is_valid_phrase(cleaned_line, freq_thresh=2.5):
                                if debug_mode: rejected_reasons.append((cleaned_line, "lexicon"))
                                continue
                        else:
                            if not is_german_word_or_compound(cleaned_line, freq_thresh=2.5):
                                if debug_mode: rejected_reasons.append((cleaned_line, "lexicon"))
                                continue
                        
                        stats["after_lexicon"] += 1
                        stats["after_variance"] += 1
                        stats["accepted"] += 1

                        accepted_second.append(cleaned_line)
                        valid_candidates.append(cleaned_line)

                logger.info(
                    "/api/rhymes Second Pass stats: "
                    f"raw={stats['raw']}, after_phrase={stats['after_phrase']}, "
                    f"after_schwa={stats['after_schwa']}, after_first={stats['after_first']}, "
                    f"after_core={stats['after_core']}, after_syll={stats['after_syll']}, "
                    f"after_variance={stats['after_variance']}, after_lexicon={stats['after_lexicon']}, "
                    f"accepted={stats['accepted']}"
                )
                
                logger.info(f"/api/rhymes Second Pass accepted: {accepted_second}")
                if debug_mode:
                    # Zeige maximal 60 Rejekte, damit das Log lesbar bleibt
                    logger.info(f"/api/rhymes Second Pass rejected (first 60): {rejected_reasons[:60]}")
                
                # NEU: "Wer ist neu?" - Analyse
                after_set = set(valid_candidates)
                new_candidates = after_set - before_set
                if new_candidates:
                    logger.info(f"/api/rhymes Second Pass: Neue Kandidaten: {sorted(new_candidates)}")
                
                logger.info(f"/api/rhymes Second Pass: +{max(0, len(valid_candidates)-len(banned_words))} neue Kandidaten (gesamt: {len(valid_candidates)}).")
                
                # === NEU: Pass 3 – Diversity-Füller, wenn noch zu wenig ===
                if len(valid_candidates) < MAX_RESULTS:
                    logger.info(f"/api/rhymes: {len(valid_candidates)} < {MAX_RESULTS} – starte Pass 3 (Diversity).")

                    # Bereits verwendete *coarse* Familien sperren (z.B. 'a|nz')
                    used_coarse = set()
                    for vc in valid_candidates:
                        used_coarse.add(rhyme_signature_core(vc))

                    third_prompt = (
                        f"{creative_prompt}\n\n"
                        "ERGÄNZUNG (Pass 3 / Diversity):\n"
                        "- Erzeuge 30 NEUE Kandidaten.\n"
                        f"- Vermeide strikt diese Reimfamilien (Kern|Koda): {', '.join(sorted(used_coarse)) or '—'}\n"
                        f"- EXAKT {base_syllables} Silben. Schwa-Suffix = '{base_schwa}'.\n"
                        "- Antworte NUR als JSON {\"candidates\": [\"...\"]}\n"
                        + ("- GIB AUSSCHLIESSLICH PHRASEN (2–5 Wörter).\n" if phrase_only_needed else "")
                    )

                    try:
                        if RHYME_PROVIDER == 'gpt' and openai_client:
                            third_obj = call_gpt_candidates(third_prompt, schema=creative_schema)
                            third_lines = [s for s in (third_obj.get("candidates") or []) if isinstance(s, str)]
                        else:
                            third_obj = call_claude(third_prompt, schema=creative_schema)
                            if isinstance(third_obj, dict) and isinstance(third_obj.get("candidates"), list):
                                third_lines = [s for s in third_obj["candidates"] if isinstance(s, str)]
                            else:
                                third_lines = str(third_obj).splitlines()
                    except Exception as e:
                        logger.error(f"/api/rhymes Pass 3: Provider-Fehler: {e}")
                        third_lines = []

                    accepted_third = []
                    for raw in third_lines:
                        line = (raw or "").strip()
                        if not line or any(k in line.lower() for k in ignore_keywords):
                            continue
                        cleaned_line = re.sub(r'^\s*[-*•]\s*', '', line).strip()
                        cleaned_line = re.sub(r'[,:;–—-]+$', '', cleaned_line).strip()

                        # gleiche Filter wie in Pass 2
                        if phrase_only_needed and not re.search(r"\s", cleaned_line): 
                            continue
                        if get_schwa_suffix(cleaned_line) != base_schwa: 
                            continue
                        if not first_vowel_family_match(input_word, cleaned_line): 
                            continue
                        if not vowel_core_compatible(input_word, cleaned_line): 
                            continue
                        if count_syllables(cleaned_line) != base_syllables: 
                            continue
                        # harte Sperre gegen *coarse* Familien
                        if rhyme_signature_core(cleaned_line) in used_coarse: 
                            continue
                        # Varianz auf *fine* Familie
                        sig_f = rhyme_signature_fine(cleaned_line)
                        if seen_signatures.get(sig_f, 0) >= MAX_PER_FAMILY: 
                            continue
                        seen_signatures[sig_f] = seen_signatures.get(sig_f, 0) + 1

                        # Lexikon-Gate
                        if re.search(r"\s", cleaned_line):
                            if not is_valid_phrase(cleaned_line): 
                                continue
                        else:
                            if not is_german_word_or_compound(cleaned_line): 
                                continue

                        valid_candidates.append(cleaned_line)
                        accepted_third.append(cleaned_line)

                        if len(valid_candidates) >= MAX_RESULTS:
                            break

                    logger.info(f"/api/rhymes Pass 3 accepted: {accepted_third}")
                # === ENDE NEU ===
                

                
                # --- Cap & Auswahl (12 Elemente) ---
                def _is_phrase(s: str) -> bool:
                    return bool(re.search(r"\s", s))

                def _lex_score(s: str) -> float:
                    # einfache Häufigkeitsschätzung: letztes Wort bei Phrasen, Ganzwort sonst
                    try:
                        from wordfreq import zipf_frequency as _zipf
                    except Exception:
                        return 0.0
                    if _is_phrase(s):
                        parts = re.findall(r"[A-Za-zÄÖÜäöüß\-]+", s)
                        last = (parts[-1] if parts else s).lower()
                        return _zipf(last, "de")
                    return _zipf((s or "").lower(), "de")

                # Finale Liste für die Auswahl vorbereiten (Duplikate entfernen)
                valid_candidates = list(dict.fromkeys(valid_candidates))  # Duplikate raus

                phrases = [c for c in valid_candidates if _is_phrase(c)]
                singles = [c for c in valid_candidates if not _is_phrase(c)]
                phrases.sort(key=_lex_score, reverse=True)
                singles.sort(key=_lex_score, reverse=True)

                target_phr = int((TARGET_PHRASE_RATIO * MAX_RESULTS + 0.999))  # ceil
                sel_phr = phrases[:target_phr]
                sel_sgl = singles[: max(0, MAX_RESULTS - len(sel_phr))]
                if len(sel_phr) + len(sel_sgl) < MAX_RESULTS:
                    need = MAX_RESULTS - (len(sel_phr) + len(sel_sgl))
                    # aus der jeweils anderen Gruppe auffüllen
                    if len(sel_phr) < target_phr:
                        sel_phr += phrases[len(sel_phr): len(sel_phr)+need]
                    else:
                        sel_sgl += singles[len(sel_sgl): len(sel_sgl)+need]

                final_list = (sel_phr + sel_sgl)[:MAX_RESULTS]
                
                # --- NEU: finale Auswahl mit Familienlimit + Phrasenquote (strenger) ---
                from collections import defaultdict

                pool = valid_candidates  # Liste nach allen Gates
                family_count = defaultdict(int)

                def fam(x: str) -> str:
                    return rhyme_signature_fine(x)  # trennt z.B. -kranz/-tanz/-schwanz

                target_phr = 0 if phrase_only_needed else max(0, int(round(TARGET_PHRASE_RATIO * MAX_RESULTS)))

                phr_queue  = [c for c in pool if ' ' in c]
                sing_queue = [c for c in pool if ' ' not in c]

                chosen = []

                # 1) Phrasen bis Zielquote (mit Familienlimit)
                for c in list(phr_queue):
                    if sum(1 for y in chosen if ' ' in y) >= target_phr: break
                    f = fam(c)
                    if family_count[f] >= MAX_PER_FAMILY: continue
                    chosen.append(c); family_count[f] += 1
                    if len(chosen) >= MAX_RESULTS: break

                # 2) Singles auffüllen (mit Familienlimit)
                for c in list(sing_queue):
                    if len(chosen) >= MAX_RESULTS: break
                    f = fam(c)
                    if family_count[f] >= MAX_PER_FAMILY: continue
                    chosen.append(c); family_count[f] += 1

                # 3) Rest ggf. wieder mit Phrasen (mit Familienlimit)
                if len(chosen) < MAX_RESULTS:
                    for c in phr_queue:
                        if len(chosen) >= MAX_RESULTS: break
                        if c in chosen: continue
                        f = fam(c)
                        if family_count[f] >= MAX_PER_FAMILY: continue
                        chosen.append(c); family_count[f] += 1

                phr_ct = sum(1 for x in chosen if ' ' in x)
                sing_ct = len(chosen) - phr_ct
                logger.info(f"/api/rhymes Cap (neu): target={MAX_RESULTS}, chosen={len(chosen)} (phrases={phr_ct}, singles={sing_ct}), max_per_family={MAX_PER_FAMILY}, target_phrase_ratio={TARGET_PHRASE_RATIO}")
                
                logger.info(f"/api/rhymes Cap: target={MAX_RESULTS}, chosen={len(chosen)} (phrases={len([x for x in chosen if ' ' in x])}, singles={len([x for x in chosen if ' ' not in x])})")
                logger.info(f"/api/rhymes Accepted (final {len(chosen)}): {chosen}")

                # Finale Antwort nach dem Second Pass neu erstellen
                final_rhymes = []
                for candidate in chosen:
                    final_rhymes.append({"rhyme": candidate})
                random.shuffle(final_rhymes)
            # === ENDE NEU ===
            
            logger.info(f"--> Finale Ausgabe: {len(final_rhymes)} Reime werden an das Frontend gesendet.")
            logger.info("="*50 + "\n")
            return jsonify({"rhymes": final_rhymes}), 200

        except Exception as e:
            logger.error(f"Fehler bei kreativer Generierung: {e}")
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"Error in /api/rhymes: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-rhyme-line', methods=['POST'])
def generate_rhyme_line_endpoint():
    """
    Finaler, ganzheitlicher Endpunkt zur Generierung von Reimzeilen in höchster Qualität.
    Ersetzt alle vorherigen Versionen und Experimente.
    """
    try:
        data = request.get_json()
        input_line = data.get('input_line', '')
        knowledge_base = data.get('knowledge_base', '')
        num_lines = data.get('num_lines', 7)

        if not input_line:
            return jsonify({"error": "Keine Eingabezeile gefunden."}), 400

        logger.info(f"--- [FINAL V5 - GANZHEITLICH] Anfrage für: '{input_line}' ---")

        # Wir verwenden das Sprachverzeichnis direkt hier, falls es existiert.
        # (Annahme: SPRACHVERZEICHNIS_TEXT ist in main.py global verfügbar)
        try:
            language_directory = SPRACHVERZEICHNIS_TEXT
        except NameError:
            language_directory = "Kein Sprachverzeichnis gefunden."

        prompt = f"""
        **Systemanweisung: Du bist ein Meister-Lyriker und Phonetiker. Deine Aufgabe ist es, zu einer gegebenen Zeile mehrere, qualitativ herausragende Reimzeilen zu erschaffen.**

        **SPRACHVERZEICHNIS (Deine Wissensbasis für Definitionen):**
        ---
        {language_directory}
        ---

        **KÜNSTLER-DNA (Stil- und Inhaltsvorgaben):**
        ---
        {knowledge_base}
        ---

        **EINGABEZEILE (DEINE VORLAGE):**
        "{input_line}"

        **AUFGABE UND FINALE QUALITÄTS-HIERARCHIE:**
        Generiere {num_lines} neue, vollständige Zeilen. Jede einzelne Zeile muss die folgenden Regeln in absteigender Wichtigkeit befolgen:

        **PRIORITÄT 1 (Absolut Unverhandelbar - Das Fundament):**
        - **Sinnhaftigkeit & Logik:** Die generierte Zeile muss ein grammatikalisch korrekter, logisch nachvollziehbarer und in sich geschlossener Satz sein. Sie muss für sich allein stehend Sinn ergeben. Beziehe dich auf die Definition von "Sinnhaftigkeit & Kreativität" aus dem Sprachverzeichnis.

        **PRIORITÄT 2 (Sehr Wichtig - Der kreative Rahmen):**
        - **Thematischer Bezug:** Die Zeile muss thematisch und stimmungsvoll zur EINGABEZEILE passen.
        - **Prinzip der Reim-Varianz:** Das Endreimwort darf keine Wiederholung des Ziel-Reimwortes sein.

        **PRIORITÄT 3 (Wichtig, aber flexibel - Die technische Kunst):**
        - **Ganzzeilige Phonetische Resonanz:** Versuche, die Klangfarbe der gesamten Vorlage-Zeile so gut wie möglich zu treffen.
        - **Betonte Vokal-Harmonie:** Konzentriere dich darauf, die Vokale der HAUPTBETONTEN Silben klanglich anzugleichen. Ein kleiner Kompromiss in der Phonetik ist akzeptabel, wenn dadurch Priorität 1 und 2 perfekt erfüllt werden.

        **Strukturierte Ausgabe:** Gib deine Antwort als valides JSON zurück, das dem geforderten Schema exakt entspricht.
        """

        schema = {
            "type": "object",
            "properties": {
                "generated_lines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "line": {"type": "string", "description": "Die generierte Reimzeile."},
                            "analysis": {
                                "type": "object",
                                "properties": {
                                    "overall_similarity_score": {"type": "string", "description": "Bewerte die GANZZEILIGE phonetische Ähnlichkeit (1-10)."},
                                    "vowel_rhythm_match": {"type": "boolean", "description": "True, wenn der Vokal-Rhythmus stark gespiegelt wurde."}
                                }, "required": ["overall_similarity_score", "vowel_rhythm_match"]
                            }
                        }, "required": ["line", "analysis"]
                    }
                }
            }, "required": ["generated_lines"]
        }

        result = call_claude(prompt, schema=schema)
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in /api/generate-rhyme-line: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/process-vocal-emphasis', methods=['POST'])
def process_vocal_emphasis_endpoint():
    """
    Verarbeitet Audio-Feedback zur Betonung und generiert eine neue Analyse. [cite: 172]
    """
    try:
        data = request.get_json()
        audio_part = make_audio_part(data['base64Audio'], data['mimeType'])
        knowledge_base = data['knowledgeBase']
        item_content = data['itemContent']  # The lyrics

        prompt = f"""
        **System Instruction: You are a phonetic analyst. The user has provided an audio recording that represents the "absolute truth" for emphasis and rhythm.**

        **ARTIST'S DNA (for context):**
        ---
        {knowledge_base}
        ---

        **LYRICS:**
        ---
        {item_content}
        ---

        **AUDIO FOR ANALYSIS IS ATTACHED.**

        **YOUR TASKS (based ONLY on the audio):**
        1.  `learnedExplanation`: Formulate a single, general technical rule about the artist's emphasis style derived from this recording.
        2.  `newEmphasisPattern`: Create a new, complete phonetic rhyme analysis for the entire text, based on the emphasis in the audio.

        Return a single JSON object.
        """
        schema = {
            "type": "object",
            "properties": {
                "learnedExplanation": {"type": "string"},
                "newEmphasisPattern": {"type": "string"}
            },
            "required": ["learnedExplanation", "newEmphasisPattern"]
        }
        result = call_claude([prompt, audio_part], schema=schema)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in /api/process-vocal-emphasis: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/reanalyze-explanation', methods=['POST'])
def reanalyze_explanation_endpoint():
    """
    Korrigiert eine Analyse basierend auf textlichem Nutzerfeedback. [cite: 172]
    """
    try:
        data = request.get_json()
        # patternType: 'emphasis' | 'rhymeFlow'
        # userExplanation: "The words X and Y rhyme because..."
        # originalLyrics: The lyrics of the song
        # knowledge_base: The user's DNA

        prompt = f"""
        **System Instruction: You are a writing assistant. The user is correcting your previous analysis. Their explanation is the "absolute truth" and must be followed.**

        **ARTIST'S DNA (for context):**
        ---
        {data['knowledge_base']}
        ---

        **USER'S CORRECTION/EXPLANATION (Absolute Truth):**
        ---
        "{data['userExplanation']}"
        ---

        **ORIGINAL LYRICS:**
        ---
        {data['originalLyrics']}
        ---

        **TASK: Generate a new, corrected analysis pattern for `{data['patternType']}`.**
        Your new analysis must fully incorporate the user's correction for the entire text.

        Return a single JSON object with the key `correctedPattern` containing the new text.
        """
        schema = {
            "type": "object",
            "properties": {"correctedPattern": {"type": "string"}},
            "required": ["correctedPattern"]
        }
        result = call_claude(prompt, schema=schema)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in /api/reanalyze-explanation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/reanalyze-technique', methods=['POST'])
def reanalyze_technique_endpoint():
    """
    Generalisiert aus einer Nutzerkorrektur eine neue "technicalSkill". [cite: 181]
    """
    try:
        data = request.get_json()
        # learned_rule: "The artist often rhymes..."
        # original_lyrics: The song lyrics
        # original_skills: The array of old skills

        prompt = f"""
        **System Instruction: You are an analytical system that learns from user corrections.**

        **NEWLY LEARNED RULE (from user feedback):**
        ---
        "{data['learned_rule']}"
        ---

        **ORIGINAL LYRICS (for context):**
        ---
        {data['original_lyrics']}
        ---

        **PREVIOUSLY DERIVED TECHNICAL SKILLS:**
        ---
        {json.dumps(data['original_skills'], indent=2)}
        ---

        **TASK: Based on the 'newly learned rule', re-evaluate the 'previously derived technical skills'.**
        Return an array of strings representing the new, updated list of technical skills for this song. You may refine, remove, or replace the old skills.

        Return a single JSON object with the key `newTechnicalSkills`, which is an array of strings.
        """
        schema = {
            "type": "object",
            "properties": {
                "newTechnicalSkills": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["newTechnicalSkills"]
        }
        result = call_claude(prompt, schema=schema)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in /api/reanalyze-technique: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-lyrics', methods=['POST'])
def generate_lyrics_endpoint():
    """
    Generiert oder überarbeitet Songtexte basierend auf verschiedenen Modi (generation, revision, fusion).
    """
    try:
        data = request.get_json()
        mode = data.get('mode', 'generation')
        knowledge_base = data['knowledgeBase']

        if mode == 'generation':
            prompt = f"""
            **System Instruction: Du bist ein meisterhafter Songtexter und Reimkünstler. Deine Aufgabe ist es, einen Text zu verfassen, der den Anweisungen des Nutzers exakt folgt.

Du erhältst eine 'Künstler-DNA' mit Beispielen für Stil, Technik, Betonungsmuster und Reimfluss usw. Nutze diese DNA als deine primäre Inspirationsquelle. Analysiere die Reimschemata und die Komplexität der Reime in der DNA und emuliere diesen Stil.

Priorität 1: Hohe Reimqualität. Vermeide simple oder erzwungene Reime (Haus/Maus, Herz/Schmerz). Bevorzuge mehrsilbige, komplexe und unerwartete Reime, die thematisch sinnvoll sind. Nutze dazu die Informationen aus der DNA.

Priorität 2: Einhaltung der Struktur. Halte dich strikt an die vom Nutzer vorgegebene Struktur (z.B. '12-zeiliger Part').**

            **ARTIST'S DNA (Your Rule Set):**
            ---
            {knowledge_base}
            ---

            **USER'S PROMPT:**
            - Theme: {data.get('userPrompt', 'No theme provided.')}
            - Style: {data.get('additionalContext', {}).get('style', 'Not specified.')}
            - Technique: {data.get('additionalContext', {}).get('technique', 'Not specified.')}
            - Fusion: {data.get('additionalContext', {}).get('fusion', 'Not specified.')}
            - Beat Description: {data.get('additionalContext', {}).get('beatDescription', 'Not specified.')}
            - BPM: {data.get('additionalContext', {}).get('bpm', 'Not specified.')}
            - Key: {data.get('additionalContext', {}).get('key', 'Not specified.')}
            - Performance Style: {data.get('additionalContext', {}).get('performanceStyle', 'Not specified.')}

            **TASK: Write a complete song text (verses, chorus, etc.) that matches the prompt and perfectly embodies the Artist's DNA.**
            The song should reflect the specified style, technique, and musical context while maintaining the artist's unique voice.
            
            **CRITICAL REQUIREMENTS:**
            1. **Follow the Artist's Emphasis Pattern**: If the artist's DNA contains emphasis patterns, replicate the same rhythmic emphasis and stress patterns in your new lyrics.
            2. **Follow the Artist's Rhyme Flow Pattern**: If the artist's DNA contains rhyme flow patterns, use the same rhyme scheme structure, verse-chorus connections, and overall musical flow.
            3. **Maintain Consistency**: The new song should sound like it was written by the same artist, using their established rhythmic and rhyming techniques.
            
            **OUTPUT:** Write a complete song with proper structure tags ([Verse], [Chorus], etc.) that demonstrates the artist's unique emphasis and rhyme flow patterns.
            """
        elif mode == 'revision':  # [cite: 51]
            prompt = f"""
            **System Instruction: You are a lyric editor. Your task is to revise a specific part of a song while preserving the artist's core style.**

            **ARTIST'S DNA (Style Guide):**
            ---
            {knowledge_base}
            ---

            **ORIGINAL TEXT:**
            ---
            {data['originalText']}
            ---

            **REVISION REQUEST:**
            ---
            "{data['revisionRequest']}"
            ---

            **TASK: Execute the revision request precisely. Only change what is necessary and ensure the new text is coherent and still conforms to the Artist's DNA.**
            """
        elif mode == 'fusion':  # [cite: 56]
            source_songs_text = ""
            for i, song in enumerate(data.get('sourceSongs', [])):
                source_songs_text += f"\n--- SONG {i + 1} ---\n{song['content']}\n"

            prompt = f"""
            **System Instruction: You are a musical producer and songwriter, specializing in creating hybrid songs.**

            **ARTIST'S DNA (Overarching Framework):**
            ---
            {knowledge_base}
            ---

            **TASK: Create a new, song that believably merges the themes, imagery, and styles of the source songs.**
            The final product must be a single, coherent piece of art that still respects the global rules of the Artist's DNA. [cite: 65]
            """
        else:
            return jsonify({"error": "Invalid mode specified."}), 400

        # For generation, we expect a simple text response
        result_text = call_claude(prompt, is_json_output=False)
        return jsonify({"generatedText": result_text}), 200

    except Exception as e:
        logger.error(f"Error in /api/generate-lyrics: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/test-phonetics', methods=['POST'])
def test_phonetics_endpoint():
    """
    Test-Endpunkt zum Überprüfen der phonetischen Analyse ohne Claude API.
    """
    try:
        data = request.get_json()
        word = data.get('word', '')
        
        if not word:
            return jsonify({"error": "No word provided"}), 400
            
        analysis = get_phonetic_breakdown_py(word)
        
        # Teste auch einige bekannte Reime
        test_rhymes = []
        if word.lower() == 'geschichte':
            test_rhymes = ['errichtet', 'vernichten', 'errichtet', 'vernichten', 'gewissen']
        elif word.lower() == 'eigentlich':
            test_rhymes = ['reinen Tisch', 'bleiben nicht', 'Einzelkind', 'schreiben Pflicht']
        
        rhyme_analyses = []
        for rhyme in test_rhymes:
            rhyme_analyses.append({
                'rhyme': rhyme,
                'analysis': get_phonetic_breakdown_py(rhyme)
            })
        
        return jsonify({
            "input_word": word,
            "analysis": analysis,
            "test_rhymes": rhyme_analyses,
            "is_valid_rhyme": all(
                rhyme['analysis']['syllableCount'] == analysis['syllableCount'] and
                rhyme['analysis']['vowelSequence'] == analysis['vowelSequence']
                for rhyme in rhyme_analyses
            )
        }), 200
        
    except Exception as e:
        logger.error(f"Error in /api/test-phonetics: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# === API Endpunkte ===

@app.route('/api/synthesize-styles', methods=['POST'])
def synthesize_styles_endpoint():
    """
    Verbesserte Version: Analysiert Stil-Merkmale, führt semantisches Clustering durch
    UND weist jedem Cluster eine übergeordnete Kategorie zu.
    """
    try:
        data = request.get_json()
        style_items = data.get('style_items', [])

        if not style_items:
            return jsonify({"error": "Keine Stil-Elemente zur Analyse übergeben."}), 400

        items_text = "\n".join([f"- {item['content']}" for item in style_items])

        logger.info(f"--- [SYNTHESIZE V2] Starte Cluster-Analyse & Kategorisierung für {len(style_items)} Elemente ---")

        prompt = f"""
        **Systemanweisung: Du bist ein Datenanalyst und Bibliothekar, spezialisiert auf semantisches Clustering und Kategorisierung von literarischen Themen.**
        Deine Aufgabe ist es, die folgende ungeordnete Liste von Stil-Merkmalen zu analysieren, sie in Themen-Cluster zu gruppieren UND jeden Cluster einer passenden, übergeordneten Kategorie zuzuweisen.

        **VORGEGEBENE KATEGORIEN (Nur diese verwenden!):**
        - **Emotion:** Beschreibt Gefühle und Stimmungen.
        - **Thema/Motiv:** Beschreibt wiederkehrende inhaltliche Sujets.
        - **Stilmittel:** Beschreibt technische Aspekte der Sprache (z.B. Metaphern, Ironie).
        - **Erzählperspektive:** Beschreibt die Haltung oder Rolle des Erzählers (z.B. Selbstreflexion).
        - **Wortwahl:** Beschreibt die Art der verwendeten Sprache.

        **UNGEORDNETE LISTE VON STIL-MERKMALEN:**
        ---
        {items_text}
        ---

        **DEINE AUFGABE (in 4 Schritten):**
        1.  **Identifiziere Kernthemen:** Finde die zentralen, wiederkehrenden Hauptthemen in der Liste.
        2.  **Bilde Cluster:** Erstelle für jedes Kernthema einen Cluster mit einem prägnanten Titel.
        3.  **Ordne Facetten zu:** Ordne jeden ursprünglichen Eintrag dem passenden Cluster zu.
        4.  **Kategorisiere jeden Cluster:** Weise jedem Cluster eine der oben vordefinierten Kategorien zu.

        **Strukturierte Ausgabe:** Gib deine Antwort als valides JSON zurück, das dem geforderten Schema exakt entspricht.
        """

        schema = {
            "type": "object",
            "properties": {
                "style_clusters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cluster_title": {"type": "string", "description": "Der übergeordnete Name des Themas."},
                            "category": {
                                "type": "string",
                                "description": "Eine der vordefinierten Kategorien.",
                                "enum": ["Emotion", "Thema/Motiv", "Stilmittel", "Erzählperspektive", "Wortwahl"]
                            },
                            "facets": {"type": "array", "description": "Die ursprünglichen Stil-Einträge als Strings.", "items": {"type": "string"}}
                        },
                        "required": ["cluster_title", "category", "facets"]
                    }
                }
            },
            "required": ["style_clusters"]
        }

        # Aufruf über die neue zentrale Weiche
        result = call_ai_model("SYNTHESIZE_STYLES", prompt, schema=schema)
        
        # Mappe die originalen IDs zurück, um dem Frontend die Arbeit zu erleichtern
        original_items_by_content = {item['content']: item['id'] for item in style_items}
        
        for cluster in result.get('style_clusters', []):
            mapped_facets = []
            for content in cluster.get('facets', []):
                item_id = original_items_by_content.get(content)
                if item_id:
                    mapped_facets.append({'id': item_id, 'content': content})
            cluster['facets'] = mapped_facets
            cluster['cluster_id'] = f"cluster-{uuid.uuid4().hex}"

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in /api/synthesize-styles: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyze-beat', methods=['POST'])
def analyze_beat_endpoint():
    """
    Analysiert eine Audio-Datei mit Gemini 1.5 Pro, um BPM, Tonart und Beat-Beschreibung zu extrahieren.
    Verwendet die korrekte Multimodalität durch Datei-Upload auf Google-Server.
    """
    import tempfile
    import os
    import base64
    
    temp_audio_path = None  # Initialisieren für den finally-Block
    
    try:
        data = request.get_json()
        base64_audio = data.get('base64Audio')
        mime_type = data.get('mimeType')

        if not base64_audio or not mime_type:
            return jsonify({"error": "base64Audio und mimeType sind erforderlich."}), 400

        # 1. Base64 dekodieren und in eine temporäre Datei schreiben
        audio_bytes = base64.b64decode(base64_audio)
        
        # Erstellt eine temporäre Datei mit passender Endung
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            temp_audio_path = temp_audio_file.name

        # 2. Datei auf den Google-Server hochladen
        logger.info(f"Uploading audio file '{temp_audio_path}' to Google...")
        audio_file = genai.upload_file(path=temp_audio_path, mime_type=mime_type)
        logger.info("File uploaded successfully.")

        # 3. Prompt erstellen
        analysis_prompt = """
        Analysiere die angehängte Audiodatei. Gib deine Antwort als sauberes JSON-Objekt mit den folgenden drei Schlüsseln zurück: "bpm", "key", und "description".

        - "bpm": Ermittle das exakte Tempo in Beats pro Minute (nur die Zahl).
        - "key": Ermittle die Tonart (z.B. "C-Moll" oder "A-Dur").
        - "description": Beschreibe den Beat in 2-3 Sätzen. Gehe auf die Stimmung, die verwendeten Instrumente und den Rhythmus ein (z.B. "schleppend", "treibend", "entspannt").

        WICHTIG: Gib deine Antwort AUSSCHLIESSLICH als gültiges JSON aus, das diesem Schema entspricht. Keine zusätzlichen Texte, keine Erklärungen, nur das JSON:
        {
            "bpm": 120,
            "key": "C-Dur",
            "description": "Ein treibender Beat mit..."
        }
        """
        
        # 4. Gemini mit Text-Prompt UND Datei-Referenz aufrufen
        response = gemini_model.generate_content([analysis_prompt, audio_file])

        # 5. Antwort bereinigen und parsen
        parsed_result, parse_err = parse_json_safely(response.text)
        if parse_err:
            logger.error(f"/api/analyze-beat JSON-Fehler: {parse_err}\nRoh: {response.text[:4000]}")
            return jsonify({"error": "Beat-Analyse: JSON nicht lesbar.", "detail": parse_err}), 500

        # Schicke die extrahierten Daten an das Frontend zurück
        return jsonify({
            "bpm": parsed_result.get("bpm", ""),
            "key": parsed_result.get("key", ""),
            "description": parsed_result.get("description", "")
        }), 200

    except Exception as e:
        logger.error(f"Fehler bei der Gemini Beat-Analyse: {e}", exc_info=True)
        return jsonify({"error": "Die KI konnte den Beat nicht analysieren."}), 500
    
    finally:
        # 6. Temporäre Datei nach Gebrauch immer löschen
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            logger.info(f"Cleaned up temporary file: {temp_audio_path}")


@app.route('/api/structure-analysis', methods=['POST'])
def structure_analysis_endpoint():
    """
    Analysiert Text mit Gemini 1.5 Pro und wandelt ihn in strukturiertes JSON um.
    Unterstützt 'emphasis' und 'rhyme_flow' Analyse-Typen.
    """
    try:
        data = request.get_json()
        analysis_text = data.get('analysisText')
        analysis_type = data.get('analysisType')

        if not analysis_text or not analysis_type:
            return jsonify({"error": "analysisText and analysisType are required."}), 400

        # Erstelle den spezifischen Prompt für die KI
        prompt = f"""
Du bist ein Experte für musikalische Lyrik-Analyse. Deine Aufgabe ist es, den folgenden Analyse-Text in ein kompaktes, strukturiertes JSON-Objekt umzuwandeln. Extrahiere nur die wichtigsten, quantifizierbaren Merkmale.

**Analyse-Typ:**
"{analysis_type}"

**Analyse-Text:**
"{analysis_text}"

**Gewünschtes JSON-Format für "emphasis":**
{{
  "pattern_type": "<Hauptmuster, z.B. 'rhythmic_hip-hop'>",
  "main_focus": "<Hauptschwerpunkt, z.B. 'end_syllables'>",
  "key_features": ["<Merkmal 1 als Stichwort>", "<Merkmal 2 als Stichwort>"],
  "examples": ["<Beispiel 1>", "<Beispiel 2>"]
}}

**Gewünschtes JSON-Format für "rhyme_flow":**
{{
  "scheme": "<Reimschema, z.B. 'AABB'>",
  "flow_type": "<Flow-Art, z.B. 'continuous'>",
  "features": ["<Merkmal 1 als Stichwort>", "<Merkmal 2 als Stichwort>"],
  "rhyme_types": ["<Reim-Art 1, z.B. 'clean_rhyme'>", "<Reim-Art 2, z.B. 'impure_rhyme'>"]
}}

Gib NUR das fertige JSON-Objekt als Antwort zurück. Formatiere es ohne umschließende Markdown-Syntax.
"""

        # Rufe Gemini 1.5 Pro auf
        logger.info(f"Calling Gemini 1.5 Pro for {analysis_type} analysis...")
        response = gemini_model.generate_content(prompt)
        
        # Bereinige die Antwort und versuche sie zu parsen
        parsed_result, parse_err = parse_json_safely(response.text)
        if parse_err:
            logger.error(f"/api/structure-analysis JSON-Fehler: {parse_err}\nRoh-Antwort: {response.text[:4000]}")
            return jsonify({"error": "Analyse-Antwort konnte nicht als JSON gelesen werden.", "detail": parse_err}), 500

        
        # Entferne mögliche Markdown-Formatierung
        if cleaned_response.startswith("```") and cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[3:-3].strip()
        
        # Versuche, die Antwort der KI zu parsen
        try:
            structured_data = json.loads(cleaned_response)
        except json.JSONDecodeError as json_error:
            logger.error(f"JSON parsing failed: {json_error}")
            logger.error(f"Raw response: {response.text}")
            # Fallback: Erstelle ein einfaches strukturiertes Objekt
            if analysis_type == "emphasis":
                structured_data = {
                    "pattern_type": "unknown",
                    "main_focus": "unknown",
                    "key_features": ["analysis_failed"],
                    "examples": [analysis_text[:100] + "..." if len(analysis_text) > 100 else analysis_text]
                }
            else:  # rhyme_flow
                structured_data = {
                    "scheme": "unknown",
                    "flow_type": "unknown",
                    "features": ["analysis_failed"],
                    "rhyme_types": ["unknown"]
                }

        # Sende die strukturierten Daten zurück an das Frontend
        logger.info(f"Successfully structured {analysis_type} analysis")
        return jsonify({"structuredData": structured_data}), 200

    except Exception as e:
        logger.error(f"Error structuring analysis: {e}", exc_info=True)
        return jsonify({"error": "Failed to structure the analysis text."}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
