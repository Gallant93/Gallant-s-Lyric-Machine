import os
import json
import logging
import sys
import time
import random
import uuid
from typing import Union
import re

# --- Phonetik: Silbentrennung (deutsch) für Schwa/„er"-Normalisierung
try:
    import pyphen
    _DIC_DE = pyphen.Pyphen(lang="de_DE")
except Exception:
    _DIC_DE = None

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



def _log_module_paths_once():
    try:
        m_main = sys.modules.get("backend.main")
        m_idx  = sys.modules.get("backend.dna_indexer")
        print(f"[BOOT] main.py from: {getattr(m_main, '__file__', '<?>')}")
        print(f"[BOOT] dna_indexer.py from: {getattr(m_idx, '__file__', '<?>')}")
    except Exception as e:
        print(f"[BOOT] module path log failed: {e!r}")

_log_module_paths_once()

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

        # Nur das „echte" JSON heraus schneiden
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

# Nähe der anderen Konstanten (z.B. nach OPENAI_RHYMES_MODEL)
PREFIX_FAMILY_CAP = 2

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

# EINFÜGEN - Helper binden für DNA-Indexer
def _dna_stub(name):
    """Stub für nicht vorhandene Funktionen"""
    def stub(*args, **kwargs):
        logger.warning(f"DNA-Indexer: Function '{name}' not implemented, returning default")
        if name == "extract_vowel_sequence":
            return ["a", "e", "i"]  # Dummy-Sequenz
        elif name == "first_vowel_and_length":
            return ("a", "short")  # Dummy-Familie und Länge
        elif name == "inner_stressed_core":
            return None  # Kein innerer Kern
        elif name == "last_vowel_family":
            return "e"  # Dummy-Familie
        elif name.startswith("passes_"):
            return True  # Optimistisch durchlassen
        elif name == "prefix_family_key":
            return args[0][:4] if args else "default"  # Ersatz-Schlüssel
        return None
    return stub




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

_DEF_CONS = set("bcdfghjklmnpqrstvwxyzß")

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

# --- Vokal-Familien-Nachbarschaften: nur für den ERSTEN Vokal erlaubt
_NEAR_FAMILY = {
    "o": {"eu", "au"},
    "eu": {"o"},
    "au": {"o"},
    "e": {"ä"},
    "ä": {"e"},
    "i": {"ie"},
    "ie": {"i"},
}

def same_or_neighbor_family(f1: str, f2: str, position: str) -> bool:
    if f1 == f2:
        return True
    if position == "first":
        return (f2 in _NEAR_FAMILY.get(f1, set())) or (f1 in _NEAR_FAMILY.get(f2, set()))
    return False

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








# --- Silbifizierung (deutsch): benutzt pyphen, fallback auf simple Heuristik
def _syllabify_de(word: str) -> list[str]:
    w = word.lower()
    if _DIC_DE:
        return _DIC_DE.inserted(w).split("-")
    # Fallback: grobe Heuristik (nur als Notnagel)
    # trennt nach Vokalgruppen. Für Schwa-Erkennung reicht's als Plan B.
    parts = re.findall(r"[bcdfghjklmnpqrstvwxyz]*[aeiouyäöü]+(?:h)?[bcdfghjklmnpqrstvwxyz]*", w)
    return parts if parts else [w]

# --- Schwa-/„er"-Normalisierung NUR für phonetische Analyse:
#     - 'er' am Wortende → 'a'
#     - 'er' am Silbenende (nicht erste Silbe) → 'a'  (z.B. HammER-werfER → Hamma-werfA)
def normalize_er_schwa(word: str) -> str:
    w = word.lower()

    # 1) Wortfinales '-er' → 'a'
    w = re.sub(r"er\b", "a", w)

    # 2) Silben-finales '-er' innerhalb von Wörtern (nicht in der ersten Silbe)
    syls = _syllabify_de(w)
    if len(syls) > 1:
        new_syls = []
        for i, syl in enumerate(syls):
            s = syl
            # „-er" am Ende der Silbe (sehr häufig Schwa) → 'a', aber erste Silbe ausklammern
            if i > 0 and s.endswith("er"):
                s = s[:-2] + "a"
            new_syls.append(s)
        w = "".join(new_syls)

    # Sicherung: falls doch noch Rest 'er' + Konsonantgrenze, entschärfen
    w = re.sub(r"er(?=[bcdfghjklmnpqrstvwxyz]{1,2}\b)", "a", w)
    return w


def count_syllables(word: str) -> int:
    """Öffentlicher Silbenzähler – nutzt unsere neue Silbifizierung.
    Hält die alte API am Leben und vermeidet NameErrors."""
    try:
        return max(1, len(_syllabify_de(word)))
    except Exception:
        return 1


# --- Präfix-Blocker: verhindere Serien wie "Polter..." bei "Polterabend" ---
_VOWELS_DE = "aeiouäöüyAEIOUÄÖÜY"

def get_forbidden_prefix(word: str) -> list[str]:
    """
    Liefert eine kleine Menge an Startfolgen, mit denen KEIN Kandidat beginnen darf.
    Heuristik:
      - nimm das erste Token (vor Leer- oder Bindestrich)
      - reduziere auf Buchstaben
      - schneide vor dem Beginn der 2. Vokalkern-Gruppe ab (z.B. 'polter' | 'abend')
      - generiere ein paar robuste Varianten (mit/ohne Bindestrich, ohne End-e)
    """
    import re

    token = re.split(r"[\s\-]", word.strip(), maxsplit=1)[0].lower()
    token = re.sub(r"[^a-zäöüß]", "", token)
    if not token:
        return []

    # Vokalkerne finden
    nuclei = [m.start() for m in re.finditer(rf"[{_VOWELS_DE}]+", token)]
    if len(nuclei) >= 2:
        prefix = token[:nuclei[1]]  # bis vor Beginn des 2. Kerns (z.B. 'polter')
    else:
        prefix = token

    if len(prefix) < 4:  # zu kurz? bringt nichts zu blocken
        return []

    variants = {prefix, prefix + "-", prefix.rstrip("e"), prefix.rstrip("e") + "-"}
    # Doppel-s/Deklination etc. sind hier egal – wir blocken nur Starts
    return sorted(variants)


def starts_with_forbidden(name: str, forbidden: list[str]) -> bool:
    """
    True, wenn 'name' mit einem der verbotenen Präfixe beginnt.
    Bindestriche/Leerzeichen werden ignoriert (robust gegen Varianten).
    """
    import re
    base = re.sub(r"[\s\-]+", "", name.lower())
    for f in forbidden:
        f_clean = re.sub(r"[\s\-]+", "", f.lower())
        if f_clean and base.startswith(f_clean):
            return True
    return False


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

# --- Einheitliche Voranalyse (liefert Sequenz, first/last, Länge, Silben) ---
def compute_voranalyse(word: str):
    # Silbenzählung mit Original-Wort (unveränderlich)
    sylls = count_syllables(word)

    # Vokalsequenz mit normalisiertem Wort (für korrekte Phonetik)
    normalized_word = normalize_er_schwa(word) if 'normalize_er_schwa' in globals() else word

    if '_vowel_seq_for_compare' in globals() and callable(_vowel_seq_for_compare):
        seq = _vowel_seq_for_compare(normalized_word)
    elif 'vowel_seq_for_compare' in globals() and callable(vowel_seq_for_compare):
        seq = vowel_seq_for_compare(normalized_word)
    elif 'simple_vowel_scan_de' in globals() and callable(simple_vowel_scan_de):
        seq = simple_vowel_scan_de(normalized_word)
    else:
        # Fallback mit normalisiertem Wort (ultra-einfacher Fallback)
        w = (normalized_word or "").lower()
        out, i = [], 0
        V = "aeiouyäöü"
        while i < len(w):
            ch2 = w[i:i+2]
            if ch2 == "er" and (i+2 == len(w) or all(c not in V for c in w[i+2:])):
                out.append("a"); i += 2; continue
            if ch2 in ("ei","ie","ai","au","eu","äu","oi"):
                out.append(ch2[0]); i += 2; continue
            if w[i] in V: out.append(w[i])
            i += 1
        seq = out

    first_family = seq[0] if seq else None
    last_family  = seq[-1] if seq else None
    first_len = get_first_length_class(normalized_word) if 'get_first_length_class' in globals() else "short"

    return sylls, seq, first_family, first_len, last_family






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


# --- PHRASE-CORE: letztes Inhaltswort extrahieren ---
def _last_content_token(text: str) -> str:
    import re
    # nur Buchstaben als Token (inkl. Umlaute ß)
    toks = re.findall(r"[A-Za-zÄÖÜäöüß]+", text or "")
    if not toks:
        return (text or "").strip()
    # End-Stoppwörter überspringen (nicht verbieten – nur für die Reimprüfung)
    stop = {"und", "den", "der", "die", "das", "am", "an", "im", "vom", "zum"}
    i = len(toks) - 1
    while i >= 0 and toks[i].lower() in stop:
        i -= 1
    return toks[i] if i >= 0 else toks[-1]

# --- „er" am Ende klingt wie „a" (nur für den letzten Vokal relevant) ---
def _normalize_final_e_schwa_for_last(word: str) -> str:
    import re
    # nur am Wortende ersetzen
    return re.sub(r"er\b", "a", word or "", flags=re.IGNORECASE)

# --- Vokal-Familien-Helper (Wrapper für bestehende Logik) ---
def get_first_vowel_family(word: str) -> str:
    """Erste Vokalfamilie aus der Sequenz extrahieren."""
    # Einfache Vokal-Extraktion als Fallback
    w = word.lower()
    V = "aeiouyäöü"
    for i, ch in enumerate(w):
        if ch in V:
            return ch
    return ""

def get_last_vowel_family(word: str) -> str:
    """Letzte Vokalfamilie aus der Sequenz extrahieren."""
    # Einfache Vokal-Extraktion als Fallback
    w = word.lower()
    V = "aeiouyäöü"
    for i in range(len(w) - 1, -1, -1):
        if w[i] in V:
            return w[i]
    return ""

def get_first_length_class(word: str) -> str:
    """Erste Vokallängenklasse bestimmen."""
    # Vereinfachte Implementierung
    w = word.lower()
    V = "aeiouyäöü"
    i = next((k for k, ch in enumerate(w) if ch in V), -1)
    if i == -1:
        return "unknown"
    
    # Einfache Heuristik
    if i + 1 < len(w) and w[i+1] in V:
        return "long"
    if i + 1 < len(w) and w[i+1] == "h":
        return "long"
    
    # Konsonanten bis zum nächsten Vokal zählen
    j = i + 1
    cons = 0
    while j < len(w) and w[j] not in V:
        cons += 1
        j += 1
    
    return "short" if cons >= 2 else "long"

def get_last_length_class(word: str) -> str:
    """Letzte Vokallängenklasse bestimmen (vereinfacht)."""
    # Für den letzten Vokal verwenden wir die gleiche Logik wie für den ersten
    return get_first_length_class(word)

@app.route('/api/rhymes', methods=['POST'])
def find_rhymes_endpoint():
    """
    Korrigierte Version: Verwendet den exakten zweistufigen Prozess aus test_models.py
    MIT den neuen sprachlichen Verbesserungen im Prompt
    """
    # Mini-Fallback, falls keine Vokalsequenz aus der Analyse kommt
    def _simple_vowel_scan_de(word: str):
        w = word.lower()
        out = []
        i = 0
        while i < len(w):
            ch2 = w[i:i+2]
            # sehr grob – reicht als Fallback; deine Analyse ist primär
            if ch2 in ("ei", "ai", "au", "eu", "äu", "oi", "ie"):
                out.append(ch2[0])      # Vokalfamilie ~ erster Buchstabe
                i += 2
                continue
            if w[i] in "aeiouyäöü":
                out.append(w[i])
            i += 1
        return out

    # --- Sequence-Gate Helpers (deutsche Vokalspur; inkl. "er" → 'a' Mapping) ---
    def _vowel_seq_for_compare(word: str):
        w = word.lower()
        V = "aeiouyäöü"
        out = []
        i = 0
        while i < len(w):
            # "er" → 'a', wenn 'er' in Koda (nach 'r' kein Vokal oder Wortende)
            if w[i] == "e" and i + 1 < len(w) and w[i + 1] == "r":
                nxt = w[i + 2] if i + 2 < len(w) else ""
                if not nxt or nxt not in V:
                    out.append("a")
                    i += 2
                    continue
            # einfache Diphthong-Heuristik: erste Komponente als Familie
            dig = w[i:i+2]
            if dig in ("ei", "ie", "ai", "au", "eu", "äu", "oi"):
                out.append(dig[0])
                i += 2
                continue
            if w[i] in V:
                out.append(w[i])
            i += 1
        return out

    def _hamming(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    def _seq_ok(base_seq, cand_seq, var_limit: int):
        return (
            len(cand_seq) == len(base_seq)
            and base_seq and cand_seq
            and cand_seq[0] == base_seq[0]
            and cand_seq[-1] == base_seq[-1]
            and _hamming(base_seq, cand_seq) <= var_limit
        )

    # --- "er" → "a" Normalisierung für Endsilbe ---
    def _normalize_final_e_schwa(word: str) -> str:
        # "er" am Wortende klingt i.d.R. wie "a"
        return re.sub(r"er\b", "a", word, flags=re.IGNORECASE)

    # --- Erste-Vokal-Länge (kurz/lang), einfache deutsche Heuristik ---
    def _first_vowel_length(word: str) -> str:
        """
        'long' wenn: Doppelvokal (aa/ee/oo/..), Dehnungs-h nach dem Vokal,
        oder die erste Silbe ist 'offen' (genau 1 Konsonant zwischen erstem und nächstem Vokal).
        'short' wenn: Konsonantencluster >= 2 nach dem Vokal => geschlossene Silbe.
        Fallback: 'unknown' (nicht prüfen).
        """
        w = word.lower()
        V = "aeiouyäöü"
        # Position des ersten Vokals
        i = next((k for k,ch in enumerate(w) if ch in V), -1)
        if i == -1:
            return "unknown"

        # Doppelvokal (aa/ee/oo/…)
        if i + 1 < len(w) and w[i+1] in V:
            return "long"
        # Dehnungs-h (z. B. "oh", "ah")
        if i + 1 < len(w) and w[i+1] == "h":
            return "long"

        # Konsonanten bis zum nächsten Vokal zählen
        j = i + 1
        cons = 0
        while j < len(w) and w[j] not in V:
            cons += 1
            j += 1

        if cons >= 2:
            return "short"   # geschlossen -> kurz
        # cons == 0 (direkt Vokal) oder cons == 1 (typisch offene Silbe) -> tendenziell lang
        return "long"

    logger.info(">>> /api/rhymes LOADED (patch v1)")
    # === Request-Parsing & Defaults (robust) ===
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = {}

    input_word: str = (data.get("input") or data.get("word") or "").strip()
    if not input_word:
        return jsonify({"error": "Missing 'input'"}), 400

    knowledge_base = data.get('knowledge_base', 'Keine DNA vorhanden.')
    # Kontrakt: UPPER-Namen werden in der Funktion weiterverwendet
    MAX_RESULTS: int = int(data.get("max_results", 20))
    TARGET_PHRASE_RATIO: float = float(data.get("target_phrase_ratio", 0.35))
    max_words: int = int(data.get("max_words", 8))
    DEBUG_VALIDATION: bool = bool(data.get("debug_validation", False))

    # Nutzer-Parameter / weitere Defaults
    MAX_PER_FAMILY     = int(data.get('max_per_family', 2))

    # DNA-System entfernt: kein Binding mehr notwendig

    # Präfix-Verbotsregel vorab berechnen (kurz & robust)
    ban_prefix_rule: str = ""
    forbidden_prefix = get_forbidden_prefix(input_word) or []
    if forbidden_prefix:
        quoted = ", ".join([f'"{p}"' for p in forbidden_prefix])
        ban_prefix_rule = (
            f"10) Verbotene Präfixe: {quoted}. Keine Kandidaten dürfen damit beginnen."
        )

    # --- Literal-Input-Stamm (zusätzlicher Hartfilter gegen Near-Duplicates) ---
    try:
        forbidden_literal = (
            normalize_letters(input_word) if 'normalize_letters' in globals() else input_word
        ).lower()
    except Exception:
        forbidden_literal = (input_word or "").lower()

    # Nachfolgender Block war zuvor eingerückt; kapseln, damit Struktur konsistent bleibt
    if True:
        # Strenge Toleranz NUR für den Validator (Hamming-Distanz)
        VAR_LIMIT = int(data.get("var_limit", 1))  # 1 bleibt hart

        # Etwas lockerer NUR in der Generierung (Prompt), kostet fast nichts
        GEN_VAR_LIMIT = int(data.get("gen_var_limit", 2))  # 2 = innere Abweichungen bis 2 im Text

        # Basissequenz- und Erstlänge-Logging entfernt (doppelte Analyse)

        # Basis-Schwa & Silben
        base_syllables = count_syllables(input_word)
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
        # core_word = input_word (das zu analysierende Wort / Phrase-Core)
        norm_core = normalize_er_schwa(input_word)

        # Zähle & analysiere ab jetzt IMMER auf der Normalform
        target_analysis = get_phonetic_breakdown_py(norm_core)
        target_syllables = target_analysis.get("syllableCount")
        target_vowels = target_analysis.get("vowelSequence") or ""
        first_family = target_analysis.get("firstVowelFamily")
        last_family = target_analysis.get("lastVowelFamily")
        first_length = target_analysis.get("firstLengthClass")
        last_length = target_analysis.get("lastLengthClass")

        # Einheitliche Voranalyse
        base_syllables, base_seq, first_family, base_len_class, last_family = compute_voranalyse(input_word)
        norm_input = normalize_er_schwa(input_word) if 'normalize_er_schwa' in globals() else input_word
        app.logger.info(f"--> Vor-Analyse für '{input_word}': Silben={base_syllables}, Vokale='{ '-'.join(base_seq) if base_seq else '' }', first=({first_family}/{base_len_class}), last={last_family}, norm='{norm_input}'")

        # === DNA-FIRST LOGIK ===
        mode = (data.get("mode") or "").lower().strip()  # "", "dna_first", "dna_only", "llm_only"

        llm_allowed = (mode != "dna_only")

        accepted = []



        # Präfix-Blocker direkt bestimmen und loggen
        forbidden_prefix = get_forbidden_prefix(input_word)
        logger.info(f"Prefix-Blocker: {forbidden_prefix}")
        
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

        # --- Vokal-Sequenz aus der Analyse / Fallback ---
        vseq_str = (
            target_analysis.get("vowelSequence")
            or target_analysis.get("vowel_sequence")
            or ""
        )
        vowels = [v for v in vseq_str.split("-") if v] if vseq_str else []
        if not vowels:
            vowels = _simple_vowel_scan_de(norm_for_last)

        # --- "er" → "a" Normalisierung für Endsilbe (Aussprache) ---
        norm_for_last = _normalize_final_e_schwa(input_word)
        # ab hier 'norm_for_last' statt 'input_word' für den **letzten** Vokal-Familien-Check benutzen

        # --- Schwa-Kandidat am Wortende erkennen (einfach, einmalig) ---
        base_word_lc = input_word.lower()
        schwa_suffixes = ("e", "en", "er", "el", "em", "es")
        base_core = base_word_lc
        for _suf in schwa_suffixes:
            if base_word_lc.endswith(_suf):
                base_core = base_word_lc[: -len(_suf)]
                break

        # --- first/last Vokalfamilie/Laenge robust füllen ---
        if not first_vowel_family and vowels:
            first_vowel_family = vowels[0]
        if not first_vowel_length:
            first_vowel_length = len_class  # "short"/"long" aus deiner Analyse

        # letzter relevanter Vokal (Schwa ignorieren, falls erkannt)
        if vowels:
            last_vowel_family = vowels[-1]
            if base_core != base_word_lc and len(vowels) >= 2:
                last_vowel_family = vowels[-2]
        if not last_vowel_length:
            # Heuristik: wenn wir Schwa abgeschnitten haben, letzten Vokal eher "long"
            last_vowel_length = "long" if base_core != base_word_lc else len_class

        # --- Kernvokal-Heuristik, falls keine Betonung erkannt (und >=3 Vokale) ---
        if not core_vowel_family and len(vowels) >= 3:
            # meist sitzt der hörbare Kern vor der letzten Silbe -> zweitletzter
            core_vowel_family = vowels[-2]
            if not core_vowel_length:
                core_vowel_length = last_vowel_length  # sanfte Annahme

        logger.info(
            f"-- Anker: vowels={vowels}, first={first_vowel_family}/{first_vowel_length}, "
            f"core={core_vowel_family or '-'} , last={last_vowel_family}/{last_vowel_length}"
        )

        if core_vowel_family:
            core_rule       = f"Zwischen erstem und letztem Vokal muss **genau einmal** ein betonter Vokal der Familie {core_vowel_family} (Länge={core_vowel_length}) auftreten."
            core_rule_hint  = "Dieser Kern liegt syllabisch vor der letzten Silbe."
            core_selfcheck  = f"den inneren betonten {core_vowel_family}-Kern nicht enthält"
        else:
            core_rule       = "Kein zusätzlicher innerer Kern erforderlich."
            core_rule_hint  = ""
            core_selfcheck  = "False"   # Platzhalter -> fällt im Prompt als Bedingung faktisch weg

        # --- Prefix-Diversity (weiche Kappung) vorbereiten ---
        PREFIX_FAMILY_CAP: int = int(data.get("prefix_family_cap", 2))
        prefix_rule: str = (
            f"11) Präfix-Diversität: Pro Präfix-Familie max. {PREFIX_FAMILY_CAP} Kandidaten "
            f"(z. B. 'ab-', 'ver-', 'sonnen-')."
        )

        # Neuer Single-Pass Prompt
        creative_prompt = f"""
Finde {MAX_RESULTS} deutsche Reime für "{input_word}".

REGELN:
1) Exakt {base_syllables} Silben
2) Vokalfolge: {'-'.join(base_seq) if base_seq else '???'} (identisch)
3) Deutsche Wörter oder sinnvolle Komposita/Phrasen
4) {int(TARGET_PHRASE_RATIO * 100)}% Phrasen (2-4 Wörter), Rest Einzelwörter

Gib nur JSON zurück: {{"candidates":["Wort1","Phrase mit zwei Wörtern","Kompositum","..."]}}
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

            # --- JSON-SAFE EXTRACT + LINIEN-FALLBACK ---
            obj, err = parse_json_safely(creative_output)  # nutzt deine parse_json_safely oben
            if obj and isinstance(obj, dict) and "candidates" in obj and isinstance(obj["candidates"], list):
                candidates = [c.strip() for c in obj["candidates"] if isinstance(c, str) and c.strip()]
            else:
                # Fallback: plain lines zu candidates machen
                lines = [ln.strip("-*• 0123456789.").strip() for ln in creative_output.splitlines() if ln.strip()]
                # Alles was wie JSON aussieht (Klammern) vorher raus
                lines = [ln for ln in lines if not (ln.startswith("{") or ln.endswith("}"))]
                candidates = lines[:MAX_RESULTS]

            logger.info(f"--> Extrahierte Kandidaten: {candidates!r}")

            # --- Sofort-Filter: verbotene Präfixe direkt rauswerfen ---
            if forbidden_prefix:
                clean_candidates = []
                for cand in candidates:
                    if starts_with_forbidden(cand, forbidden_prefix):
                        logger.debug(f"verworfen (Prefix): {cand}")
                        continue
                    clean_candidates.append(cand)
                candidates = clean_candidates

            # Post-Processing direkt hier durchführen
            valid_candidates = []
            prefix_counts = {}
            # NEU: Robuste Schleife, die "Junk"-Wörter UND KI-Kommentare ignoriert
            
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

            for line in candidates:
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

                # --- Hardban: literaler Stamm + Präfixliste ---
                cand = cleaned_line.strip()
                cand_norm = (normalize_letters(cand) if 'normalize_letters' in globals() else cand).lower()
                # 2a) literal gleicher Anfang wie das Ausgangswort
                if cand_norm.startswith(forbidden_literal):
                    if DEBUG_VALIDATION:
                        logger.info(f"    REJECT forbidden_literal: {cand}")
                    continue
                # 2b) konfigurierter Präfix-Ban (z. B. 'polter-')
                if starts_with_forbidden(cand, forbidden_prefix):
                    if DEBUG_VALIDATION:
                        logger.info(f"    REJECT ban_prefix: cand='{cand}' prefixes={forbidden_prefix}")
                    continue

                # 1) Phrase-Core bestimmen
                core = _last_content_token(cleaned_line)

                # 2) Für den letzten-Vokal-Check die „er→a"-Normalisierung nur am Ende anwenden
                core_for_last = _normalize_final_e_schwa_for_last(core)

                # (optional, aber hilfreich) – kurzer Log
                logger.debug(f"Phrase-Core: '{cleaned_line}' -> '{core}', last-norm='{core_for_last}'")

                # Phonetik
                if get_schwa_suffix(cleaned_line) != base_schwa:
                    continue

                # --- Normalisierung für Kandidaten (wie beim Zielwort) ---
                cand_norm = normalize_er_schwa(core)
                cand_norm_for_last = normalize_er_schwa(core_for_last)

                # --- Erster Vokal (Familie/Längenklasse) – mit Nachbar-Toleranz ---
                cand_first_family = get_first_vowel_family(cand_norm)
                if not same_or_neighbor_family(cand_first_family, _vowel_seq_for_compare(input_word)[0], "first"):
                    continue

                cand_first_len = get_first_length_class(cand_norm)
                if cand_first_len and base_len_class and cand_first_len != base_len_class:
                    continue

                # --- Letzter Vokal (Familie/Längenklasse) – streng identisch ---
                cand_last_family = get_last_vowel_family(cand_norm_for_last)
                if cand_last_family != _vowel_seq_for_compare(input_word)[-1]:
                    continue

                # Letzte Vokallänge-Check entfernt (vereinfachter Ansatz)

                if not vowel_core_compatible(input_word, cleaned_line):
                    continue
                if count_syllables(cand_norm) != base_syllables:
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

                # Diversity-Cap je Präfix-Familie
                key = prefix_family_key(cleaned_line) if 'prefix_family_key' in globals() else normalize_letters(cleaned_line)[:4]
                if prefix_counts.get(key, 0) >= PREFIX_FAMILY_CAP:
                    if DEBUG_VALIDATION:
                        logger.info(f"    REJECT prefix_diversity[{key}]: {cleaned_line}")
                    continue
                prefix_counts[key] = prefix_counts.get(key, 0) + 1
                valid_candidates.append(cleaned_line)

            valid_candidates = list(dict.fromkeys(valid_candidates))[:20]

            logger.info(f"--> Schritt 2: Post-Processing")
            if DEBUG_VALIDATION:
                try:
                    app.logger.info(f"prefix_counts={prefix_counts}")
                except Exception:
                    pass
            logger.info(f"    - {len(valid_candidates)} valide Kandidaten aus der Antwort extrahiert.")
            logger.info(f"    - Extrahierte Kandidaten: {valid_candidates}")

            # --- NEU: strenges Silben-Gate für Run 1 ---
            raw_count = len(valid_candidates)
            syllable_ok = []
            if DEBUG_VALIDATION:
                rejects = []

            for c in valid_candidates:
                syl = count_syllables(c)
                if syl == base_syllables:
                    # --- Sequence-Gate: Vokalspur muss passen ---
                    # Phrase-Core für Vokalprüfung extrahieren
                    core = _last_content_token(c)
                    core_for_last = _normalize_final_e_schwa_for_last(core)
                    
                    cand_seq_cmp = _vowel_seq_for_compare(cand_norm_strict)
                    if not _seq_ok(_vowel_seq_for_compare(input_word), cand_seq_cmp, VAR_LIMIT):
                        if DEBUG_VALIDATION:
                            logger.info(f"    REJECT seq: base={base_seq_str} cand={'-'.join(cand_seq_cmp)}")
                        continue

                    # --- Normalisierung für Kandidaten in strenger Phase ---
                    cand_norm_strict = normalize_er_schwa(core)
                    
                    # --- Erste-Vokal-Länge muss mit der des Basisworts übereinstimmen (falls bekannt) ---
                    if base_len_class in ("short", "long"):
                        cand_first_len = _first_vowel_length(cand_norm_strict)
                        if cand_first_len != base_len_class:
                            if DEBUG_VALIDATION:
                                logger.info(f"    REJECT len: base={base_len_class} cand={cand_first_len} word={c}")
                            continue

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

            # --- Optional: Sortierung nach Hamming-Distanz (beste zuerst) ---
            def _score_by_seq(cand: str):
                core = _last_content_token(cand)
                cseq = _vowel_seq_for_compare(core)
                return _hamming(_vowel_seq_for_compare(input_word), cseq)  # 0 (perfekt) vor 1 vor 2 ...

            valid_candidates.sort(key=_score_by_seq)

            # ---- Diversity-Cap: max. 2 pro End-Familie (letzte 4 Buchstaben)
            def _family_key(w: str) -> str:
                s = re.sub(r"[^a-zäöüß]", "", w.lower())
                return s[-4:] if len(s) >= 4 else s

            MAX_PER_FAMILY = 2
            diverse = []
            seen = {}
            for c in valid_candidates:
                k = _family_key(c)
                if seen.get(k, 0) >= MAX_PER_FAMILY:
                    if DEBUG_VALIDATION:
                        logger.info(f"    DROP by diversity cap: {c} (family {k})")
                    continue
                diverse.append(c)
                seen[k] = seen.get(k, 0) + 1

            valid_candidates = diverse

            # --- Optional: Sortierung nach Hamming-Distanz (beste zuerst) ---
            def _score_by_seq(cand: str):
                core = _last_content_token(cand)
                return _hamming(_vowel_seq_for_compare(input_word), _vowel_seq_for_compare(core))

            valid_candidates.sort(key=_score_by_seq)  # 0-Fehler vor 1-Fehler vor 2-Fehler

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
            
            # === PASS 2: Quality-aware Reject-Sampling ===
            PHRASE_TOL = float(data.get("phrase_tolerance", 0.10))         # ±10% Toleranz vs. Ziel
            MIN_PREFIX_FAMILIES = int(data.get("min_prefix_families", 4))  # Diversitätsuntergrenze
            N_MIN = int(data.get("n_min", 12))

            total_now = len(valid_candidates)
            phrases_now = sum(1 for x in valid_candidates if isinstance(x, str) and " " in x)
            desired_total   = min(MAX_RESULTS, max(N_MIN, 1))
            desired_phrases = int(round(TARGET_PHRASE_RATIO * desired_total))

            phrase_ratio_now = (phrases_now / max(1, total_now))
            phrase_ratio_low  = phrase_ratio_now < (TARGET_PHRASE_RATIO - PHRASE_TOL)
            need_total        = max(0, desired_total - total_now)
            families_now      = len(seen_signatures.keys())
            need_diversity    = families_now < MIN_PREFIX_FAMILIES

            # Single-Pass System - vereinfachtes Post-Processing
            final_candidates = []
            for line in candidates:
                cleaned = line.strip()
                if not cleaned: continue
                
                # Nur harte Checks
                if count_syllables(cleaned) == base_syllables:
                    if is_valid_phrase(cleaned) if ' ' in cleaned else is_german_word_or_compound(cleaned):
                        final_candidates.append(cleaned)
                
                if len(final_candidates) >= MAX_RESULTS:
                    break

            # Finale Antwort erstellen
            final_rhymes = []
            for candidate in final_candidates[:MAX_RESULTS]:
                final_rhymes.append({"rhyme": candidate})
            random.shuffle(final_rhymes)

            logger.info(f"--> Finale Ausgabe: {len(final_rhymes)} Reime werden an das Frontend gesendet.")
            logger.info("="*50 + "\n")
            return jsonify({"rhymes": final_rhymes}), 200
