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
from google import genai

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
GENAI = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash-exp")

def gemini_generate(contents):
    return GENAI.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=contents
    )

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
                        response = gemini_generate([text_prompt, audio_part])
                    else:
                        text_prompt = " ".join([str(p) for p in prompt])
                        response = gemini_generate(text_prompt)
                else:
                    response = gemini_generate(prompt)
                
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
        audio_file = GENAI.files.upload(path=temp_audio_path, mime_type=mime_type)
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
        response = GENAI.models.generate_content(
            model=os.getenv("GEMINI_AUDIO_MODEL", GEMINI_TEXT_MODEL),
            contents=[analysis_prompt, audio_file]
        )

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


def _two_syllable_tail_key(text: str) -> str:
    """Schlüssel über die letzten ZWEI Silben für exakte Deduplizierung von Endsilben-Familien."""
    word = _last_content_token(text).lower()
    syls = _syllabify_de(word)
    if len(syls) >= 2:
        return "".join(syls[-2:])  # z.B. 'la'+'ge' -> 'lage' (deckt '-anlage'-Familie ab)
    return syls[-1] if syls else word


def generate_single_batch(creative_prompt, input_word, base_syllables, MAX_RESULTS, forbidden_prefix=None, var_limit=0):
    """Generiert EINEN Batch von validierten Reim-Kandidaten"""
    try:
        # Schema für die KI-Abfrage
        creative_schema = {
            "type": "object",
            "properties": {"candidates": {"type": "array", "items": {"type": "string"}, "minItems": 10, "maxItems": 30}},
            "required": ["candidates"]
        }

        # KI-Abfrage durchführen
        if RHYME_PROVIDER == 'gpt' and openai_client:
            obj = call_gpt_candidates(creative_prompt, schema=creative_schema)
            candidates = [c for c in obj.get("candidates", []) if isinstance(c, str)]
        else:
            response = anthropic_client.messages.create(
                    model="claude-opus-4-1-20250805",
                    max_tokens=2048,
                    temperature=0.7,
                    messages=[{"role": "user", "content": creative_prompt + "\n\nANTWORTFORMAT:\n" + json.dumps(creative_schema, ensure_ascii=False)}]
                )
            obj, err = parse_json_safely(response.content[0].text)
            if err:
                return []
            candidates = [c.strip() for c in obj.get("candidates", []) if isinstance(c, str) and c.strip()]

        # Sofort-Filter: verbotene Präfixe rauswerfen
        if forbidden_prefix:
            candidates = [cand for cand in candidates if not starts_with_forbidden(cand, forbidden_prefix)]

        # Validierung der Kandidaten
        valid_candidates = []
        ignore_keywords = [
            'analyse', 'analysier', 'denkschritt', 'mehrwort', 'finale liste',
            'silben', 'vokal', 'ergebnis', 'reimsammlung', 'qualitäts', 'iteration',
            'regel', 'prüfung', 'schema', 'format', 'json', 'beispiel', 'beschreibung', 'erklärung',
            'alle sind deutsche wörter', 'keine beginnt mit gegen'
        ]

        # Vollständige Validierung mit allen Checks
        base_schwa = get_schwa_suffix(input_word)  # NEU: sichtbares Suffix des Basisworts
        seen_signatures = {}  # Varianz-Zähler je Reimfamilie
        MAX_PER_FAMILY = 1  # Default für Familien-Kappung (strikte Einmaligkeit)

        for line in candidates:
            line = line.strip()
            if not line or any(keyword in line.lower() for keyword in ignore_keywords):
                continue

            # Meta-Elemente überspringen
            if line.startswith('#') or line.startswith('##') or re.match(r'^(?:[-=]{3,}|>{1,}|\[.+\]:)', line):
                continue
            if any(ch in line for ch in ['{','}','[',']','`']) or 'http' in line.lower():
                continue

            # Bereinigung
            cleaned_line = re.sub(r'^[-*•]\s*', '', line)
            cleaned_line = re.sub(r'^\d+\.\s*', '', cleaned_line)
            cleaned_line = cleaned_line.strip().replace('*', '')
            cleaned_line = re.sub(r'[,:;–—-]+$', '', cleaned_line).strip()

            # Bindestrich-Trennung
            if " - " in cleaned_line:
                cleaned_line = cleaned_line.split(" - ")[0].strip()
            
            # Finale Prüfung
            if not cleaned_line or ':' in cleaned_line:
                continue

            # NEU: Schwa-Suffix muss übereinstimmen
            if get_schwa_suffix(cleaned_line) != base_schwa:
                continue

            # NEU: Erster Vokal (Familie) muss passen
            if not first_vowel_family_match(input_word, cleaned_line):
                    continue

            # NEU: Kern (betonter Endvokal) muss passen
            if not vowel_core_compatible(input_word, cleaned_line):
                    continue

            # NEU: Silbenzahl muss exakt stimmen
            if count_syllables(cleaned_line) != base_syllables:
                    continue

            # NEU: Varianzlimit (Hamming-Distanz für Vokalfolge)
            if var_limit == 0:  # Strikte Sequenz-Prüfung
                if not _seq_ok(input_word, cleaned_line, tolerance=0):
                    continue

            # NEU: Varianzlimit (max 2 pro Reimfamilie)
            sig = rhyme_signature_fine(cleaned_line)
            if seen_signatures.get(sig, 0) >= MAX_PER_FAMILY:
                continue
            seen_signatures[sig] = seen_signatures.get(sig, 0) + 1

            # NEU: Lexikon-Gate
            is_phrase = bool(re.search(r"\s", cleaned_line))
            if is_phrase:
                if not is_valid_phrase(cleaned_line, freq_thresh=2.5):
                        continue

            # Grundlegende Validität
            if is_valid_phrase(cleaned_line) if is_phrase else is_german_word_or_compound(cleaned_line):
                valid_candidates.append(cleaned_line)

            if len(valid_candidates) >= MAX_RESULTS:
                break

        # NEU: Harte Deduplizierung über 2-Silben-Endsilben
        if valid_candidates:
            seen_tail = set()
            unique = []
            for c in valid_candidates:
                k = _two_syllable_tail_key(c)
                if k in seen_tail:
                        continue
                seen_tail.add(k)
                unique.append(c)
            valid_candidates = unique

        return valid_candidates

    except Exception as e:
        logger.error(f"Fehler in generate_single_batch: {e}")
        return []


@app.route('/api/rhymes', methods=['POST'])
def find_rhymes_endpoint():
    """
    Universeller Reim-Endpoint mit Vokal- und Längenanalyse
    """
    try:
        data = request.get_json(force=True) or {}
        input_word = (data.get("input") or data.get("word") or "").strip()
        
        if not input_word:
            return jsonify({"error": "Kein Eingabewort gefunden"}), 400

        logger.info(f"Analysiere Reime für: '{input_word}'")

        # Phonetische Analyse
        def extract_vowel_pattern(word):
            vowels = []
            for char in word.lower():
                if char in 'aeiouäöü':
                    vowels.append(char)
            return '-'.join(vowels)

        def analyze_vowel_length(word):
            word = word.lower()
            vowel_lengths = []
            i = 0
            while i < len(word):
                if word[i] in 'aeiouäöü':
                    if i + 1 < len(word):
                        next_chars = word[i+1:i+3]
                        if (len(next_chars) >= 2 and 
                            next_chars[0] in 'bcdfghjklmnpqrstvwxyz' and 
                            next_chars[1] in 'bcdfghjklmnpqrstvwxyz'):
                            vowel_lengths.append('kurz')
                        else:
                            vowel_lengths.append('lang')
                    else:
                        vowel_lengths.append('lang')
                i += 1
            return vowel_lengths

        def count_syllables_simple(word):
            return len([char for char in word.lower() if char in 'aeiouäöü'])

        # Analysiere Eingabewort
        vowel_pattern = extract_vowel_pattern(input_word)
        syllable_count = count_syllables_simple(input_word)
        vowel_lengths = analyze_vowel_length(input_word)
        length_pattern = '-'.join(vowel_lengths)

        logger.info(f"Vokalfolge: {vowel_pattern}, Silben: {syllable_count}, Längen: {length_pattern}")

        # Präziserer Prompt ohne Erklärungen
        prompt = f'''Liste 20 deutsche Wörter auf. Jedes Wort muss:
- Genau {syllable_count} Silben haben
- Exakt die Vokalfolge "{vowel_pattern}" haben
- Das Vokallängenmuster "{length_pattern}" befolgen

Format: Ein Wort pro Zeile, keine Erklärungen, keine Nummerierung.

Beispiel für korrektes Format:
Holzfassade
Volksballade
Goldanlage'''
        
        logger.info(f"Sende Prompt an Gemini: {prompt}")

        # Gemini 2.5 Pro (via neues google-genai API)
        response = GENAI.models.generate_content(
            model=os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-pro"),
            contents=prompt
        )
        
        logger.info(f"📨 Gemini Rohe Antwort: {response.text}")
        
        # Verbesserte Filterung der Gemini-Antwort
        lines = []
        for line in response.text.split('\n'):
            line = line.strip()
            # Überspringe leere Zeilen und offensichtliche Erklärungen
            if (not line or 
                line.startswith('#') or 
                line.startswith('Hier sind') or
                line.startswith('Ein wichtiger') or
                line.startswith('Dein Beispielwort') or
                line.startswith('Ich gehe davon') or
                line.lower().startswith('**') or
                ':' in line or
                len(line.split()) > 3):  # Mehr als 3 Wörter = wahrscheinlich Erklärung
                continue
                
            # Entferne Nummerierung und Aufzählungszeichen
            line = re.sub(r'^\d+\.?\s*', '', line)
            line = re.sub(r'^[-•*]\s*', '', line)
            line = re.sub(r'["""„"«»]', '', line)  # Entferne Anführungszeichen
            line = line.strip()
            
            # Nur echte Einzelwörter oder kurze Komposita
            if line and len(line) > 2 and len(line.split()) <= 2:
                lines.append(line)
        
        # Diversitäts-Analyse und zweiter Durchgang
        def get_suffix(word, length=4):
            return word.lower()[-length:]
        
        def get_prefix(word, length=4):  
            return word.lower()[:length]

        # Analysiere erste Runde auf Diversität
        suffix_counts = {}
        prefix_counts = {}
        for line in lines:
            suffix = get_suffix(line)
            prefix = get_prefix(line)
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

        # Filtere: Max 2 pro Endung
        diverse_lines = []
        used_suffixes = {}
        for line in lines:
            suffix = get_suffix(line)
            count = used_suffixes.get(suffix, 0)
            if count < 2:
                diverse_lines.append(line)
                used_suffixes[suffix] = count + 1

        logger.info(f"Nach Diversitäts-Filter: {len(diverse_lines)} von {len(lines)} Kandidaten")

        # ZWEITER DURCHGANG wenn < 20 diverse Ergebnisse
        if len(diverse_lines) < 20:
            needed = 20 - len(diverse_lines)
            forbidden_suffixes = [suffix for suffix, count in used_suffixes.items() if count >= 2]
            
            logger.info(f"Starte zweiten Durchgang für {needed} weitere Kandidaten")
            logger.info(f"Verbiete Endungen: {forbidden_suffixes}")
            
            prompt_2 = f'''Liste {needed + 10} weitere deutsche Wörter auf. Jedes Wort muss:
- Genau {syllable_count} Silben haben  
- Exakt die Vokalfolge "{vowel_pattern}" haben
- Das Vokallängenmuster "{length_pattern}" befolgen
- NICHT auf diese Endungen enden: {", ".join(forbidden_suffixes)}

Format: Ein Wort pro Zeile, keine Erklärungen.'''

            response_2 = GENAI.models.generate_content(
                model=os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-pro"),
                contents=prompt_2
            )
            
            if hasattr(response_2, 'text') and response_2.text:
                # Parse zweite Runde mit gleicher Logik
                lines_2 = []
                for line in response_2.text.split('\n'):
                    line = line.strip()
                    if (not line or line.startswith('#') or line.startswith('Hier sind') or 
                        ':' in line or len(line.split()) > 3):
                        continue
                    line = re.sub(r'^\d+\.?\s*', '', line)
                    line = re.sub(r'^[-•*]\s*', '', line)
                    line = line.strip()
                    if line and len(line) > 2:
                        # Prüfe Diversität
                        suffix = get_suffix(line)
                        if used_suffixes.get(suffix, 0) < 2:
                            lines_2.append(line)
                            used_suffixes[suffix] = used_suffixes.get(suffix, 0) + 1

                diverse_lines.extend(lines_2[:needed])
                logger.info(f"Nach zweitem Durchgang: {len(diverse_lines)} Kandidaten gesamt")

        rhymes = [{"rhyme": line} for line in diverse_lines[:20]]
        logger.info(f"✅ Ergebnis: {len(rhymes)} diverse Reime gefunden")
        
        return jsonify({"rhymes": rhymes}), 200
        
    except Exception as e:
        logger.error(f"Fehler: {e}", exc_info=True)
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

        # TODO: Implement rhyme line generation logic here
        return jsonify({"error": "Not implemented yet"}), 501

    except Exception as e:
        logger.error(f"Fehler in generate_rhyme_line_endpoint: {e}")
        return jsonify({"error": str(e)}), 500


