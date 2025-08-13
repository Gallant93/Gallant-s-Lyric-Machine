import os
import json
import logging
import sys
import time
import random
from typing import Union
import re

import google.generativeai as genai
from flask import Flask, jsonify, request
from flask_cors import CORS
from google.generativeai.types import GenerationConfig

from dotenv import load_dotenv

load_dotenv()

# === Logging Konfiguration ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# === Initialisierung der Dienste ===
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY: raise ValueError("FATAL: GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

app = Flask(__name__)
# Die CORS-Konfiguration erlaubt Anfragen vom Frontend, sowohl lokal als auch deployed.
CORS(app, origins=[r"https://.*\.run\.app", "http://localhost:3000", "http://localhost:5173"],
     supports_credentials=True)


# === Helper-Funktionen ===
def make_audio_part(base64_audio: str, mime_type: str):
    """Erstellt den inlineData-Teil für multimodale Prompts."""
    return {"inline_data": {"mime_type": mime_type, "data": base64_audio}}


def call_gemini(prompt: Union[str, list], schema: dict = None, is_json_output: bool = True):
    """
    Ruft die Gemini API auf. Wenn ein Schema vorhanden ist, wird JSON-Output erzwungen.
    """
    try:
        if schema:
            config = GenerationConfig(response_mime_type="application/json", response_schema=schema)
            response = model.generate_content(prompt, generation_config=config)
            return json.loads(response.text)
        else:
            response = model.generate_content(prompt)
            # Falls trotzdem JSON erwartet wird (z.B. bei komplexen Anweisungen ohne striktes Schema)
            if is_json_output:
                # Bereinigen des Textes, um Markdown-Formatierung zu entfernen
                clean_text = response.text.strip().replace("```json", "").replace("```", "")
                return json.loads(clean_text)
            return response.text
    except Exception as e:
        logger.error(f"GEMINI_API_ERROR: {e}\nPrompt: {prompt}", exc_info=True)
        # Versuche, eine genauere Fehlermeldung aus der Gemini-Antwort zu extrahieren, falls vorhanden
        error_details = str(e)
        if 'response' in locals() and hasattr(response, 'prompt_feedback'):
            error_details = f"Blocked by API due to: {response.prompt_feedback}"
        raise ValueError(f"Error calling Gemini API. Details: {error_details}")


def get_phonetic_breakdown_py(word: str) -> dict:
    """Eine Python-Implementierung der phonetischen Analyse."""
    temp_word = word.lower()

    if temp_word.endswith('er'):
        temp_word = temp_word[:-2] + 'a'

    # Schritt 1: Deutsche Eigenheiten normalisieren
    temp_word = re.sub(r'([aouäöü])h', r'\1', temp_word)  # Dehnungs-h
    temp_word = temp_word.replace('ch', 'X').replace('sch', 'Y').replace('ck', 'k')

    # Schritt 2: Diphthonge und Vokale in der Reihenfolge ihrer Länge suchen
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

    return {
        "syllableCount": len(vowel_sequence_list),
        "vowelSequence": "-".join(vowel_sequence_list)
    }


# === API Endpunkte ===

@app.route('/')
def index():
    return "Gallant's Lyric Machine Backend is running!"


@app.route('/api/analyze-audio', methods=['POST'])
def analyze_audio_endpoint():
    """
    Schritt 1 der Analyse: Transkribiert Audio und führt eine schnelle Voranalyse durch. [cite: 147]
    """
    try:
        data = request.get_json()
        audio_part = make_audio_part(data['base64Audio'], data['mimeType'])

        prompt = "Transcribe the following audio. Then, suggest a short, catchy title for the song and determine if the performance style is 'sung', 'rapped', or 'unknown'. Only return a single JSON object."

        schema = {
            "type": "object",
            "properties": {
                "lyrics": {"type": "string"},
                "title": {"type": "string"},
                "performanceStyle": {"type": "string", "enum": ["sung", "rapped", "unknown"]}
            },
            "required": ["lyrics", "title", "performanceStyle"]
        }

        result = call_gemini([audio_part, prompt], schema=schema)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in /api/analyze-audio: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# === ZU ERSETZENDER BLOCK ===

@app.route('/api/deep-analyze', methods=['POST'])
def deep_analyze_endpoint():
    """
    Schritt 3 der Analyse: Führt die zentrale Analyse aus, um die Künstler-DNA zu extrahieren.
    """
    try:
        data = request.get_json()
        lyrics = data['lyrics']
        knowledge_base = data['knowledgeBase']  # Die Künstler-DNA

        prompt = f"""
        **System Instruction: You are a world-class music analyst. Your task is to perform a deep analysis of the provided lyrics, considering the user's existing artistic DNA.**

        **ARTISTIC DNA (Your primary rule set):**
        ---
        {knowledge_base}
        ---

        **LYRICS TO ANALYZE:**
        ---
        {lyrics}
        ---

        **Perform the following analyses and return a single JSON object with the specified keys:**

        1.  **`formattedLyrics`**: Structure the lyrics by adding `[Verse]`, `[Chorus]`, etc., tags where appropriate.
        2.  **`characterTraits`**: Derive an array of 3-4 key themes, moods, or character traits as concise strings.
        3.  **`technicalSkills`**: Identify an array of 3-4 specific songwriting techniques (e.g., "Metaphor usage", "AABB rhyme scheme", "Narrative perspective") as concise strings.
        4.  **`emphasisPattern`**: Provide a descriptive text analyzing the phonetic patterns and stressed syllables.
        5.  **`rhymeFlowPattern`**: Provide a descriptive text analyzing the rhyme scheme's effect on the song's flow.
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

        result = call_gemini(prompt, schema=schema)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in /api/deep-analyze: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


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
        **Systemanweisung: Du bist die 'Lyric Machine' KI, ein Meister-Trainer. Deine Aufgabe ist es, aus dem Gespräch mit dem Künstler zu lernen.**
        **ANTWORTE AUSSCHLIESSLICH AUF DEUTSCH.**
        **DEINE ANTWORT MUSS EIN EINZELNES JSON-OBJEKT SEIN.**

        **ANALYSE-AUFGABE:**
        Analysiere die letzte Nachricht des Benutzers. Wenn du daraus eine neue, allgemeingültige Regel für die Künstler-DNA lernst (z.B. über Stil, Themen, Wortwahl, Reim-Strukturen), fasse diese Regel in einem strukturierten `rule_category`-Objekt zusammen. Gib der Kategorie und der Regel einen passenden Titel.

        Wenn du keine klare Regel lernst, führe nur das Gespräch weiter und gib KEIN `learningObject` aus.

        **Beantworte danach die letzte Nachricht des Benutzers im "reply"-Feld.**
        """

        # Vereinfachtes Schema, das nur noch rule_category kennt
        trainer_response_schema = {
            "type": "object",
            "properties": {
                "reply": {
                    "type": "string",
                    "description": "Deine textliche Antwort an den Benutzer für den Chat."
                },
                "learningObject": {
                    "type": "object",
                    "description": "Eine allgemeine Wissensregel. NUR ausgeben, wenn eine Regel gelernt wurde.",
                    "properties": {
                        "type": {"type": "string", "enum": "rule_category"},
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
                }
            }, "required": ["reply"]
        }

        result = call_gemini(prompt, schema=trainer_response_schema)

        learning_object = result.get('learningObject')
        if learning_object:
            # Füge die IDs hinzu, so wie es ursprünglich war
            learning_object['id'] = f"rcat-{int(time.time() * 1000)}-{random.randint(100, 999)}"
            for rule in learning_object.get('rules', []):
                rule['id'] = f"rule-{int(time.time() * 1000)}-{random.randint(100, 999)}"

        return jsonify({"reply": result.get('reply'), "learning": learning_object}), 200

    except Exception as e:
        logger.error(f"Error in /api/trainer-chat: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/rhymes', methods=['POST'])
def find_rhymes_endpoint():
    """
    Dieser Endpunkt verwendet einen detaillierten, kreativen Prompt, um eine breite Palette
    von Reim-Kandidaten zu generieren. Ein nachgeschalteter Python-Validator sorgt für die
    Einhaltung der harten phonetischen Regeln.
    """
    try:
        data = request.get_json()
        input_word = data.get('input_word') or data.get('input')
        knowledge_base = data.get('knowledge_base', 'Keine DNA vorhanden.')

        logger.info(f"--- Reim-Anfrage gestartet für: '{input_word}' ---")
        if not input_word:
            logger.error("Fehler: Kein 'input_word' im Request gefunden.")
            return jsonify({"error": "input_word is missing"}), 400

        # === PRÄ - ANALYSE FÜR EINEN PRÄZISEN PROMPT ===
        target_analysis = get_phonetic_breakdown_py(input_word)
        target_syllables = target_analysis.get("syllableCount")
        target_vowels = target_analysis.get("vowelSequence")
        logger.info(f"Vor-Analyse für '{input_word}': Silben={target_syllables}, Vokale='{target_vowels}'")

        # === SCHRITT 1: DER KREATIVE PROMPT AN DIE KI ===
        # Dieser Prompt wurde direkt aus den Nutzeranweisungen übernommen, um maximale Kreativität zu fördern.
        creative_prompt = f"""
        Du bist ein deutscher Reim-Experte. 

        EINGABEWORT: "{input_word}"
        PHONETISCHE ANALYSE: {target_syllables} Silben, Vokalfolge: {target_vowels}

        AUFGABE: Finde 20-30 deutsche Wörter oder Phrasen, die sich auf "{input_word}" reimen.

        WICHTIGE REGELN:
        1. JEDER Reim muss EXAKT {target_syllables} Silben haben
        2. JEDER Reim muss EXAKT die Vokalfolge "{target_vowels}" haben
        3. Keine identischen Wörter zum Eingabewort
        4. Verschiedene Wortarten verwenden (Nomen, Verben, Adjektive)
        5. Auch Phrasen sind erlaubt (z.B. "mehr Licht")

        BEISPIELE FÜR KORREKTE REIME:
        - Geschichte (3 Silben, e-i-e) → "errichtet", "vernichten", "mehr Lichter", "der Richter"
        - Feuer (2 Silben, eu-e) → "teuer", "scheuer", "neuer", "treuer"
        - Katze (2 Silben, a-e) → "Matze", "Tatze", "Platze", "kratze"

        SCHRITT-FÜR-SCHRITT:
        1. Zähle die Silben deiner Kandidaten: {target_syllables}
        2. Prüfe die Vokalfolge deiner Kandidaten: {target_vowels}
        3. Nur Kandidaten ausgeben, die beide Kriterien erfüllen

        Antworte nur mit einem JSON-Objekt: {{"candidates": ["reim1", "reim2", ...]}}
        """

        schema_creative = {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["candidates"]
        }

        logger.info(f"Rhyme-Gen: Asking 'Creative' with new detailed prompt for '{input_word}'...")
        # Wir rufen die KI mit dem neuen, kreativen Prompt auf
        creative_result = call_gemini(creative_prompt, schema=schema_creative)
        logger.info(f"Rohe Antwort von Gemini: {creative_result}")
        candidates = creative_result.get("candidates", [])
        logger.info(f"{len(candidates)} Kandidaten von Gemini extrahiert.")

        if not candidates:
            logger.warning(f"No candidates returned from Gemini for '{input_word}'.")
            return jsonify({"rhymes": []}), 200

        # === SCHRITT 2: DER "PYTHON-RICHTER" MIT FESTEM REGELBUCH ===
        # Dieser Teil bleibt bestehen, um die Qualität und Einhaltung der Regeln sicherzustellen.
        valid_rhymes = []
        target_analysis = get_phonetic_breakdown_py(input_word)
        target_word_lower = input_word.lower()

        logger.info(f"Validator starting for '{input_word}' with analysis: {target_analysis}")

        for candidate in candidates:
            candidate_lower = candidate.lower()
            logger.info(f"  - Prüfe Kandidat: '{candidate}'")

            # Regel 1: Exakte Übereinstimmung
            if target_word_lower == candidate_lower:
                logger.warning(f"    - VERWORFEN: Exakt identisch mit Eingabewort.")
                continue

            # Regel 2: Eingabewort ist ein Substring (z.B. "Geschichte" in "mit Geschichte")
            if len(target_word_lower) > 3 and target_word_lower in candidate_lower:
                logger.warning(f"    - VERWORFEN: Enthält das exakte Eingabewort.")
                continue

            candidate_analysis = get_phonetic_breakdown_py(candidate)
            logger.info(f"    - Analyse des Kandidaten: {candidate_analysis}")

            # Regel: Silbenzahl und Vokalfolge müssen exakt übereinstimmen
            if (candidate_analysis.get("syllableCount") != target_analysis.get("syllableCount") and
                    logger.warning(f"    - VERWORFEN: Falsche Silbenanzahl (ist: {candidate_analysis.get('syllableCount')}, soll: {target_analysis.get('syllableCount')}")):
                    continue

            if candidate_analysis.get("vowelSequence") != target_analysis.get("vowelSequence"):
                logger.warning(f"    - VERWORFEN: Falsche Vokalfolge (ist: '{candidate_analysis.get('vowelSequence')}', soll: '{target_analysis.get('vowelSequence')}').")
                continue

                # Wenn alle Prüfungen bestanden wurden:
                explanation = f"Erfüllt die festen Regeln für Silbenzahl ({target_analysis.get('syllableCount')}) und Vokal-Muster ({target_analysis.get('vowelSequence')})."
                valid_rhymes.append({"rhyme": candidate, "explanation": explanation})
                logger.info(f"    - AKZEPTIERT: Phonetisch valide.")

        logger.info(f"Rhyme-Gen: Python validator approved {len(valid_rhymes)} of {len(candidates)} candidates for '{input_word}'.")
        # Zur Sicherheit wird die Liste der validen Reime gemischt, um bei jeder Anfrage eine andere Reihenfolge zu erhalten
        random.shuffle(valid_rhymes)
        return jsonify({"rhymes": valid_rhymes}), 200

    except Exception as e:
        logger.error(f"Error in /api/rhymes: {e}", exc_info=True)
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
        result = call_gemini([prompt, audio_part], schema=schema)
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
        result = call_gemini(prompt, schema=schema)
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
        result = call_gemini(prompt, schema=schema)
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
            **System Instruction: You are a creative songwriter. Your task is to write a new song that strictly adheres to the provided artistic DNA and user prompt.**

            **ARTIST'S DNA (Your Rule Set):**
            ---
            {knowledge_base}
            ---

            **USER'S PROMPT:**
            - Theme: {data.get('generationPrompt', 'No theme provided.')}
            - Musical Context: {data.get('beatDescription', 'No beat description.')}
            - Performance Style: {data.get('performanceStyle', 'Not specified.')}

            **TASK: Write a complete song text (verses, chorus, etc.) that matches the prompt and perfectly embodies the Artist's DNA.**
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

            **SOURCE SONGS TO FUSE:**
            ---
            {source_songs_text}
            ---

            **TASK: Create a new, hybrid 'child song' that believably merges the themes, imagery, and styles of the source songs.**
            The final product must be a single, coherent piece of art that still respects the global rules of the Artist's DNA. [cite: 65]
            """
        else:
            return jsonify({"error": "Invalid mode specified."}), 400

        # For generation, we expect a simple text response
        result_text = call_gemini(prompt, is_json_output=False)
        return jsonify({"generatedText": result_text}), 200

    except Exception as e:
        logger.error(f"Error in /api/generate-lyrics: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500