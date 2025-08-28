# backend/dna_indexer.py 123
from __future__ import annotations
import json, os, re, random
from typing import Dict, List, Tuple, Optional

# ---- Persistenzpfad
def _store_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "dna_index.json")

# ---- Helpers werden vom Backend gebunden
_BINDINGS: Dict[str, object] = {}
def bind_helpers(h: Dict[str, object]): _BINDINGS.update(h)
def _f(name): 
    if name not in _BINDINGS: 
        raise RuntimeError(f"dna_indexer: helper '{name}' not bound")
    return _BINDINGS[name]

# ---- Signatur
def signature(word: str) -> Tuple:
    # ERST normalisieren, dann analysieren
    normalized_word = word
    try:
        # Versuche die Backend-Normalisierung zu verwenden
        if 'normalize_er_schwa' in _BINDINGS:
            normalized_word = _BINDINGS['normalize_er_schwa'](word)
    except:
        pass
    
    count = _f("count_syllables")(normalized_word)
    seq   = tuple(_f("extract_vowel_sequence")(normalized_word))
    fst, fst_len = _f("first_vowel_and_length")(normalized_word)
    core  = _f("inner_stressed_core")(normalized_word)
    last  = _f("last_vowel_family")(normalized_word)
    return (count, seq, fst, fst_len, core, last)

# ---- Laden/Speichern
def _load_index() -> Dict:
    p = _store_path()
    if not os.path.exists(p): return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_index(idx: Dict): 
    with open(_store_path(), "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

# ---- Upsert eines Beispielpaars (wird beim DNA-Feature „Reim beibringen" aufgerufen)
def upsert_example(input_word: str, output_word: str) -> None:
    idx = _load_index()
    sig_in = repr(signature(input_word))

    bucket = idx.setdefault(sig_in, {
        "heads": {},              # rechte Endglieder/Köpfe
        "lefts": {},              # linke Bestandteile
        "phrase_skeletons": {},   # kurze Phrasen-Schablonen
        "verbatim_outputs": {}    # alle originalen DNA-Outputs zum späteren Ausschluss
    })

    # 1) Verbatim Output merken (damit wir ihn später NICHT 1:1 ausgeben)
    o = output_word.strip()
    bucket["verbatim_outputs"][o] = bucket["verbatim_outputs"].get(o, 0) + 1

    # 2) Zerlegung
    toks = re.findall(r"[A-Za-zÄÖÜäöüß\-]+", o)
    low = o.lower()
    if " " in o:
        # Phrase: Skeleton aus 2–5 Wörtern anlegen, Kopf = letztes Inhaltswort
        if toks:
            skeleton = " ".join(toks[-min(5, len(toks)):]).lower()
            head = toks[-1].lower()
            bucket["phrase_skeletons"][skeleton] = bucket["phrase_skeletons"].get(skeleton, 0) + 1
            bucket["heads"][head] = bucket["heads"].get(head, 0) + 1
    else:
        # Einzelwort/Kompositum
        if "-" in low:
            left, _, head = low.rpartition("-")
            left = left.strip(); head = head.strip()
            if left: bucket["lefts"][left] = bucket["lefts"].get(left, 0) + 1
            if head: bucket["heads"][head] = bucket["heads"].get(head, 0) + 1
        else:
            # heuristischer rechter Kopf (3..7)
            for k in range(3, min(7, len(low))+1):
                head = low[-k:]
                bucket["heads"][head] = bucket["heads"].get(head, 0) + 1
            if len(low) > 4:
                left = low[:-3]
                bucket["lefts"][left] = bucket["lefts"].get(left, 0) + 1

    idx[sig_in] = bucket
    _save_index(idx)

# ---- Generator: recombiniert aus Index, NICHT 1:1
def generate_from_index(
    input_word: str,
    max_results: int,
    target_phrase_ratio: float,
    rng_seed: Optional[int] = None,
    pre_sig: Optional[tuple] = None,
) -> List[str]:
    if rng_seed is not None:
        random.seed(rng_seed)

    idx = _load_index()
    # NEU: wenn eine vorgegebene Signatur mitkommt, nutzen wir die – sonst selbst berechnen
    sig_in = repr(pre_sig) if pre_sig is not None else repr(signature(input_word))

    if sig_in not in idx:
        try:
            print(f"[DNA] no bucket for sig={sig_in} available_sigs={list(idx.keys())[:5]}")
        except Exception:
            pass
        return []

    b = idx[sig_in]
    heads  = sorted(b["heads"].items(), key=lambda x: -x[1])
    lefts  = sorted(b["lefts"].items(), key=lambda x: -x[1])
    skels  = sorted(b["phrase_skeletons"].items(), key=lambda x: -x[1])
    verbatim = set(b["verbatim_outputs"].keys())

    want_phr = int(round(target_phrase_ratio * max_results))
    accepted, tried = [], set()

    # benötigte Prüfer aus dem Backend
    passes_seq  = _f("passes_phonetic_sequence_gates")
    passes_syl  = _f("passes_syllable_gate")
    is_word     = _f("is_german_word_or_compound")
    is_phrase   = _f("is_valid_phrase")
    prefix_key  = _f("prefix_family_key")
    cap         = _f("PREFIX_FAMILY_CAP")()

    prefix_counts = {}

    def _ok(c: str) -> bool:
        if c in verbatim:
            print(f"[DNA] reject verbatim: {c}")
            return False
        if not passes_seq(c): 
            print(f"[DNA] reject seq: {c}")
            return False
        if not passes_syl(c): 
            print(f"[DNA] reject syl: {c}")
            return False
        if not (is_phrase(c) if " " in c else is_word(c)): 
            print(f"[DNA] reject lexicon: {c}")
            return False
        k = prefix_key(c)
        if prefix_counts.get(k, 0) >= cap: 
            print(f"[DNA] reject prefix_cap: {c} (key={k})")
            return False
        prefix_counts[k] = prefix_counts.get(k, 0) + 1
        print(f"[DNA] accept: {c}")
        return True

    # 1) Phrasen bis Quote
    for sk, _ in skels:
        if len(accepted) >= max_results: break
        if sum(1 for x in accepted if " " in x) >= want_phr: break
        cand = sk
        if cand in tried: continue
        tried.add(cand)
        if _ok(cand):
            accepted.append(cand)

    # 2) Komposita/Einzelwörter auffüllen
    i = 0
    while len(accepted) < max_results and i < (len(heads)*3 + 50):
        i += 1
        head = heads[min(i-1, len(heads)-1)][0] if heads else None
        left = lefts[min(i-1, len(lefts)-1)][0] if lefts else None
        if head and left:
            cand = f"{left}-{head}" if len(left) > 1 else head
        elif head:
            cand = head
        else:
            break
        if cand in tried: continue
        tried.add(cand)
        if _ok(cand):
            accepted.append(cand)

    # VOR dem return:
    try:
        print(f"[DNA] generated {len(accepted)} candidates: {accepted}")
        print(f"[DNA] tried {len(tried)} total candidates")
        print(f"[DNA] prefix_counts: {prefix_counts}")
    except Exception:
        pass
    
    return accepted[:max_results]
