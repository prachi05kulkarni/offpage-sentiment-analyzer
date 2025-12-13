import os
import json
import csv
import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# --- JSONL / CSV helpers (kept & slightly hardened) -------------------------

def read_processed_jsonl_if_exists(path: str = "processed_mentions.jsonl", limit: int = 500) -> List[Dict]:
    """
    Read up to `limit` JSON objects from a newline-delimited JSON file.
    Returns an empty list if file doesn't exist or is unreadable.
    """
    if not os.path.exists(path):
        logger.debug("JSONL path does not exist: %s", path)
        return []
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                logger.debug("Skipping invalid JSONL line %d: %s", i, e)
                continue
    return out


def export_jsonl_to_csv(jsonl_path: str, csv_path: str):
    """
    Export processed JSONL to CSV. Will try to include common fields and
    also fallback to nested keys if available.
    """
    fieldnames = [
        "platform",
        "type",
        "id",
        "title",
        "text",
        "url",
        "created_utc",
        "subreddit",
        "entity",
        "entity_score",
        "sentiment_label",
        "sentiment_score",
    ]
    with open(jsonl_path, "r", encoding="utf-8") as inf, open(csv_path, "w", newline="", encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        for line in inf:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            row = {}
            for k in fieldnames:
                # allow several common alternatives for backward compatibility
                if k in obj:
                    row[k] = obj.get(k, "")
                else:
                    # try snake-case / nested or alternate names
                    alt = obj.get(k.lower(), "") or obj.get(k.upper(), "") or obj.get(k.replace("_", ""), "")
                    row[k] = alt if alt is not None else ""
            writer.writerow(row)


# --- Entity matching registry & utilities ----------------------------------

# Default registry: expand this aggressively for better accuracy.
# Add synonyms, product names, twitter handles, common misspellings, etc.
DEFAULT_ENTITY_REGISTRY: Dict[str, List[str]] = {
    "Eureka Forbes": ["Eureka Forbes", "EurekaForbes", "eureka_forbes", "eureka", "eurekaforbes"],
    "Groww": ["Groww", "groww", "groww.in", "@groww", "groww app", "growwapp"],
    "Upstox": ["Upstox", "upstox", "@upstox", "upstox.in"],
    "Zerodha": ["Zerodha", "zerodha", "@zerodha", "kite", "zerodha kite", "zerodha.kite"],
}

# Allow loading an override registry from a JSON file specified by env var ENTITIES_JSON
ENTITIES_JSON_PATH = os.getenv("ENTITIES_JSON", "")

def _load_registry_from_file(path: str) -> Optional[Dict[str, List[str]]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                # ensure lists
                out = {k: (v if isinstance(v, list) else [str(v)]) for k, v in data.items()}
                logger.info("Loaded entity registry from %s", path)
                return out
    except Exception as e:
        logger.warning("Failed to load entity registry from %s: %s", path, e)
    return None


# build active registry at import time
ENTITY_REGISTRY: Dict[str, List[str]] = DEFAULT_ENTITY_REGISTRY.copy()
if ENTITIES_JSON_PATH:
    user_reg = _load_registry_from_file(ENTITIES_JSON_PATH)
    if user_reg:
        # merge user entries, giving user registry priority
        ENTITY_REGISTRY.update(user_reg)

# flattened exact token map for fast substring/token checks
_exact_map: Dict[str, str] = {}
for canonical, syns in ENTITY_REGISTRY.items():
    for s in syns:
        _exact_map[s.lower()] = canonical


# try to import rapidfuzz for fuzzy matching; otherwise fallback to simple heuristics
try:
    from rapidfuzz import fuzz  # type: ignore

    _HAS_RAPIDFUZZ = True
    logger.info("rapidfuzz available for fuzzy entity matching")
except Exception:
    _HAS_RAPIDFUZZ = False
    logger.debug("rapidfuzz not available; falling back to substring matching")


def _clean_text(text: str) -> str:
    """Lowercase and normalize whitespace, optionally remove punctuation for matching."""
    if not text:
        return ""
    t = text.lower()
    # keep alphanum and common symbols (@, .) to help handle handles / domains
    t = re.sub(r"[^\w@\.\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def match_entity(text: str, threshold: float = 80.0) -> Optional[Tuple[str, float]]:
    """
    Attempt to match `text` to a canonical entity.

    Returns (canonical_name, score) if match found, otherwise None.

    Strategy:
      1. Fast exact token/substring checks against synonyms and canonical names.
      2. If rapidfuzz available, compute partial_ratio against each synonym and return best match if >= threshold.
      3. Fallback: canonical-name substring check.
    """
    if not text:
        return None

    t = _clean_text(text)

    # 1) exact substring/token checks (fast, deterministic)
    for token, canonical in _exact_map.items():
        if token in t:
            return canonical, 100.0

    # 2) fuzzy matching
    if _HAS_RAPIDFUZZ:
        best_entity = None
        best_score = 0.0
        for canonical, syns in ENTITY_REGISTRY.items():
            # compute best score for this canonical across its synonyms
            for s in syns:
                s_norm = s.lower()
                try:
                    score = fuzz.partial_ratio(s_norm, t)
                except Exception:
                    score = 0.0
                if score > best_score:
                    best_score = float(score)
                    best_entity = canonical
        if best_entity and best_score >= threshold:
            return best_entity, best_score

    # 3) fallback simple canonical substring check
    for canonical in ENTITY_REGISTRY.keys():
        if canonical.lower() in t:
            return canonical, 90.0

    return None


def expand_entity_registry(new_entries: Dict[str, List[str]], persist_path: Optional[str] = None):
    """
    Merge new_entries into the live ENTITY_REGISTRY. Optionally persist to path (JSON).
    Useful during tuning to programmatically add synonyms/handles.
    """
    global ENTITY_REGISTRY, _exact_map
    for k, syns in new_entries.items():
        if k in ENTITY_REGISTRY:
            # append unique synonyms
            existing = set(ENTITY_REGISTRY[k])
            for s in syns:
                if s not in existing:
                    ENTITY_REGISTRY[k].append(s)
        else:
            ENTITY_REGISTRY[k] = list(syns)
    # rebuild exact map
    _exact_map = {}
    for canonical, syns in ENTITY_REGISTRY.items():
        for s in syns:
            _exact_map[s.lower()] = canonical

    if persist_path:
        try:
            with open(persist_path, "w", encoding="utf-8") as fh:
                json.dump(ENTITY_REGISTRY, fh, indent=2, ensure_ascii=False)
            logger.info("Persisted updated entity registry to %s", persist_path)
        except Exception as e:
            logger.warning("Failed to persist entity registry to %s: %s", persist_path, e)


# --- Debugging / analysis helpers ------------------------------------------

def sample_misassigned_mentions(jsonl_path: str = "processed_mentions.jsonl", sample_size: int = 50) -> List[Dict]:
    """
    Inspect processed mentions and return mentions where entity is None or low confidence,
    or where text contains other canonical names but entity assignment doesn't match.
    This helps find systematic misassignments (e.g., everything assigned to Eureka Forbes).
    """
    candidates: List[Dict] = []
    records = read_processed_jsonl_if_exists(jsonl_path, limit=1000)
    for rec in records:
        text = rec.get("text", "") or rec.get("title", "") or ""
        assigned = rec.get("entity")
        assigned_score = rec.get("entity_score", 0) or 0
        # quick heuristic: pick ones with no entity or low score or assigned to one canonical with suspiciously high share
        if not assigned or float(assigned_score) < 50:
            candidates.append({"id": rec.get("id"), "text": text, "assigned": assigned, "score": assigned_score})
        else:
            # check if text contains another canonical name token
            for canonical, syns in ENTITY_REGISTRY.items():
                for s in syns:
                    if s.lower() in (text or "").lower():
                        # if contained canonical not equal to assigned, mark as candidate
                        if assigned != canonical:
                            candidates.append({"id": rec.get("id"), "text": text, "assigned": assigned, "expected": canonical})
                        break
        if len(candidates) >= sample_size:
            break
    return candidates


# small utility to get canonical names list (useful for UI/inspection)
def get_canonical_entities() -> List[str]:
    return list(ENTITY_REGISTRY.keys())