import os
import time
import json
import logging
import hashlib
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

from sentiment import analyze_sentiments
from utils import match_entity
from top_threads import detect_entity_spikes, get_canonical_entities

logger = logging.getLogger("stream_processor")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")

STREAMED_PATH = os.getenv("STREAMED_PATH", "streamed_mentions.jsonl")
PROCESSED_PATH = os.getenv("PROCESSED_PATH", "processed_mentions.jsonl")
CHECKPOINT_PATH = os.getenv("PROCESSOR_CHECKPOINT", ".processor_offset")
BATCH_SIZE = int(os.getenv("PROCESSOR_BATCH_SIZE", "32"))

ENTITY_MATCH_THRESHOLD = float(os.getenv("ENTITY_MATCH_THRESHOLD", "80.0"))
DEDUP_EPOCH_TTL = int(os.getenv("DEDUP_TTL_SECONDS", str(60 * 60)))

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")
NEG_SPIKE_PERCENT = float(os.getenv("NEGATIVE_SPIKE_PERCENT_THRESHOLD", "30"))
NEG_SPIKE_MIN_COUNT = int(os.getenv("NEGATIVE_SPIKE_MIN_COUNT", "10"))

ENABLE_ENTITY_ALERTS = os.getenv("ENABLE_ENTITY_ALERTS", "false").lower() in ("1", "true", "yes")
ALERT_CHECK_INTERVAL_SECONDS = int(os.getenv("ALERT_CHECK_INTERVAL_SECONDS", "300"))
ALERT_LOOKBACK_SECONDS = int(os.getenv("ALERT_LOOKBACK_SECONDS", str(3600)))
ALERT_BASELINE_HOURS = int(os.getenv("ALERT_BASELINE_HOURS", str(24)))
ALERT_PCT_THRESHOLD = float(os.getenv("ALERT_PCT_THRESHOLD", str(NEG_SPIKE_PERCENT)))
ALERT_MIN_NEG_COUNT = int(os.getenv("ALERT_MIN_NEG_COUNT", str(NEG_SPIKE_MIN_COUNT)))
ALERT_MIN_ENTITY_SCORE = float(os.getenv("ALERT_MIN_ENTITY_SCORE", str(ENTITY_MATCH_THRESHOLD)))
ALERT_STATE_PATH = os.getenv("ALERT_STATE_PATH", ".alert_state.json")

_recent_hashes: Dict[str, float] = {}
_last_alert_check: float = 0.0
_alert_state: Dict[str, Dict] = {}

# --------------------------------------------------------------------
# NEW PATCH 1: Ensure streamed input file exists
# --------------------------------------------------------------------
if not os.path.exists(STREAMED_PATH):
    with open(STREAMED_PATH, "w", encoding="utf-8") as f:
        pass  # create empty file so processor doesn't crash


def _hash_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def dedupe_and_store(text: str) -> bool:
    h = _hash_text(text)
    now = time.time()
    expired = [k for k, v in _recent_hashes.items() if now - v > DEDUP_EPOCH_TTL]
    for k in expired:
        del _recent_hashes[k]

    if h in _recent_hashes:
        return False

    _recent_hashes[h] = now
    return True


def _read_new_lines(path: str, offset: int):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            fh.seek(offset)
            lines = [raw.rstrip("\n") for raw in fh if raw.strip()]
            new_offset = fh.tell()
        return lines, new_offset
    except FileNotFoundError:
        return [], offset


def _load_json_objs(lines: List[str]) -> List[Dict]:
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except:
            logger.debug("Skipping invalid JSON")
    return out


def _write_checkpoint(offset: int):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as fh:
        fh.write(str(offset))


def _read_checkpoint() -> int:
    if os.path.exists(CHECKPOINT_PATH):
        try:
            return int(open(CHECKPOINT_PATH).read().strip() or 0)
        except:
            return 0
    return 0


def _send_slack_alert(text: str):
    if not SLACK_WEBHOOK:
        return
    try:
        import requests
        r = requests.post(SLACK_WEBHOOK, json={"text": text}, timeout=10)
        if r.status_code >= 300:
            logger.warning("Slack call failed: %s %s", r.status_code, r.text)
    except Exception:
        logger.exception("Slack alert error")


def _detect_negative_spike(recent_objs: List[Dict]):
    if not recent_objs:
        return False, 0.0, 0
    total = 0
    neg = 0
    for o in recent_objs:
        lbl = (o.get("sentiment_label") or "").upper()
        if lbl:
            total += 1
            if lbl.startswith("NEG"):
                neg += 1
    if total == 0:
        return False, 0.0, 0
    pct = (neg / total) * 100.0
    return pct >= NEG_SPIKE_PERCENT and neg >= NEG_SPIKE_MIN_COUNT, pct, neg


def _enrich_with_entity(o: Dict):
    text = (o.get("text") or "") or (o.get("title") or "")
    match = match_entity(text, threshold=ENTITY_MATCH_THRESHOLD)
    if match:
        o["entity"], o["entity_score"] = match[0], float(match[1])
    else:
        o["entity"], o["entity_score"] = None, 0.0
    return o


def _load_alert_state():
    global _alert_state
    if os.path.exists(ALERT_STATE_PATH):
        try:
            _alert_state = json.load(open(ALERT_STATE_PATH, "r"))
        except:
            _alert_state = {}
    return _alert_state


def _save_alert_state():
    try:
        json.dump(_alert_state, open(ALERT_STATE_PATH, "w"))
    except:
        pass


def _maybe_check_and_alert():
    global _last_alert_check, _alert_state

    if not ENABLE_ENTITY_ALERTS:
        return

    now = time.time()
    if now - _last_alert_check < ALERT_CHECK_INTERVAL_SECONDS:
        return
    _last_alert_check = now

    _load_alert_state()

    try:
        spikes = detect_entity_spikes(
            PROCESSED_PATH,
            lookback_seconds=ALERT_LOOKBACK_SECONDS,
            baseline_hours=ALERT_BASELINE_HOURS,
            pct_threshold=ALERT_PCT_THRESHOLD,
            min_count=ALERT_MIN_NEG_COUNT,
            min_entity_score=ALERT_MIN_ENTITY_SCORE,
            limit=200000
        )
    except Exception:
        logger.exception("Spike detection failure")
        return

    for ent in get_canonical_entities():
        info = spikes.get(ent, {
            "is_spike": False,
            "recent_count": 0,
            "recent_negative_count": 0,
            "recent_negative_pct": 0.0,
            "baseline_count": 0,
            "baseline_negative_pct": 0.0,
        })

        prev = _alert_state.get(ent, {"is_spike": False})
        cur = info.get("is_spike", False)

        if cur and not prev["is_spike"]:
            _send_slack_alert(f"Negative spike detected for {ent}")
            _alert_state[ent] = {"is_spike": True}
        elif not cur and prev["is_spike"]:
            _send_slack_alert(f"Negative spike cleared for {ent}")
            _alert_state[ent] = {"is_spike": False}

    _save_alert_state()


# --------------------------------------------------------------------
# NEW PATCH 2: Stop infinite loop if no input appears
# --------------------------------------------------------------------
EXIT_IF_NO_DATA = True   # change to False if you want infinite streaming


def run_forever():
    logger.info("Processor starting (JSONL mode)")
    offset = _read_checkpoint()

    if ENABLE_ENTITY_ALERTS:
        _load_alert_state()

    empty_cycles = 0

    while True:
        try:
            lines, new_offset = _read_new_lines(STREAMED_PATH, offset)

            if not lines:
                _maybe_check_and_alert()
                time.sleep(1)

                if EXIT_IF_NO_DATA:
                    empty_cycles += 1
                    if empty_cycles > 3:  # waited ~3 seconds
                        logger.info("No data found. Exiting.")
                        break
                continue

            empty_cycles = 0
            objs = _load_json_objs(lines)

            filtered = []
            for o in objs:
                text = (o.get("text") or "") or (o.get("title") or "")
                if text and not dedupe_and_store(text):
                    continue
                filtered.append(o)

            if not filtered:
                offset = new_offset
                _write_checkpoint(offset)
                continue

            texts = [(o.get("text") or "") or (o.get("title") or "") for o in filtered]
            sentiments = []

            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                try:
                    sentiments.extend(analyze_sentiments(batch))
                except:
                    sentiments.extend([{"label": "NEUTRAL", "score": 0.0} for _ in batch])

            now_ts = time.time()
            with open(PROCESSED_PATH, "a", encoding="utf-8") as pf:
                for o, s in zip(filtered, sentiments):
                    o["sentiment_label"] = s.get("label", "NEUTRAL")
                    o["sentiment_score"] = float(s.get("score", 0.0))
                    o["processed_at"] = now_ts
                    o = _enrich_with_entity(o)
                    pf.write(json.dumps(o, ensure_ascii=False) + "\n")

            _detect_negative_spike(filtered)
            _maybe_check_and_alert()

            offset = new_offset
            _write_checkpoint(offset)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    run_forever()
