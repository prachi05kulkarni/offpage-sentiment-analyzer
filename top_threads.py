"""
Helpers to compute "top threads", per-thread sentiment time series, and per-entity spike detection.

This file expects your processed JSONL to include:
  - "text" or "title"
  - "entity" and/or "entity_score" (optional; will fallback to matching text)
  - "sentiment_label" (POSITIVE / NEGATIVE / NEUTRAL)
  - "processed_at" (unix seconds). If missing, will try "created_utc" field.

Primary functions:
  - get_top_threads_by_entity(jsonl_path, entity=None, top_n=10, ...)
  - get_thread_sentiment_over_time(jsonl_path, thread_id, interval_seconds=60, ...)
  - detect_entity_spikes(jsonl_path, lookback_seconds=3600, baseline_hours=24, ...)
"""
from typing import List, Dict, Optional, Tuple
import json
import os
import logging
import time
from collections import defaultdict, Counter
from math import floor

from utils import match_entity, get_canonical_entities

logger = logging.getLogger(__name__)


def _thread_key(record: Dict) -> Tuple[str, str]:
    """
    Compute a stable key for grouping mentions into a "thread".
    Preference order: url -> (subreddit + title) -> title -> id
    Returns (thread_id, display_title)
    """
    url = record.get("url") or ""
    title = (record.get("title") or record.get("text") or "")[:200]
    subreddit = record.get("subreddit") or ""
    rec_id = record.get("id") or ""
    if url:
        return (f"url:{url}", title)
    if subreddit and title:
        return (f"sr:{subreddit}:{title[:80]}", f"{subreddit} — {title}")
    if title:
        return (f"title:{title[:80]}", title)
    return (f"id:{rec_id}", rec_id)


def _get_timestamp(record: Dict) -> float:
    """
    Prefer processed_at (unix seconds). Fall back to created_utc if available.
    Returns float unix seconds; if none found, returns 0.0
    """
    for k in ("processed_at", "processedAt", "created_utc", "createdAt", "timestamp"):
        v = record.get(k)
        if v:
            try:
                return float(v)
            except Exception:
                try:
                    # sometimes created_utc is an int string
                    return float(int(v))
                except Exception:
                    continue
    return 0.0


def _get_record_entity(record: Dict, min_score: float = 50.0) -> Optional[str]:
    """
    Determine canonical entity for a record:
      - Use existing 'entity' if entity_score >= min_score
      - Otherwise use match_entity(text, threshold=min_score)
      - Return None if no match
    """
    ent = record.get("entity")
    try:
        score = float(record.get("entity_score", 0) or 0)
    except Exception:
        score = 0.0
    if ent and score >= min_score:
        return ent
    text = (record.get("text") or "") or (record.get("title") or "")
    m = match_entity(text, threshold=min_score)
    if m:
        return m[0]
    return None


def get_top_threads_by_entity(
    jsonl_path: str,
    entity: Optional[str] = None,
    top_n: int = 10,
    min_entity_score: float = 50.0,
    limit: int = 10000,
) -> List[Dict]:
    """
    Return a list of top threads mentioning `entity`. If entity is None, return top threads across all entities.
    Each thread dict includes: thread_id, title, url, count, per_entity_counts, sample_texts, sentiment_breakdown
    """
    if not os.path.exists(jsonl_path):
        logger.warning("Processed JSONL not found: %s", jsonl_path)
        return []

    buckets = {}
    thread_entity_counts = defaultdict(Counter)
    thread_samples = defaultdict(list)
    thread_meta = {}

    read = 0
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            read += 1
            if read > limit:
                break

            tid, title = _thread_key(obj)
            if tid not in buckets:
                buckets[tid] = []
                thread_meta[tid] = {"title": title, "url": obj.get("url") or "", "subreddit": obj.get("subreddit") or ""}

            rec_entity = _get_record_entity(obj, min_score=min_entity_score)
            if rec_entity:
                thread_entity_counts[tid][rec_entity] += 1

            sent = (obj.get("sentiment_label") or "NEUTRAL").upper()
            thread_samples[tid].append({"text": obj.get("text") or obj.get("title") or "", "sentiment": sent})

    threads = []
    for tid, meta in thread_meta.items():
        total_mentions = sum(thread_entity_counts[tid].values())
        if entity:
            if thread_entity_counts[tid].get(entity, 0) == 0:
                continue
        s_counter = Counter([s["sentiment"] for s in thread_samples[tid]])
        threads.append(
            {
                "thread_id": tid,
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "subreddit": meta.get("subreddit", ""),
                "count": total_mentions,
                "per_entity_counts": dict(thread_entity_counts[tid]),
                "sample_texts": [s["text"] for s in thread_samples[tid][:3]],
                "sentiment_breakdown": dict(s_counter),
            }
        )

    threads.sort(key=lambda x: (-x["count"], x["title"] or ""))
    return threads[:top_n]


def get_thread_sentiment_over_time(
    jsonl_path: str,
    thread_id: str,
    interval_seconds: int = 60,
    lookback_seconds: Optional[int] = None,
    min_entity_score: float = 50.0,
    limit: int = 100000,
) -> Dict:
    """
    For a given thread_id (the same key returned by _thread_key), return a time-series
    of sentiment counts binned by interval_seconds.

    Returns dict:
      {
        "interval_seconds": interval_seconds,
        "bins": [timestamp0, timestamp1, ...],  # unix seconds (start of bin)
        "series": {
           "POSITIVE": [n0, n1, ...],
           "NEGATIVE": [...],
           "NEUTRAL": [...],
        }
      }

    If lookback_seconds is set, only data from now-lookback_seconds..now is considered.
    """
    if not os.path.exists(jsonl_path):
        logger.warning("Processed JSONL not found: %s", jsonl_path)
        return {}

    now = time.time()
    earliest_allowed = now - lookback_seconds if lookback_seconds else 0.0

    # gather timestamps and sentiments for records that belong to thread_id
    records = []
    read = 0
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            read += 1
            if read > limit:
                break
            tid, _ = _thread_key(obj)
            if tid != thread_id:
                continue
            ts = _get_timestamp(obj)
            if ts < earliest_allowed:
                continue
            sent = (obj.get("sentiment_label") or "NEUTRAL").upper()
            records.append((ts, sent))

    if not records:
        return {"interval_seconds": interval_seconds, "bins": [], "series": {"POSITIVE": [], "NEGATIVE": [], "NEUTRAL": []}}

    # determine bin range
    min_ts = min(r[0] for r in records)
    max_ts = max(r[0] for r in records)
    # if lookback_seconds is provided, force range to now-lookback..now
    if lookback_seconds:
        min_ts = max(min_ts, now - lookback_seconds)
        max_ts = now

    # align start to interval
    start_bin = floor(min_ts / interval_seconds) * interval_seconds
    end_bin = floor(max_ts / interval_seconds) * interval_seconds
    num_bins = int((end_bin - start_bin) / interval_seconds) + 1

    bins = [int(start_bin + i * interval_seconds) for i in range(num_bins)]
    series = {"POSITIVE": [0] * num_bins, "NEGATIVE": [0] * num_bins, "NEUTRAL": [0] * num_bins}

    for ts, sent in records:
        bin_idx = int(floor((ts - start_bin) / interval_seconds))
        if 0 <= bin_idx < num_bins:
            if sent.startswith("POS"):
                series["POSITIVE"][bin_idx] += 1
            elif sent.startswith("NEG"):
                series["NEGATIVE"][bin_idx] += 1
            else:
                series["NEUTRAL"][bin_idx] += 1

    return {"interval_seconds": interval_seconds, "bins": bins, "series": series}


def detect_entity_spikes(
    jsonl_path: str,
    lookback_seconds: int = 3600,
    baseline_hours: int = 24,
    pct_threshold: float = 30.0,
    min_count: int = 10,
    min_entity_score: float = 50.0,
    limit: int = 200000,
) -> Dict[str, Dict]:
    """
    Detect negative spikes per entity.

    Algorithm:
      - recent window: now - lookback_seconds .. now
      - baseline window: now - (lookback_seconds + baseline_hours*3600) .. now - lookback_seconds
      - for each entity, compute:
          - recent_total, recent_negative_count, recent_negative_pct
          - baseline_total, baseline_negative_count, baseline_negative_pct
      - mark is_spike True if recent_negative_pct >= pct_threshold and recent_negative_count >= min_count
        and recent_negative_pct > baseline_negative_pct (simple sanity check)

    Returns dict:
      {
        "EntityName": {
            "is_spike": bool,
            "recent_count": int,
            "recent_negative_count": int,
            "recent_negative_pct": float,
            "baseline_count": int,
            "baseline_negative_count": int,
            "baseline_negative_pct": float,
        },
        ...
      }
    """
    if not os.path.exists(jsonl_path):
        logger.warning("Processed JSONL not found: %s", jsonl_path)
        return {}

    now = time.time()
    recent_start = now - lookback_seconds
    baseline_end = recent_start
    baseline_start = baseline_end - (baseline_hours * 3600)

    # Prepare containers
    per_entity = defaultdict(lambda: {"recent_total": 0, "recent_neg": 0, "baseline_total": 0, "baseline_neg": 0})

    read = 0
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            read += 1
            if read > limit:
                break

            ts = _get_timestamp(obj)
            if ts <= 0:
                # skip items with no timestamp - they can't be localized in time
                continue

            # decide which window (recent / baseline / ignore)
            if ts >= recent_start and ts <= now:
                window = "recent"
            elif ts >= baseline_start and ts < baseline_end:
                window = "baseline"
            else:
                continue

            # determine entity
            ent = _get_record_entity(obj, min_score=min_entity_score)
            if not ent:
                continue

            sent = (obj.get("sentiment_label") or "NEUTRAL").upper()
            if window == "recent":
                per_entity[ent]["recent_total"] += 1
                if sent.startswith("NEG"):
                    per_entity[ent]["recent_neg"] += 1
            else:
                per_entity[ent]["baseline_total"] += 1
                if sent.startswith("NEG"):
                    per_entity[ent]["baseline_neg"] += 1

    # compute metrics and spike decisions
    results = {}
    for ent, vals in per_entity.items():
        r_tot = vals["recent_total"]
        r_neg = vals["recent_neg"]
        b_tot = vals["baseline_total"]
        b_neg = vals["baseline_neg"]

        r_pct = (r_neg / r_tot * 100.0) if r_tot > 0 else 0.0
        b_pct = (b_neg / b_tot * 100.0) if b_tot > 0 else 0.0

        # decide spike: require recent pct >= pct_threshold, recent negative count >= min_count, and improvement over baseline
        is_spike = False
        if r_tot >= min_count and r_pct >= pct_threshold and r_pct > b_pct:
            is_spike = True

        results[ent] = {
            "is_spike": is_spike,
            "recent_count": int(r_tot),
            "recent_negative_count": int(r_neg),
            "recent_negative_pct": float(r_pct),
            "baseline_count": int(b_tot),
            "baseline_negative_count": int(b_neg),
            "baseline_negative_pct": float(b_pct),
        }

    # ensure all canonical entities are present even if zero
    for ent in get_canonical_entities():
        if ent not in results:
            results[ent] = {
                "is_spike": False,
                "recent_count": 0,
                "recent_negative_count": 0,
                "recent_negative_pct": 0.0,
                "baseline_count": 0,
                "baseline_negative_count": 0,
                "baseline_negative_pct": 0.0,
            }

    return results