import logging
import re
from collections import defaultdict

logger = logging.getLogger("analysis")


def _word_match(term, text):
    """Return True if `term` exists as a word in `text` (case-insensitive)."""
    if not term:
        return False
    try:
        # use word boundary to avoid partial matches (e.g., apple != pineapple)
        return re.search(rf"\b{re.escape(term)}\b", (text or "").lower(), flags=re.IGNORECASE) is not None
    except Exception:
        return term.lower() in (text or "").lower()


def compute_share_of_voice(reddit_items, quora_items, brand, competitor=None):
    """
    Compute share of voice with robust word-boundary matching.

    Returns a dict containing:
      - brand_only_mentions: count of mentions that contain brand but not competitor
      - competitor_only_mentions: count of mentions that contain competitor but not brand
      - both_mentions: mentions that include both terms
      - neither_mentions
      - brand_mentions: inclusive count (brand_only + both)
      - competitor_mentions: inclusive count (competitor_only + both)
      - share_of_voice: inclusive (brand / (brand + competitor)) when denom>0
      - share_of_voice_exclusive: exclusive (brand_only / (brand_only + competitor_only)) when denom_excl>0
      - per_platform: breakdown of counts per platform
    """
    try:
        brand_norm = (brand or "").strip().lower()
    except Exception:
        brand_norm = str(brand or "").strip().lower()

    comp_norm = (competitor or "").strip().lower() if competitor else None

    all_items = list((reddit_items or []) + (quora_items or []))

    brand_only = 0
    comp_only = 0
    both = 0
    neither = 0

    per_platform = defaultdict(lambda: {"brand_only": 0, "comp_only": 0, "both": 0, "neither": 0})

    for it in all_items:
        text = (it.get("text") or "") or ""
        # Prefer matched_terms if already present (clients will set this)
        matched = it.get("matched_terms")
        if matched is None:
            has_brand = _word_match(brand_norm, text)
            has_comp = _word_match(comp_norm, text) if comp_norm else False
        else:
            # matched_terms may contain original-cased brand names; normalize by presence flags
            has_brand = "brand" in matched or any(m.lower() == brand_norm for m in matched)
            has_comp = False
            if comp_norm:
                has_comp = "competitor" in matched or any(m.lower() == comp_norm for m in matched)

        platform = it.get("platform", "unknown")

        if has_brand and not has_comp:
            brand_only += 1
            per_platform[platform]["brand_only"] += 1
        elif has_comp and not has_brand:
            comp_only += 1
            per_platform[platform]["comp_only"] += 1
        elif has_brand and has_comp:
            both += 1
            per_platform[platform]["both"] += 1
        else:
            neither += 1
            per_platform[platform]["neither"] += 1

    brand_inclusive = brand_only + both
    comp_inclusive = comp_only + both

    denom_inclusive = brand_inclusive + comp_inclusive
    denom_exclusive = brand_only + comp_only

    sov = {
        "brand_only_mentions": brand_only,
        "competitor_only_mentions": comp_only,
        "both_mentions": both,
        "neither_mentions": neither,
        # backward-compatible keys:
        "brand_mentions": brand_inclusive,
        "competitor_mentions": comp_inclusive,
        "share_of_voice": (brand_inclusive / denom_inclusive) if denom_inclusive > 0 else None,
        # exclusive SOV excludes overlap (both) from denominator:
        "share_of_voice_exclusive": (brand_only / denom_exclusive) if denom_exclusive > 0 else None,
        "per_platform": per_platform,
    }
    logger.debug("SOV computed: %s", sov)
    return sov