import logging
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger("analysis")

def compute_share_of_voice(reddit_items, quora_items, brand, competitor=None):
    try:
        brand_lc = brand.lower()
    except Exception:
        brand_lc = str(brand).lower()

    comp_lc = competitor.lower() if competitor else None

    r_brand = sum(1 for it in (reddit_items or []) if brand_lc in (it.get("text", "") or "").lower())
    q_brand = sum(1 for it in (quora_items or []) if brand_lc in (it.get("text", "") or "").lower())
    brand_total = r_brand + q_brand

    if competitor:
        r_comp = sum(1 for it in (reddit_items or []) if comp_lc in (it.get("text", "") or "").lower())
        q_comp = sum(1 for it in (quora_items or []) if comp_lc in (it.get("text", "") or "").lower())
        comp_total = r_comp + q_comp
    else:
        comp_total = 0

    denom = brand_total + comp_total
    sov = {
        "brand_mentions": brand_total,
        "competitor_mentions": comp_total,
        "share_of_voice": (brand_total / denom) if denom > 0 else None
    }
    logger.debug("SOV computed: %s", sov)
    return sov

def top_threads(items, top_n=5):
    scored = []
    for it in (items or []):
        try:
            score = float(it.get("score", 0) or 0)
            comments = float(it.get("comments", 0) or 0)
            s = 0.6 * score + 0.4 * comments
            if s == 0:
                s = len((it.get("text", "") or ""))
            entry = it.copy()
            entry["score"] = s
            scored.append(entry)
        except Exception as e:
            logger.debug("Error scoring item: %s", e)
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored[:top_n]

def generate_recommendations(items, brand, top_n_keywords=8):
    texts = [it.get("text", "") for it in (items or []) if it.get("text")]
    if not texts:
        return ["No mentions found — consider starting brand conversations and monitoring communities."]
    try:
        cv = CountVectorizer(stop_words="english", max_features=top_n_keywords)
        X = cv.fit_transform(texts)
        keywords = cv.get_feature_names_out().tolist()
    except Exception as e:
        logger.debug("Keyword extraction failed: %s", e)
        keywords = []
    recs = []
    if keywords:
        recs.append(f"Top topics/keywords observed: {', '.join(keywords)}")
    recs.append("Include: quick responses on threads with complaints and helpful how-to content.")
    recs.append("Exclude: overt sales pitches. Focus on value and solutions.")
    recs.append("Tone: empathetic on negative clusters; amplify positive user stories.")
    recs.append("Monitoring: set up alerts for spikes and weekly digest for teams.")
    return recs