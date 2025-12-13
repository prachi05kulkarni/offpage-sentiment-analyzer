# metrics.py
# Utility functions to compute KPI metrics from mention lists.
# Requires: pandas, numpy
import logging
from collections import defaultdict
from datetime import datetime, timezone
import hashlib

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    raise ImportError("metrics.py requires pandas and numpy. Install with: pip install pandas numpy") from e

logger = logging.getLogger("metrics")

def _ensure_datetime(val):
    # Accept unix timestamp (seconds) or iso string or datetime
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.replace(tzinfo=timezone.utc) if val.tzinfo is None else val
    try:
        # numeric timestamp
        if isinstance(val, (int, float)):
            return datetime.fromtimestamp(val, tz=timezone.utc)
        # string
        return pd.to_datetime(val).to_pydatetime()
    except Exception:
        return None

def _item_id(it):
    # generate stable id from url or text
    url = (it.get("url") or "").strip()
    if url:
        return f"url:{url}"
    text = (it.get("text") or "")[:200]
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"textsha:{h}"

def build_mentions_df(reddit_items, quora_items, brands=None):
    """
    Convert raw mention lists into a normalized pandas DataFrame with columns:
    ['id','platform','title','text','url','score','comments','created_utc','subreddit','matched_terms']
    If brands is provided (list), it will be used only for ordering/validation; matching is taken from matched_terms or left empty.
    """
    items = list((reddit_items or []) + (quora_items or []))
    rows = []
    for it in items:
        created = _ensure_datetime(it.get("created_utc") or it.get("created") or it.get("timestamp"))
        rows.append({
            "id": _item_id(it),
            "platform": it.get("platform") or "unknown",
            "title": it.get("title") or "",
            "text": it.get("text") or "",
            "url": it.get("url") or "",
            "score": float(it.get("score") or 0),
            "comments": float(it.get("comments") or 0),
            "created_utc": created,
            "subreddit": it.get("subreddit") or it.get("community") or None,
            "author": it.get("author") or None,
            "author_followers": float(it.get("author_followers") or it.get("followers") or 0),
            "matched_terms": it.get("matched_terms") or [],
            # retain original raw object
            "_raw": it
        })
    df = pd.DataFrame(rows)
    if df.empty:
        # create columns even if empty
        cols = ["id","platform","title","text","url","score","comments","created_utc","subreddit","author","author_followers","matched_terms","_raw"]
        df = pd.DataFrame(columns=cols)
    # dedupe by id (prefer first)
    df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    # ensure created_utc is datetime
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
    # a normalized 'engagement' metric
    df["engagement"] = df["score"].fillna(0) + df["comments"].fillna(0)
    return df

def dedupe_by_url_and_text(df):
    """
    Deduplicate mentions preferring entries with url; otherwise dedupe by text hash.
    Returns a deduped dataframe.
    """
    if df.empty:
        return df
    # if url exists use it; else use id (already textsha)
    df["dedupe_key"] = df["url"].where(df["url"].astype(bool), df["id"])
    df = df.drop_duplicates(subset=["dedupe_key"], keep="first").drop(columns=["dedupe_key"])
    return df.reset_index(drop=True)

def mentions_volume(df, freq='D'):
    """
    Count mentions per brand per time window.
    - df: DataFrame
    - freq: pandas offset alias (D=day, H=hour)
    Returns DataFrame with index datetime and columns per brand (brand names from matched_terms)
    """
    if df.empty:
        return pd.DataFrame()
    # expand brands from matched_terms to rows
    exploded = df.explode("matched_terms")
    exploded["brand"] = exploded["matched_terms"].fillna("").replace("", np.nan)
    exploded["brand"] = exploded["brand"].fillna("unknown")
    exploded["ts"] = exploded["created_utc"].dt.floor(freq)
    counts = (exploded.groupby(["brand", "ts"]).size().reset_index(name="mentions"))
    pivot = counts.pivot(index="ts", columns="brand", values="mentions").fillna(0).sort_index()
    return pivot

def compute_sov(df, window=None):
    """
    Compute share of voice:
    - If window is None, compute overall SOV across all time.
    - Otherwise, pass window as timestamp or pandas time index to filter.
    Returns dict: per-brand mention counts and share_of_voice (inclusive and exclusive if both present).
    """
    res = {}
    if df.empty:
        return {"brand_mentions": {}, "share_of_voice": {}}
    dedup = dedupe_by_url_and_text(df)
    exploded = dedup.explode("matched_terms")
    exploded["brand"] = exploded["matched_terms"].fillna("").replace("", np.nan)
    exploded = exploded.dropna(subset=["brand"])
    brand_counts = exploded.groupby("brand").size().to_dict()
    total = sum(brand_counts.values())
    # inclusive SOV
    sov = {b: (count / total) for b, count in brand_counts.items()} if total > 0 else {b: None for b in brand_counts}
    res["brand_mentions"] = brand_counts
    res["share_of_voice"] = sov
    # exclusive handling: counts of brand-only (mention with only that brand present)
    # find mentions and how many distinct brands mentioned per id
    per_id = exploded.groupby("id")["brand"].nunique()
    # ids that correspond to single-brand mention
    single_brand_ids = per_id[per_id == 1].index.tolist()
    single_df = exploded[exploded["id"].isin(single_brand_ids)]
    excl_counts = single_df.groupby("brand").size().to_dict()
    excl_total = sum(excl_counts.values())
    res["brand_only_mentions"] = excl_counts
    res["share_of_voice_exclusive"] = {b: (c / excl_total) for b, c in excl_counts.items()} if excl_total > 0 else {b: None for b in excl_counts}
    return res

def sentiment_breakdown(df, label_col="label"):
    """
    Returns sentiment counts and percentages per brand and platform.
    Expects each row to have 'matched_terms' and a 'label' field in the underlying _raw or index directly.
    If 'label' is not present as a column, tries to pull it from _raw.
    """
    if df.empty:
        return {}
    # try label col
    if label_col not in df.columns:
        # try extracting from _raw
        def _get_label(row):
            raw = row.get("_raw", {}) or {}
            return raw.get("label") or raw.get("sentiment") or np.nan
        df[label_col] = df.apply(_get_label, axis=1)
    exploded = df.explode("matched_terms")
    exploded["brand"] = exploded["matched_terms"].fillna("").replace("", np.nan)
    exploded = exploded.dropna(subset=["brand"])
    counts = exploded.groupby(["brand", "platform", label_col]).size().unstack(fill_value=0)
    # compute percentages
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return {"counts": counts, "percentages": pct}

def net_sentiment_score(df, label_col="label"):
    """
    NSS per brand and platform: (positive - negative) / total
    Expects labels starting with 'pos'/'neg' or exact values 'positive'/'negative'
    """
    if df.empty:
        return {}
    # ensure label column
    if label_col not in df.columns:
        def _get_label(row):
            raw = row.get("_raw", {}) or {}
            return raw.get("label") or raw.get("sentiment") or np.nan
        df[label_col] = df.apply(_get_label, axis=1)
    exploded = df.explode("matched_terms")
    exploded["brand"] = exploded["matched_terms"].fillna("").replace("", np.nan)
    exploded = exploded.dropna(subset=["brand"])
    def _classify(lbl):
        if not isinstance(lbl, str):
            return "neutral"
        l = lbl.lower()
        if l.startswith("pos") or "positive" in l:
            return "positive"
        if l.startswith("neg") or "negative" in l:
            return "negative"
        return "neutral"
    exploded["__lbl"] = exploded[label_col].apply(_classify)
    grouped = exploded.groupby(["brand", "platform"])
    out = {}
    for (brand, platform), g in grouped:
        pos = (g["__lbl"] == "positive").sum()
        neg = (g["__lbl"] == "negative").sum()
        total = len(g)
        nss = (pos - neg) / total if total > 0 else None
        out.setdefault(brand, {})[platform] = {"positive": int(pos), "negative": int(neg), "total": int(total), "nss": nss}
    return out

def share_of_positive_sentiment(df, label_col="label"):
    """
    SOPS_brand = positive_brand / sum(positive_all_brands)
    Returns dict brand -> share (0..1)
    """
    if df.empty:
        return {}
    nss_data = net_sentiment_score(df, label_col=label_col)
    # nss_data structure: brand -> platform -> {positive,...}
    pos_per_brand = {}
    total_pos = 0
    for brand, platforms in nss_data.items():
        brand_pos = sum(p["positive"] for p in platforms.values())
        pos_per_brand[brand] = brand_pos
        total_pos += brand_pos
    if total_pos == 0:
        return {b: None for b in pos_per_brand}
    return {b: (c / total_pos) for b, c in pos_per_brand.items()}

def sentiment_trend(df, freq='D', label_col="label"):
    """
    Time-series of %positive and NSS per brand.
    Returns: dict of brand -> DataFrame indexed by timestamp with columns ['positive_pct','nss']
    """
    if df.empty:
        return {}
    if label_col not in df.columns:
        def _get_label(row):
            raw = row.get("_raw", {}) or {}
            return raw.get("label") or raw.get("sentiment") or np.nan
        df[label_col] = df.apply(_get_label, axis=1)
    exploded = df.explode("matched_terms")
    exploded["brand"] = exploded["matched_terms"].fillna("").replace("", np.nan)
    exploded = exploded.dropna(subset=["brand"]).copy()
    exploded["ts"] = exploded["created_utc"].dt.floor(freq)
    def _classify(lbl):
        if not isinstance(lbl, str):
            return "neutral"
        l = lbl.lower()
        if l.startswith("pos") or "positive" in l:
            return "positive"
        if l.startswith("neg") or "negative" in l:
            return "negative"
        return "neutral"
    exploded["__lbl"] = exploded[label_col].apply(_classify)
    out = {}
    for brand, g in exploded.groupby("brand"):
        agg = g.groupby("ts")["__lbl"].value_counts().unstack(fill_value=0)
        # ensure columns exist
        for c in ["positive", "negative", "neutral"]:
            if c not in agg.columns:
                agg[c] = 0
        agg["total"] = agg[["positive", "negative", "neutral"]].sum(axis=1)
        agg["positive_pct"] = agg["positive"] / agg["total"].replace(0, np.nan)
        agg["nss"] = (agg["positive"] - agg["negative"]) / agg["total"].replace(0, np.nan)
        out[brand] = agg.sort_index()
    return out

def engagement_metrics(df):
    """
    total_engagement and avg_engagement per brand.
    """
    if df.empty:
        return {}
    exploded = df.explode("matched_terms")
    exploded["brand"] = exploded["matched_terms"].fillna("").replace("", np.nan)
    exploded = exploded.dropna(subset=["brand"])
    grouped = exploded.groupby("brand")
    out = {}
    for brand, g in grouped:
        total_eng = g["engagement"].sum()
        avg_eng = g["engagement"].mean()
        out[brand] = {"total_engagement": float(total_eng), "avg_engagement": float(avg_eng), "mentions": int(len(g))}
    return out

def top_communities(df, top_n=10):
    """
    Top communities (subreddit/topic) with counts and sentiment breakdown.
    """
    if df.empty:
        return []
    exploded = df.explode("matched_terms")
    exploded["brand"] = exploded["matched_terms"].fillna("").replace("", np.nan)
    exploded = exploded.dropna(subset=["brand"])
    grouped = exploded.groupby("subreddit")
    rows = []
    for name, g in grouped:
        rows.append({
            "community": name,
            "mentions": int(len(g)),
            "positive": int(((g.get("label") or g.get("_raw").apply(lambda r: (r or {}).get("label"))) == "positive").sum()) if "label" in g.columns or "_raw" in g.columns else None
        })
    rows = sorted(rows, key=lambda r: r["mentions"] if r["mentions"] is not None else 0, reverse=True)[:top_n]
    return rows

def top_mentions(df, top_n=10):
    """
    Rank mentions by engagement (score + comments).
    Returns list of dicts with url, title, text, engagement, platform, sentiment (if available).
    """
    if df.empty:
        return []
    df_sorted = df.sort_values(by="engagement", ascending=False).head(top_n)
    out = []
    for _, row in df_sorted.iterrows():
        raw = row.get("_raw") or {}
        out.append({
            "url": row.get("url"),
            "title": row.get("title"),
            "text": (row.get("text") or "")[:400],
            "engagement": float(row.get("engagement") or 0),
            "platform": row.get("platform"),
            "sentiment": raw.get("label") or raw.get("sentiment")
        })
    return out

def crisis_spike_alerts(df, freq='D', multiplier=3.0, neg_multiplier=1.5):
    """
    Rule-based spike detection:
    - For each brand: if mentions_today > multiplier * average_daily_mentions AND %negative_today > neg_multiplier * baseline_negative_pct -> alert
    Returns list of alert dicts per brand/date.
    """
    if df.empty:
        return []
    dedup = dedupe_by_url_and_text(df)
    exploded = dedup.explode("matched_terms")
    exploded["brand"] = exploded["matched_terms"].fillna("").replace("", np.nan)
    exploded = exploded.dropna(subset=["brand"])
    exploded["ts"] = exploded["created_utc"].dt.floor(freq)
    alerts = []
    for brand, g in exploded.groupby("brand"):
        daily = g.groupby("ts").size().rename("mentions")
        baseline_mean = daily[:-1].mean() if len(daily) > 1 else daily.mean()
        if np.isnan(baseline_mean) or baseline_mean == 0:
            baseline_mean = 0
        # compute negative pct baseline
        # try to get label
        if "label" not in g.columns:
            g["label"] = g["_raw"].apply(lambda r: (r or {}).get("label"))
        def _neg(x):
            if not isinstance(x, str):
                return False
            return x.lower().startswith("neg") or "negative" in x.lower()
        daily_neg_pct = g.groupby("ts")["label"].apply(lambda s: s.apply(_neg).sum() / max(1, len(s)))
        for ts in daily.index:
            today_count = int(daily.loc[ts])
            baseline = baseline_mean
            if baseline == 0:
                # require absolute minimum to avoid noisy alerts
                if today_count < 5:
                    continue
                compare = today_count >= multiplier * 1
            else:
                compare = today_count > multiplier * baseline
            neg_pct_today = float(daily_neg_pct.get(ts, 0.0))
            neg_baseline = daily_neg_pct[:-1].mean() if len(daily_neg_pct) > 1 else daily_neg_pct.mean()
            neg_check = neg_pct_today > (neg_multiplier * (neg_baseline if (neg_baseline and not np.isnan(neg_baseline)) else 0))
            if compare and neg_check:
                alerts.append({"brand": brand, "ts": ts, "mentions": today_count, "neg_pct": neg_pct_today, "baseline_mentions": baseline, "baseline_neg_pct": float(neg_baseline or 0)})
    return alerts

def volume_vs_sentiment_ratio(df):
    """
    Compute negative_per_mention = negative / mentions per brand.
    Returns dict of brand -> ratio
    """
    if df.empty:
        return {}
    if "label" not in df.columns:
        df["label"] = df["_raw"].apply(lambda r: (r or {}).get("label"))
    exploded = df.explode("matched_terms")
    exploded["brand"] = exploded["matched_terms"].fillna("").replace("", np.nan)
    exploded = exploded.dropna(subset=["brand"])
    def _is_neg(l):
        if not isinstance(l, str):
            return False
        return l.lower().startswith("neg") or "negative" in l.lower()
    grouped = exploded.groupby("brand")
    out = {}
    for brand, g in grouped:
        mentions = len(g)
        negatives = g["label"].apply(_is_neg).sum()
        ratio = negatives / mentions if mentions > 0 else None
        out[brand] = {"mentions": mentions, "negatives": int(negatives), "negative_per_mention": ratio}
    return out

def influencer_impact(df):
    """
    Compute author_impact = author_followers * avg_engagement for each author.
    Returns top authors with their impact.
    """
    if df.empty:
        return []
    if "author" not in df.columns:
        return []
    g = df.groupby("author").agg({"author_followers": "max", "engagement": ["mean", "sum"], "id": "count"})
    g.columns = ["followers", "avg_engagement", "total_engagement", "mentions"]
    g = g.reset_index()
    g["impact"] = g["followers"] * g["avg_engagement"]
    g = g.sort_values("impact", ascending=False)
    out = []
    for _, row in g.head(20).iterrows():
        out.append({
            "author": row["author"],
            "followers": float(row["followers"]),
            "avg_engagement": float(row["avg_engagement"]),
            "mentions": int(row["mentions"]),
            "impact": float(row["impact"])
        })
    return out