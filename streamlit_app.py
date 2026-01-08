"""
Final Streamlit dashboard with:
- premium navbar
- robust fetching, SOV, sentiment
- per-brand example captures with confidence scores
- "Explain why labeled X" view (keyword highlights + simple importance)
- defensive use of utils.normalize_text (no detect_industry calls)

Drop this file into your repo (replace existing streamlit_app.py) and run:
  pip install -r requirements.txt
  pip install emoji contractions
  streamlit run streamlit_app.py
"""
from typing import List, Dict, Tuple, Optional
import streamlit as st
import pandas as pd
import time
import json
import tempfile
import os
import re
from datetime import datetime, date

# load .env if present
from dotenv import load_dotenv
load_dotenv()

# repo modules
from data_connectors import (
    fetch_reddit_mentions_for_brand,
    fetch_quora_mentions_for_brand,
    find_sentiment_fn,
)
from analytics_visuals import render_share_of_voice, render_sentiment_cards
import top_threads

# sentiment + utils
import sentiment
import utils  # keep import (we may use normalize_text if present), but we will NOT call detect_industry

# Page setup
st.set_page_config(page_title="Offpage Sentiment Analyzer", layout="wide", initial_sidebar_state="collapsed")

# --- Premium styles & navbar ---
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%); }
      .top-nav {
        background: linear-gradient(90deg, #0f172a 0%, #0ea5a4 100%);
        color: white; padding: 18px 32px; border-radius: 12px;
        box-shadow: 0 6px 30px rgba(2,6,23,0.10); margin-bottom: 18px;
      }
      .nav-inner { display:flex; align-items:center; justify-content:space-between; gap:20px; }
      .brand-title { font-size:22px; font-weight:700; color: #ffffff; margin:0; }
      .brand-sub { font-size:12px; color: rgba(255,255,255,0.9); margin:0; }
      .nav-links { display:flex; gap:18px; align-items:center; }
      .nav-link { color: rgba(255,255,255,0.9); text-decoration:none; padding:8px 12px; border-radius:8px; font-weight:600; }
      .nav-link:hover { background: rgba(255,255,255,0.06); color: #fff; }
      .nav-active { background: rgba(255,255,255,0.12); box-shadow: inset 0 -3px 0 rgba(255,255,255,0.06); }
      .kpi-pill { background: rgba(255,255,255,0.08); color: #fff; padding:8px 12px; border-radius:999px; font-weight:600; font-size:13px; }
      .card { background: #ffffff; border-radius:12px; padding:18px; box-shadow: 0 6px 20px rgba(15,23,42,0.06); }
      .muted { color:#64748b; font-size:13px; }
      .small-muted { color:#6b7280; font-size:12px; }
      mark { background: #fff3b0; padding:0 4px; border-radius:3px; }
      @media (max-width: 800px) { .nav-inner { flex-direction:column; align-items:flex-start; gap:12px; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top navbar
if "page" not in st.session_state:
    st.session_state["page"] = "Overview"

col_top_left, col_top_right = st.columns([3, 1])
with col_top_left:
    st.markdown(
        f"""
        <div class="top-nav">
          <div class="nav-inner">
            <div style="display:flex;align-items:center;gap:14px">
              <div style="width:44px;height:44px;border-radius:8px;background:rgba(255,255,255,0.12);display:flex;align-items:center;justify-content:center;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="white"/></svg>
              </div>
              <div>
                <div class="brand-title">Offpage Sentiment Analyzer</div>
                <div class="brand-sub">Share-of-Voice & Sentiment Monitoring — Reddit & Quora</div>
              </div>
            </div>
            <div style="display:flex;align-items:center;gap:12px">
              <div class="nav-links">
                <span style="padding:8px 12px;border-radius:8px;" class="nav-active">Overview</span>
                <span style="padding:8px 12px;border-radius:8px;">Threads</span>
                <span style="padding:8px 12px;border-radius:8px;">Settings</span>
                <span style="padding:8px 12px;border-radius:8px;">About</span>
              </div>
              <div class="kpi-pill">Live</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_top_right:
    st.write("")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Query & Settings")
    st.markdown("Pick date range, change model or controls below.")
    today = date.today()
    default_start = today.replace(day=max(1, today.day - 13))
    dr = st.date_input("Date range", value=(default_start, today))
    if isinstance(dr, tuple) and len(dr) == 2:
        start_date, end_date = dr[0], dr[1]
    else:
        start_date, end_date = dr, dr

    st.markdown("---")
    st.subheader("Model / Sentiment settings")
    hf_model_env = os.getenv("HF_SENTIMENT_MODEL", "textattack/bert-base-uncased-SST-2")
    hf_model = st.text_input("HuggingFace model (used if installed)", value=hf_model_env)
    st.caption("Override HF model via HF_SENTIMENT_MODEL in .env")
    st.markdown("---")
    st.subheader("Developer")
    st.checkbox("Show raw data in sidebar", value=False, key="show_raw")
    st.write("API keys (in .env) are used if present. Missing keys -> public fallbacks used.")

# Analysis form
with st.container():
    with st.card() if hasattr(st, "card") else st.container():
        with st.form("analyze_form"):
            cols = st.columns([3, 2])
            with cols[0]:
                brand = st.text_input("Primary brand", placeholder="e.g., Pond's", value=os.getenv("STREAM_BRAND", ""))
                competitors = st.text_input("Competitors (comma-separated)", placeholder="Nivea, Garnier")
            with cols[1]:
                max_per_source = st.number_input("Max mentions per source", min_value=10, max_value=1000, value=200, step=10)
                submitted = st.form_submit_button("Analyze")

if not submitted:
    st.info("Enter brand and (optional) competitors, choose date range, then click Analyze.")
    st.stop()

if not brand or not brand.strip():
    st.error("Please enter a primary brand to analyze.")
    st.stop()

brands = [brand.strip()] + [c.strip() for c in competitors.split(",") if c.strip()]
st.markdown(f"### Analysis for: **{', '.join(brands)}**  —  Date range: **{start_date} → {end_date}**")

# helper: date filter
def mention_in_range(m: Dict, start_dt: date, end_dt: date) -> bool:
    ts_candidates = []
    for k in ("processed_at", "processedAt", "created_utc", "createdAt", "timestamp"):
        if k in m and m.get(k):
            ts_candidates.append(m.get(k))
    if not ts_candidates:
        return True
    for v in ts_candidates:
        try:
            if isinstance(v, (int, float)):
                dt = datetime.utcfromtimestamp(float(v)).date()
            else:
                dt = pd.to_datetime(str(v), utc=True, errors="coerce").date()
            if dt is None:
                continue
            if start_dt <= dt <= end_dt:
                return True
        except Exception:
            continue
    return False

# --- explainability lexicons & helpers (simple, transparent approach) ---
POS_LEXICON = {"good","great","love","excellent","best","amazing","happy","like","awesome","positive","win","improve","recommend","helpful","safe"}
NEG_LEXICON = {"bad","terrible","hate","awful","worst","angry","disappointed","poor","negative","problem","issue","complain","complaint","scam","risk","fail","broken","delay"}

_WORD_RE = re.compile(r"\w+")
def extract_keyword_matches(text: str) -> Dict[str, List[str]]:
    """Return matched positive and negative words (lowercased) present in text."""
    tokens = [t.lower() for t in _WORD_RE.findall(text or "")]
    pos = sorted({t for t in tokens if t in POS_LEXICON})
    neg = sorted({t for t in tokens if t in NEG_LEXICON})
    return {"pos": pos, "neg": neg, "tokens": tokens}

def highlight_text(text: str, words: List[str]) -> str:
    """Return HTML with matched words wrapped in <mark> for visual emphasis."""
    if not words:
        return st.markdown(text)
    # simple replacement that respects word boundaries (case-insensitive)
    out = text
    for w in sorted(words, key=len, reverse=True):
        pattern = re.compile(r"(?i)\b(" + re.escape(w) + r")\b")
        out = pattern.sub(r"<mark>\1</mark>", out)
    return out

# Containers for results and examples
all_raw = []
sov_results = []
sentiment_results = []
sample_mentions = {}
brand_examples: Dict[str, Dict[str, List[Dict]]] = {}  # brand -> {positive: [examples], neutral: [], negative: []}

sentiment_fn = find_sentiment_fn()
use_builtin_sentiment = sentiment_fn is None

# Main fetch + sentiment loop
for b in brands:
    st.markdown(f"#### Fetching mentions for: {b}")
    reddit_mentions = []
    quora_mentions = []
    try:
        reddit_mentions = fetch_reddit_mentions_for_brand(b, int(max_per_source))
    except Exception as e:
        st.sidebar.warning(f"Reddit fetch error for {b}: {e}")
    try:
        quora_mentions = fetch_quora_mentions_for_brand(b, int(max_per_source))
    except Exception as e:
        st.sidebar.warning(f"Quora fetch error for {b}: {e}")

    if not isinstance(reddit_mentions, list):
        reddit_mentions = []
    if not isinstance(quora_mentions, list):
        quora_mentions = []

    reddit_filtered = [m for m in reddit_mentions if mention_in_range(m, start_date, end_date)]
    quora_filtered = [m for m in quora_mentions if mention_in_range(m, start_date, end_date)]

    sample_mentions[b] = {"reddit_sample": reddit_filtered[:3], "quora_sample": quora_filtered[:3]}
    sov_results.append({"brand": b, "reddit": len(reddit_filtered), "quora": len(quora_filtered)})

    for m in reddit_filtered:
        m.setdefault("platform", "reddit")
        m["brand"] = b
        all_raw.append(m)
    for m in quora_filtered:
        m.setdefault("platform", "quora")
        m["brand"] = b
        all_raw.append(m)

    # Sentiment inference: always produce a list of dicts {"label":..., "score":...}
    texts = [(m.get("text") or m.get("title") or "")[:600].strip() for m in reddit_filtered + quora_filtered]
    results: List[Dict] = []
    if texts:
        try:
            pre_texts = [utils.normalize_text(t) for t in texts]
        except Exception:
            pre_texts = [t for t in texts]

        # First, try user-supplied sentiment_fn if present
        if sentiment_fn:
            try:
                res = sentiment_fn(pre_texts) if callable(sentiment_fn) else None
                if isinstance(res, list):
                    # normalize into dict form
                    for r in res:
                        if isinstance(r, dict):
                            label = str(r.get("label", "NEUTRAL")).upper()
                            score = float(r.get("score", 1.0)) if r.get("score") is not None else 1.0
                            results.append({"label": label, "score": score})
                        else:
                            # assume string label
                            results.append({"label": str(r).upper(), "score": 1.0})
            except Exception:
                results = []

        # If user fn didn't produce results, use our sentiment.analyze_sentiments (HF/VADER/heuristics)
        if not results:
            try:
                out = sentiment.analyze_sentiments(pre_texts, batch_size=32)
                # normalize to label/score entries
                for o in out:
                    lab = (o.get("label") or "NEUTRAL").upper()
                    score = float(o.get("score", 0.0)) if o.get("score") is not None else 0.0
                    results.append({"label": lab, "score": score})
            except Exception:
                # fallback lexicon if everything fails
                results = []
                POS = POS_LEXICON
                NEG = NEG_LEXICON
                for t in pre_texts:
                    lower = (t or "").lower()
                    p = sum(1 for w in POS if w in lower)
                    n = sum(1 for w in NEG if w in lower)
                    if p > n:
                        results.append({"label": "POSITIVE", "score": float(min(1.0, p/(p+n if p+n>0 else 1)))})
                    elif n > p:
                        results.append({"label": "NEGATIVE", "score": float(min(1.0, n/(p+n if p+n>0 else 1)))})
                    else:
                        results.append({"label": "NEUTRAL", "score": 0.0})

    # pad if mismatch
    if len(results) < len(texts):
        # fill with neutral entries
        for _ in range(len(texts) - len(results)):
            results.append({"label": "NEUTRAL", "score": 0.0})

    # aggregate counts and capture examples + confidence
    reddit_counts = {"positive": 0, "neutral": 0, "negative": 0}
    quora_counts = {"positive": 0, "neutral": 0, "negative": 0}
    brand_examples[b] = {"positive": [], "neutral": [], "negative": []}

    meta = [("reddit", m) for m in reddit_filtered] + [("quora", m) for m in quora_filtered]
    for (plat, m_obj), res, raw_text in zip(meta, results, texts):
        lab_raw = (res.get("label") or "NEUTRAL").upper()
        score = float(res.get("score", 0.0))
        if lab_raw.startswith("POS"):
            lab = "positive"
        elif lab_raw.startswith("NEG"):
            lab = "negative"
        else:
            lab = "neutral"

        if plat == "reddit":
            reddit_counts[lab] = reddit_counts.get(lab, 0) + 1
        elif plat == "quora":
            quora_counts[lab] = quora_counts.get(lab, 0) + 1
        else:
            reddit_counts["neutral"] = reddit_counts.get("neutral", 0) + 1

        # store example (up to 3 per sentiment)
        try:
            if len(brand_examples[b][lab]) < 3:
                kw = extract_keyword_matches(raw_text)
                example = {
                    "platform": plat,
                    "text": raw_text,
                    "url": m_obj.get("url") or "",
                    "confidence": round(score, 3),
                    "matches": kw,
                }
                brand_examples[b][lab].append(example)
        except Exception:
            pass

    sentiment_results.append({"brand": b, "reddit": reddit_counts, "quora": quora_counts})

# Optional debug
if st.sidebar.checkbox("Show sample mentions", value=False):
    st.sidebar.json(sample_mentions)

# Build df_sov and compute percentages
df_sov = pd.DataFrame(sov_results).fillna(0)
if df_sov.empty:
    st.warning("No mentions found for selected range/brands. Try expanding the range or adding fewer filters.")
else:
    df_sov["reddit"] = pd.to_numeric(df_sov["reddit"], errors="coerce").fillna(0).astype(int)
    df_sov["quora"] = pd.to_numeric(df_sov["quora"], errors="coerce").fillna(0).astype(int)
    df_sov["total"] = df_sov["reddit"] + df_sov["quora"]
    grand_total = int(df_sov["total"].sum()) if not df_sov["total"].empty else 0
    df_sov["sov_pct"] = (df_sov["total"] / max(grand_total, 1) * 100.0).round(1)

    # pad sentiment counts to match totals
    for sr in sentiment_results:
        bname = sr.get("brand")
        matching = df_sov[df_sov["brand"] == bname]
        desired_total = int(matching["total"].iloc[0]) if not matching.empty else 0
        current_total = 0
        for plat in ("reddit", "quora"):
            p_counts = sr.get(plat) or {}
            current_total += int(p_counts.get("positive", 0) + p_counts.get("neutral", 0) + p_counts.get("negative", 0))
        diff = desired_total - current_total
        if diff > 0:
            if "reddit" in sr and isinstance(sr["reddit"], dict):
                sr["reddit"]["neutral"] = sr["reddit"].get("neutral", 0) + diff
            else:
                sr["reddit"] = sr.get("reddit", {"positive": 0, "neutral": diff, "negative": 0})

    # KPI row
    top_row = df_sov.sort_values("sov_pct", ascending=False).iloc[0] if not df_sov.empty else None
    if top_row is not None:
        k1, k2, k3, k4 = st.columns([2, 2, 2, 4])
        with k1:
            st.metric(f"Top SOV — {top_row['brand']}", f"{top_row['sov_pct']:.1f}%")
        with k2:
            st.metric("Total mentions (all brands)", f"{int(grand_total)}")
        with k3:
            neg_total = 0
            tot_mentions = 0
            for sr in sentiment_results:
                r = sr.get("reddit", {}); q = sr.get("quora", {})
                neg_total += int(r.get("negative", 0)) + int(q.get("negative", 0))
                tot_mentions += int(r.get("positive", 0)) + int(r.get("neutral", 0)) + int(r.get("negative", 0)) + int(q.get("positive", 0)) + int(q.get("neutral", 0)) + int(q.get("negative", 0))
            neg_pct = round(neg_total / max(tot_mentions, 1) * 100, 1)
            st.metric("Overall Negative %", f"{neg_pct}%")
        with k4:
            st.write("")

    st.markdown("---")
    left, right = st.columns([2, 1])
    with left:
        render_share_of_voice(df_sov.to_dict(orient="records"))
    with right:
        render_sentiment_cards(sentiment_results)

    # Sentiment definitions & examples with explanations
    st.markdown("---")
    st.subheader("How sentiment is determined — definitions, examples & explanations")

    st.markdown("""
    - Positive: model/pipeline labeled text as positive (high confidence). Typical cues — praise, recommendations, positive experiences.
    - Negative: model/pipeline labeled text as negative (high confidence). Typical cues — complaints, reports of problems, criticism.
    - Neutral: model/pipeline had low confidence or the text is informational / question / mixed sentiment.
    """)
    st.markdown("Notes: Texts are preprocessed before inference. We show confidence scores (0..1). Use the Explain view to see which keywords matched and highlighted in text.")

    # Show per-brand examples with confidence and explainers
    for b in brands:
        with st.expander(f"Examples for {b} (positive / neutral / negative)"):
            be = brand_examples.get(b, {"positive": [], "neutral": [], "negative": []})
            totals = {"positive": 0, "neutral": 0, "negative": 0}
            sr = next((s for s in sentiment_results if s.get("brand") == b), None)
            if sr:
                for plat in ("reddit", "quora"):
                    p = sr.get(plat, {})
                    totals["positive"] += int(p.get("positive", 0))
                    totals["neutral"] += int(p.get("neutral", 0))
                    totals["negative"] += int(p.get("negative", 0))
            st.markdown(f"**Counts:** Positive: {totals['positive']} • Neutral: {totals['neutral']} • Negative: {totals['negative']}")
            colp, coln, colneg = st.columns(3)
            def render_examples(col, ex_list, title):
                with col:
                    st.markdown(f"**{title} examples**")
                    if not ex_list:
                        st.write("No examples captured")
                        return
                    for idx, ex in enumerate(ex_list):
                        txt = ex.get("text", "")[:600]
                        plat = ex.get("platform", "")
                        url = ex.get("url", "")
                        conf = ex.get("confidence", None)
                        matches = ex.get("matches", {})
                        # display summary line with confidence
                        label_line = f"- ({plat}) "
                        if url:
                            label_line += f"[link]({url})  "
                        label_line += (txt[:220] + ("..." if len(txt) > 220 else ""))
                        if conf is not None:
                            label_line += f"  •  confidence: **{conf:.3f}**"
                        st.markdown(label_line)
                        # Explain expander for each example
                        with st.expander("Explain this example", expanded=False):
                            st.markdown("**Matched keywords**")
                            pos_matched = matches.get("pos", []) if isinstance(matches, dict) else []
                            neg_matched = matches.get("neg", []) if isinstance(matches, dict) else []
                            st.markdown(f"- Positive keywords: `{', '.join(pos_matched)}`" if pos_matched else "- Positive keywords: none")
                            st.markdown(f"- Negative keywords: `{', '.join(neg_matched)}`" if neg_matched else "- Negative keywords: none")
                            # Show highlighted text
                            highlighted = txt
                            highlight_words = (pos_matched + neg_matched)
                            if highlight_words:
                                # render highlighted HTML
                                html = highlight_text(txt, highlight_words)
                                st.markdown(html, unsafe_allow_html=True)
                            else:
                                st.write(txt)
                            # Quick reasoning: simple rule-based importance
                            reasons = []
                            if pos_matched:
                                reasons.append("Contains positive keywords: " + ", ".join(pos_matched))
                            if neg_matched:
                                reasons.append("Contains negative keywords: " + ", ".join(neg_matched))
                            if conf is not None and conf < 0.55:
                                reasons.append("Low confidence — likely neutral/mixed content")
                            if not reasons:
                                reasons = ["No strong keywords matched; model used contextual cues or neutral fallback."]
                            st.markdown("**Quick reasoning:**")
                            for r in reasons:
                                st.markdown(f"- {r}")
            render_examples(colp, be["positive"], "Positive")
            render_examples(coln, be["neutral"], "Neutral")
            render_examples(colneg, be["negative"], "Negative")

    # Top Threads
    st.markdown("---")
    st.subheader("Top Threads (brand/competitor filtered)")
    if not all_raw:
        st.info("No mentions available to compute top threads.")
    else:
        now_ts = int(time.time())
        for m in all_raw:
            if not m.get("sentiment_label"):
                m["sentiment_label"] = m.get("sentiment_label", "NEUTRAL")
            if not m.get("processed_at"):
                try:
                    m["processed_at"] = int(float(m.get("created_utc") or now_ts))
                except Exception:
                    m["processed_at"] = now_ts

        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
        try:
            with open(tmpf.name, "w", encoding="utf-8") as fh:
                for m in all_raw:
                    fh.write(json.dumps(m) + "\n")
            try:
                top_list = top_threads.get_top_threads_by_entity(tmpf.name, entity=None, top_n=200, limit=2000)
            except Exception as e:
                st.error(f"top_threads.get_top_threads_by_entity failed: {e}")
                top_list = []
        finally:
            try:
                os.unlink(tmpf.name)
            except Exception:
                pass

        if not top_list:
            st.info("No top threads found.")
        else:
            def thread_mentions_include_entities(thread: Dict, entities: List[str]) -> bool:
                mentions = thread.get("mentions") or []
                ents = [e.lower() for e in entities if e]
                for m in mentions:
                    matched = m.get("matched_terms") or []
                    if isinstance(matched, (list, tuple)) and any(ent in (",".join(matched)).lower() for ent in ents):
                        return True
                    txt = " ".join([str(m.get(k, "") or "") for k in ("text","title","url")]).lower()
                    if any(ent in txt for ent in ents):
                        return True
                for key in ("display_title","title","thread_title","url"):
                    val = str(thread.get(key,"") or "").lower()
                    if any(ent in val for ent in ents):
                        return True
                return False

            filtered = [t for t in top_list if thread_mentions_include_entities(t, brands)]
            if not filtered:
                st.info("No top threads mention the brand or competitors. Try increasing limits.")
            else:
                rows = []
                for t in filtered:
                    rows.append({
                        "display_title": t.get("display_title") or t.get("title") or t.get("thread_title") or t.get("url") or "n/a",
                        "mentions": t.get("count") or len(t.get("mentions") or []),
                        "dominant_sentiment": t.get("dominant_sentiment") or t.get("sentiment") or "",
                        "engagement": t.get("engagement") or t.get("score") or 0,
                        "url": t.get("url") or "",
                    })
                top_df = pd.DataFrame(rows).sort_values("engagement", ascending=False).reset_index(drop=True)
                st.dataframe(top_df.head(50))
                st.markdown("**Clickable Top Threads**")
                for _, r in top_df.head(50).iterrows():
                    title = r["display_title"] or "Untitled"
                    url = r["url"] or ""
                    mentions_count = int(r["mentions"] or 0)
                    sentiment = r["dominant_sentiment"] or ""
                    engagement = r["engagement"] or 0
                    if url:
                        st.markdown(f"- <a class='thread-link' href='{url}' target='_blank'>{title}</a>  <span class='small-muted'>| mentions: <b>{mentions_count}</b> | sentiment: {sentiment} | engagement: {engagement}</span>", unsafe_allow_html=True)
                    else:
                        st.write(f"- {title} — mentions: {mentions_count} — sentiment: {sentiment} — engagement: {engagement}")

    st.success("Analysis complete.")