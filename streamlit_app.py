# streamlit_app.py

from typing import List, Dict
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from data_connectors import (
    fetch_reddit_mentions_for_brand,
    fetch_quora_mentions_for_brand,
    find_sentiment_fn,
)
from analytics_visuals import render_share_of_voice, render_sentiment_cards

st.set_page_config(page_title="Offpage Sentiment Analyzer", layout="wide")
st.title("Offpage Sentiment Analyzer")

# Sidebar diagnostics and controls
if "logs" not in st.session_state:
    st.session_state["logs"] = []

def log(msg: str):
    st.session_state["logs"].append(msg)
    st.session_state["logs"] = st.session_state["logs"][-300:]

with st.sidebar:
    st.header("Diagnostics")
    if st.button("Clear logs"):
        st.session_state["logs"] = []
    for line in st.session_state["logs"][-200:]:
        st.text(line)
    st.markdown("---")
    st.markdown("Notes:")
    st.markdown("- Make sure reddit_client.py (and optional quora_client.py / sentiment.py) are next to this file.")
    st.markdown("- If you want better sentiment, add a sentiment.py with a `predict(text|list)` function.")

# Form input
with st.form("analyze"):
    brand = st.text_input("Your Brand (required)", placeholder="e.g., Pond's")
    competitors = st.text_input("Competitors (comma-separated, optional)", placeholder="Nivea, Garnier")
    limit = st.number_input("Max mentions per brand/platform", value=200, min_value=10, max_value=1000)
    submitted = st.form_submit_button("Analyze")

if not submitted:
    st.info("Enter brand + (optional) competitors and press Analyze.")
else:
    if not brand.strip():
        st.warning("Please enter a brand.")
    else:
        brands = [brand.strip()] + [c.strip() for c in competitors.split(",") if c.strip()]
        st.write("Analyzing:", ", ".join(brands))
        
        # Check for mock warning
        if st.session_state.get("mock_warning_shown"):
             st.warning("⚠️ **DEMO MODE**: Reddit credentials are missing and public fetch is blocked. Displaying **MOCK DATA** for Reddit. Quora data may still be real if key is valid.")
        # try to find sentiment function (if provided)
        sentiment_fn = find_sentiment_fn()
        if sentiment_fn:
            log(f"Using sentiment function: {getattr(sentiment_fn, '__name__', str(sentiment_fn))}")
        else:
            log("No sentiment module found — using simple lexicon fallback (built-in).")

        # fetch mentions and compute sentiment
        sov_results: List[Dict] = []
        sentiment_results: List[Dict] = []

        # Keep sample mentions for debugging
        sample_mentions = {}

        for b in brands:
            st.write(f"Fetching for {b} ...")
            # Ensure brand passed as str
            b_str = str(b).strip()

            # Fetch per-brand (fresh lists)
            reddit_mentions = fetch_reddit_mentions_for_brand(b_str, limit)
            quora_mentions = fetch_quora_mentions_for_brand(b_str, limit)

            # Defensive: coerce to lists
            if not isinstance(reddit_mentions, list):
                log(f"[WARN] reddit fetch for '{b_str}' returned non-list; coerced to empty.")
                reddit_mentions = []
            if not isinstance(quora_mentions, list):
                log(f"[WARN] quora fetch for '{b_str}' returned non-list; coerced to empty.")
                quora_mentions = []

            log(f"{b_str}: reddit={len(reddit_mentions)}, quora={len(quora_mentions)}")

            # store sample for debugging
            sample_mentions[b_str] = {
                "reddit_sample": reddit_mentions[:3],
                "quora_sample": quora_mentions[:3],
            }

            # build text list for sentiment prediction (prefer sentiment_fn)
            texts: List[str] = []
            text_meta: List[tuple] = []  # (platform, raw_mention)
            for m in reddit_mentions:
                texts.append((m.get("text") or m.get("title") or "").strip())
                text_meta.append(("reddit", m))
            for m in quora_mentions:
                texts.append((m.get("text") or m.get("title") or "").strip())
                text_meta.append(("quora", m))

            # compute labels
            labels: List[str] = []
            if sentiment_fn and texts:
                try:
                    # attempt batch call first
                    res = sentiment_fn(texts) if callable(sentiment_fn) else None
                    if isinstance(res, list) and len(res) == len(texts):
                        labels = [str(x).lower() for x in res]
                    else:
                        # fallback to per-text calls
                        labels = [str(sentiment_fn(t)).lower() for t in texts]
                except Exception as e:
                    log(f"Sentiment function error: {e}")
                    labels = []

            if not labels and texts:
                # lexicon fallback (simple)
                POS = {
                    "good",
                    "great",
                    "love",
                    "excellent",
                    "best",
                    "amazing",
                    "happy",
                    "like",
                    "awesome",
                    "positive",
                    "win",
                    "improve",
                    "liked",
                    "recommend",
                }
                NEG = {
                    "bad",
                    "terrible",
                    "hate",
                    "awful",
                    "worst",
                    "angry",
                    "disappointed",
                    "poor",
                    "negative",
                    "problem",
                    "issue",
                    "complain",
                    "complaint",
                    "risk",
                    "scam",
                }
                for t in texts:
                    lower = (t or "").lower()
                    p = sum(1 for w in POS if w in lower)
                    n = sum(1 for w in NEG if w in lower)
                    if p > n:
                        labels.append("positive")
                    elif n > p:
                        labels.append("negative")
                    else:
                        labels.append("neutral")

            # aggregate counts
            reddit_counts = {"positive": 0, "neutral": 0, "negative": 0}
            quora_counts = {"positive": 0, "neutral": 0, "negative": 0}
            for (plat, _m), lbl in zip(text_meta, labels):
                l = lbl.lower() if isinstance(lbl, str) else "neutral"
                if "pos" in l:
                    lab = "positive"
                elif "neg" in l:
                    lab = "negative"
                else:
                    lab = "neutral"
                if plat == "reddit":
                    reddit_counts[lab] += 1
                else:
                    quora_counts[lab] += 1

            sov_results.append({"brand": b_str, "reddit": len(reddit_mentions), "quora": len(quora_mentions)})
            sentiment_results.append({"brand": b_str, "reddit": reddit_counts, "quora": quora_counts})

        # --- DEBUG: show raw SOV structure to sidebar and validate ---
        st.sidebar.markdown("### SOV raw data (debug)")
        st.sidebar.json(sov_results)

        # Convert to DataFrame and coerce numeric types before plotting
        df_sov = pd.DataFrame(sov_results).fillna(0)
        # Defensive column checks
        if not {"brand", "reddit", "quora"}.issubset(set(df_sov.columns)):
            st.error("SOV data missing expected columns (brand, reddit, quora). See sidebar debug JSON.")
            log("SOV data missing expected columns; aborting render.")
        else:
            # ensure numeric
            df_sov["reddit"] = pd.to_numeric(df_sov["reddit"], errors="coerce").fillna(0).astype(int)
            df_sov["quora"] = pd.to_numeric(df_sov["quora"], errors="coerce").fillna(0).astype(int)

            # Show DataFrame in-app for quick inspection (helps pinpoint identical distributions)
            st.write("SOV DataFrame (debug):")
            st.dataframe(df_sov)

            # Detect identical distributions (every brand has same counts) and warn
            reddit_unique = df_sov["reddit"].nunique()
            quora_unique = df_sov["quora"].nunique()
            if reddit_unique == 1 and quora_unique == 1:
                st.warning(
                    "All brands currently have identical Reddit & Quora counts. "
                    "This usually means the fetch function returned identical results for each brand or the query wasn't applied per-brand.\n"
                    "Check the sidebar 'SOV raw data' and the sample mentions below."
                )
                # show sample mentions per brand to help debug
                st.sidebar.markdown("### Sample mentions (first 3 per brand)")
                for br, samp in sample_mentions.items():
                    st.sidebar.markdown(f"**{br}**")
                    st.sidebar.markdown("Reddit sample:")
                    st.sidebar.json(samp["reddit_sample"])
                    st.sidebar.markdown("Quora sample:")
                    st.sidebar.json(samp["quora_sample"])

            # show visuals using cleaned sov_results
            cleaned_sov = df_sov.to_dict(orient="records")
            render_share_of_voice(cleaned_sov)

            st.markdown("---")
            render_sentiment_cards(sentiment_results)