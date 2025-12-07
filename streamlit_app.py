import os
import logging
import traceback
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from reddit_client import fetch_reddit_mentions
from quora_client import fetch_quora_mentions
from sentiment import analyze_sentiments
from analysis import compute_share_of_voice, top_threads, generate_recommendations
from utils import read_processed_jsonl_if_exists

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("app")

st.set_page_config(page_title="Offpage Sentiment Analyzer", layout="wide")
st.title("Offpage Sentiment Analyzer — Reddit & Quora")

# Sidebar / inputs
with st.sidebar:
    st.header("Settings")
    brand = st.text_input("Brand name", value="", help="Brand to analyze (required)")
    competitor = st.text_input("Competitor (optional)", value="", help="Competitor for SOV")
    days = st.slider("Lookback days (historical search)", min_value=1, max_value=90, value=14)
    use_streamed = st.checkbox("Show processed streamed mentions (if running stream worker)", value=True)
    run_btn = st.button("Run analysis")

if run_btn:
    if not brand.strip():
        st.error("Please enter a brand name to proceed.")
    else:
        try:
            with st.spinner("Fetching historical Reddit mentions..."):
                reddit_items = fetch_reddit_mentions(brand.strip(), competitor.strip() or None, days=days)
            with st.spinner("Fetching Quora mentions..."):
                quora_items = fetch_quora_mentions(brand.strip(), competitor.strip() or None, days=days)

            all_items = {"reddit (historical)": reddit_items or [], "quora": quora_items or []}

            st.subheader("Raw mentions count (historical)")
            st.write({k: len(v) for k, v in all_items.items()})

            st.subheader("Sentiment analysis (historical)")
            for platform, items in all_items.items():
                texts = [it.get("text", "") for it in items if it.get("text")]
                if not texts:
                    st.info(f"No {platform} mentions found")
                    continue
                sentiments = analyze_sentiments(texts)
                pos = sum(1 for s in sentiments if s.get("label", "").lower().startswith("pos"))
                neg = sum(1 for s in sentiments if s.get("label", "").lower().startswith("neg"))
                neu = len(sentiments) - pos - neg
                st.write(f"{platform}: +{pos} / -{neg} / neutral:{neu}")
                st.bar_chart({"positive": pos, "negative": neg, "neutral": neu})

            st.subheader("Share of Voice (historical)")
            sov = compute_share_of_voice(reddit_items, quora_items, brand.strip(), competitor.strip() or None)
            st.json(sov)

            st.subheader("Top threads (historical)")
            for platform, items in all_items.items():
                st.write(platform.upper())
                for item in top_threads(items, top_n=5):
                    title = item.get("title") or (item.get("text") or "")[:80]
                    url = item.get("url", "")
                    comments = item.get("comments", 0)
                    score = item.get("score", 0)
                    if url:
                        st.markdown(f"- [{title}]({url}) — score:{score} comments:{comments}")
                    else:
                        st.markdown(f"- {title} — score:{score} comments:{comments}")

            st.subheader("Recommendations (historical)")
            recs = generate_recommendations(reddit_items + quora_items, brand.strip())
            for r in recs:
                st.markdown(f"- {r}")

            # Optionally show streamed processed mentions (near-real-time) if available
            if use_streamed:
                processed = read_processed_jsonl_if_exists()
                if processed:
                    st.subheader("Processed streamed mentions (recent)")
                    st.write(f"Showing up to latest {min(200, len(processed))} items")
                    st.dataframe(processed[:200])
                else:
                    st.info("No processed streamed mentions found (run the stream worker and processor).")

        except Exception as e:
            logger.error("Unhandled exception in main flow: %s", traceback.format_exc())
            st.error(f"Analysis failed: {e}. Check logs for details.")