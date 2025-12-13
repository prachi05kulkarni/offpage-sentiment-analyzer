# analytics_visuals.py
# Minimal visualization helpers using Altair and simple HTML for card-like sentiment.

from typing import List, Dict
import streamlit as st
import pandas as pd
import altair as alt
alt.themes.enable("opaque")

COLOR_PALETTE = ["#FF4500", "#B92B27", "#6366F1", "#06B6D4", "#F59E0B", "#10B981", "#EF4444"]

def render_share_of_voice(sov_data: List[Dict]):
    if not sov_data:
        st.info("No share-of-voice data to display.")
        return
    df = pd.DataFrame(sov_data).fillna(0)
    df["reddit"] = df["reddit"].astype(int)
    df["quora"] = df["quora"].astype(int)

    st.subheader("Share of Voice")
    st.caption("Mentions per brand and platform")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Reddit**")
        if df["reddit"].sum() == 0:
            st.warning("No Reddit mentions found.")
        else:
            chart = (
                alt.Chart(df)
                .mark_arc(innerRadius=60)
                .encode(theta=alt.Theta("reddit:Q"), color=alt.Color("brand:N", scale=alt.Scale(range=COLOR_PALETTE)), tooltip=["brand", "reddit"])
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)
    with col2:
        st.markdown("**Quora**")
        if df["quora"].sum() == 0:
            st.warning("No Quora mentions found.")
        else:
            chart = (
                alt.Chart(df)
                .mark_arc(innerRadius=60)
                .encode(theta=alt.Theta("quora:Q"), color=alt.Color("brand:N", scale=alt.Scale(range=COLOR_PALETTE)), tooltip=["brand", "quora"])
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)

# Small HTML stacked bar for sentiment percentages
def _stacked_progress_html(positive:int, neutral:int, negative:int) -> str:
    total = max(positive + neutral + negative, 1)
    p = round(positive/total*100)
    n = round(neutral/total*100)
    neg = 100 - p - n
    return f"""
    <div style="font-family:inherit">
      <div style="height:12px;border-radius:999px;overflow:hidden;display:flex;background:#f3f4f6;">
        <div style="width:{p}%;background:#10B981;"></div>
        <div style="width:{n}%;background:#F59E0B;"></div>
        <div style="width:{neg}%;background:#EF4444;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:12px;color:#374151;margin-top:6px;">
        <div><span style="display:inline-block;width:10px;height:10px;background:#10B981;margin-right:6px;"></span>{p}% Positive</div>
        <div><span style="display:inline-block;width:10px;height:10px;background:#F59E0B;margin-right:6px;"></span>{n}% Neutral</div>
        <div><span style="display:inline-block;width:10px;height:10px;background:#EF4444;margin-right:6px;"></span>{neg}% Negative</div>
      </div>
    </div>
    """

def render_sentiment_cards(sentiment_data: List[Dict]):
    if not sentiment_data:
        st.info("No sentiment data to display.")
        return
    st.subheader("Sentiment Overview — Cards")
    for item in sentiment_data:
        brand = item.get("brand", "Unknown")
        reddit = item.get("reddit", {"positive":0,"neutral":0,"negative":0})
        quora = item.get("quora", {"positive":0,"neutral":0,"negative":0})

        st.markdown(f"### {brand}")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Reddit**")
            st.markdown(_stacked_progress_html(int(reddit.get("positive",0)), int(reddit.get("neutral",0)), int(reddit.get("negative",0))), unsafe_allow_html=True)
        with cols[1]:
            st.markdown("**Quora**")
            st.markdown(_stacked_progress_html(int(quora.get("positive",0)), int(quora.get("neutral",0)), int(quora.get("negative",0))), unsafe_allow_html=True)