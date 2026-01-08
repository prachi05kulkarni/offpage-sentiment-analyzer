# analytics_visuals.py
# Enhanced visuals: donuts retained, percent labels, polished bar and cards and industry comparison.

from typing import List, Dict
import streamlit as st
import pandas as pd
import altair as alt
alt.themes.enable("opaque")

COLOR_PALETTE = ["#0ea5a4", "#06b6d4", "#6366f1", "#f59e0b", "#10b981", "#ef4444", "#ff4500"]


def render_share_of_voice(sov_data: List[Dict]):
    """
    Input: list of dicts with brand, reddit, quora and optionally total,sov_pct
    Renders:
      - Reddit & Quora donut charts (keeps the pie/donut requirement)
      - Horizontal SOV % bar with numeric labels (clear percent KPI)
      - Summary table
    """
    if not sov_data:
        st.info("No share-of-voice data to display.")
        return

    df = pd.DataFrame(sov_data).fillna(0)
    df["reddit"] = pd.to_numeric(df.get("reddit", 0), errors="coerce").fillna(0).astype(int)
    df["quora"] = pd.to_numeric(df.get("quora", 0), errors="coerce").fillna(0).astype(int)
    df["total"] = df.get("total", df["reddit"] + df["quora"])
    df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(df["reddit"] + df["quora"]).astype(int)

    grand_total = int(df["total"].sum()) if not df["total"].empty else 0
    df["sov_pct"] = (df["total"] / max(grand_total, 1) * 100.0).round(1)

    st.subheader("Share of Voice")
    st.caption("Donut charts show raw counts; SOV % bar highlights market share")

    col1, col2 = st.columns([2, 3])

    # Left column: two donuts stacked with textual percent indicator
    with col1:
        st.markdown("**Reddit — raw counts**")
        if df["reddit"].sum() == 0:
            st.warning("No Reddit mentions found.")
        else:
            chart = (
                alt.Chart(df)
                .mark_arc(innerRadius=60)
                .encode(
                    theta=alt.Theta("reddit:Q"),
                    color=alt.Color("brand:N", scale=alt.Scale(range=COLOR_PALETTE)),
                    tooltip=["brand:N", "reddit:Q", "total:Q", "sov_pct:Q"],
                )
                .properties(height=240)
            )
            st.altair_chart(chart, use_container_width=True)

        st.markdown("**Quora — raw counts**")
        if df["quora"].sum() == 0:
            st.warning("No Quora mentions found.")
        else:
            chart = (
                alt.Chart(df)
                .mark_arc(innerRadius=60)
                .encode(
                    theta=alt.Theta("quora:Q"),
                    color=alt.Color("brand:N", scale=alt.Scale(range=COLOR_PALETTE)),
                    tooltip=["brand:N", "quora:Q", "total:Q", "sov_pct:Q"],
                )
                .properties(height=240)
            )
            st.altair_chart(chart, use_container_width=True)

    # Right column: horizontal SOV %
    with col2:
        st.markdown("**Overall SOV % by Brand**")
        if grand_total == 0:
            st.warning("No mentions found to compute SOV percentages.")
        else:
            df_sorted = df.sort_values("sov_pct", ascending=True)
            bar = (
                alt.Chart(df_sorted)
                .mark_bar()
                .encode(
                    x=alt.X("sov_pct:Q", title="SOV %"),
                    y=alt.Y("brand:N", sort=alt.EncodingSortField(field="sov_pct", order="descending")),
                    color=alt.Color("brand:N", scale=alt.Scale(range=COLOR_PALETTE), legend=None),
                    tooltip=["brand:N", "total:Q", "sov_pct:Q"],
                )
                .properties(height=300)
            )
            text = bar.mark_text(align="left", dx=4, color="black").encode(text=alt.Text("sov_pct:Q", format=".1f"))
            st.altair_chart((bar + text).configure_view(strokeWidth=0), use_container_width=True)

    st.markdown("**SOV summary**")
    st.dataframe(df[["brand", "reddit", "quora", "total", "sov_pct"]].sort_values("sov_pct", ascending=False).reset_index(drop=True))
    return


def _stacked_progress_html(positive: int, neutral: int, negative: int) -> str:
    total = max(positive + neutral + negative, 1)
    p = round(positive / total * 100)
    n = round(neutral / total * 100)
    neg = 100 - p - n
    return f"""
    <div style="font-family:inherit">
      <div style="height:14px;border-radius:999px;overflow:hidden;display:flex;background:#f3f4f6;">
        <div style="width:{p}%;background:#10B981;"></div>
        <div style="width:{n}%;background:#F59E0B;"></div>
        <div style="width:{neg}%;background:#EF4444;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:12px;color:#374151;margin-top:8px;">
        <div><span style="display:inline-block;width:10px;height:10px;background:#10B981;margin-right:6px;"></span>{p}% Positive</div>
        <div><span style="display:inline-block;width:10px;height:10px;background:#F59E0B;margin-right:6px;"></span>{n}% Neutral</div>
        <div><span style="display:inline-block;width:10px;height:10px;background:#EF4444;margin-right:6px;"></span>{neg}% Negative</div>
      </div>
    </div>
    """


def render_sentiment_cards(sentiment_results: List[Dict]):
    """
    Polished sentiment cards. Accepts:
      [
        {"brand":"A", "reddit":{pos,neu,neg}, "quora":{...}},
        ...
      ]
    Renders a grid of brand cards with clear metrics and a stacked progress bar.
    """
    if not sentiment_results:
        st.info("No sentiment data to display.")
        return

    st.subheader("Sentiment overview")
    per_row = 3
    for i in range(0, len(sentiment_results), per_row):
        chunk = sentiment_results[i:i + per_row]
        cols = st.columns(len(chunk))
        for col, sr in zip(cols, chunk):
            with col:
                brand = sr.get("brand", "Unknown")
                r = sr.get("reddit") or {"positive": 0, "neutral": 0, "negative": 0}
                q = sr.get("quora") or {"positive": 0, "neutral": 0, "negative": 0}
                r_pos, r_neu, r_neg = int(r.get("positive", 0)), int(r.get("neutral", 0)), int(r.get("negative", 0))
                q_pos, q_neu, q_neg = int(q.get("positive", 0)), int(q.get("neutral", 0)), int(q.get("negative", 0))
                total_pos = r_pos + q_pos
                total_neu = r_neu + q_neu
                total_neg = r_neg + q_neg
                total = total_pos + total_neu + total_neg or 1
                neg_pct = round(total_neg / total * 100, 1)

                st.markdown(f"**{brand}**")
                st.metric("Mentions", str(total))
                st.metric("Negative %", f"{neg_pct}%")
                st.markdown(_stacked_progress_html(total_pos, total_neu, total_neg), unsafe_allow_html=True)
                st.markdown(f"<small>Reddit: {r_pos}/{r_neu}/{r_neg}  &nbsp;&nbsp; Quora: {q_pos}/{q_neu}/{q_neg}</small>", unsafe_allow_html=True)

    return


def render_industry_universe(brand: str, industry: str, brand_total: int, reddit_universe: int, quora_universe: int):
    """
    Visualize brand vs estimated industry universe for Reddit & Quora.
    This is sample-based and will be labeled as Estimated in the UI.
    """
    st.markdown(f"**{brand} — share vs the {industry.title()} industry (estimated)**")
    # Build two small datasets: Reddit and Quora (brand vs industry)
    r_rows = [{"label": f"{brand}", "count": max(brand_total, 0)}, {"label": f"{industry.title()} (sample)", "count": max(reddit_universe, 0)}]
    q_rows = [{"label": f"{brand}", "count": max(brand_total, 0)}, {"label": f"{industry.title()} (sample)", "count": max(quora_universe, 0)}]

    df_r = pd.DataFrame(r_rows)
    df_q = pd.DataFrame(q_rows)
    df_r["pct"] = (df_r["count"] / max(df_r["count"].sum(), 1) * 100.0).round(1)
    df_q["pct"] = (df_q["count"] / max(df_q["count"].sum(), 1) * 100.0).round(1)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Reddit: Brand vs Industry (sample)**")
        if df_r["count"].sum() == 0:
            st.info("No sample data for Reddit.")
        else:
            chart = (
                alt.Chart(df_r)
                .mark_arc(innerRadius=60)
                .encode(theta=alt.Theta("count:Q"), color=alt.Color("label:N", scale=alt.Scale(range=["#06b6d4","#6366f1"])), tooltip=["label:N","count:Q","pct:Q"])
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(df_r[["label","count","pct"]])

    with col2:
        st.markdown("**Quora: Brand vs Industry (sample)**")
        if df_q["count"].sum() == 0:
            st.info("No sample data for Quora.")
        else:
            chart = (
                alt.Chart(df_q)
                .mark_arc(innerRadius=60)
                .encode(theta=alt.Theta("count:Q"), color=alt.Color("label:N", scale=alt.Scale(range=["#10b981","#f59e0b"])), tooltip=["label:N","count:Q","pct:Q"])
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(df_q[["label","count","pct"]])

    st.markdown("_Note: Industry counts are sample-based estimates computed by searching the industry keyword within the selected date range._")
    return