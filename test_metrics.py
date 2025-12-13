# test_metrics.py
# Small script to run locally to exercise metrics functions without external APIs.
from metrics import (
    build_mentions_df, dedupe_by_url_and_text, mentions_volume, compute_sov,
    sentiment_breakdown, net_sentiment_score, share_of_positive_sentiment,
    sentiment_trend, engagement_metrics, top_communities, top_mentions,
    crisis_spike_alerts, volume_vs_sentiment_ratio, influencer_impact
)
import json

reddit_items = [
    {"platform": "reddit", "title": "Love BrandX", "text": "I love BrandX. Their product is great.", "url": "https://r/1", "score": 10, "comments": 2, "created_utc": 1700000000, "subreddit": "examples", "matched_terms": ["BrandX"], "_raw": {"label": "positive"}},
    {"platform": "reddit", "title": "BrandX helped", "text": "CompetitorY has some issues, but BrandX helped me.", "url": "https://r/2", "score": 5, "comments": 1, "created_utc": 1700000100, "subreddit": "examples", "matched_terms": ["BrandX","CompetitorY"], "_raw": {"label": "positive"}},
    {"platform": "reddit", "title": "CompetitorY promo", "text": "CompetitorY is cheaper.", "url": "https://r/3", "score": 2, "comments": 0, "created_utc": 1699999000, "subreddit": "examples", "matched_terms": ["CompetitorY"], "_raw": {"label": "neutral"}},
    {"platform": "reddit", "title": "Pineapple", "text": "Pineapple is tasty.", "url": "https://r/4", "score": 0, "comments": 0, "created_utc": 1699998000, "subreddit": "food", "matched_terms": [], "_raw": {"label": "neutral"}}
]

quora_items = [
    {"platform": "quora", "title": "Q: is BrandX better", "text": "Is BrandX better than CompetitorY?", "url": "https://q/1", "created_utc": 1700000200, "matched_terms": ["BrandX","CompetitorY"], "_raw": {"label": "neutral"}},
    {"platform": "quora", "title": "Why CompetitorY", "text": "I prefer CompetitorY for price.", "url": "https://q/2", "created_utc": 1699997000, "matched_terms": ["CompetitorY"], "_raw": {"label": "negative"}},
    {"platform": "quora", "title": "BrandX help", "text": "BrandX customer service was helpful.", "url": "https://q/3", "created_utc": 1699996000, "matched_terms": ["BrandX"], "_raw": {"label": "positive"}}
]

df = build_mentions_df(reddit_items, quora_items)
print("DEDUPE BEFORE:", len(df))
df = dedupe_by_url_and_text(df)
print("DEDUPE AFTER:", len(df))

print("\n=== Mentions Volume ===")
print(mentions_volume(df, freq='D').to_json(orient='split', default_handler=str))

print("\n=== Share of Voice ===")
print(json.dumps(compute_sov(df), indent=2, default=str))

print("\n=== Sentiment Breakdown ===")
sb = sentiment_breakdown(df)
print("Counts:\n", sb['counts'])
print("Pct:\n", sb['percentages'])

print("\n=== Net Sentiment Score ===")
print(json.dumps(net_sentiment_score(df), indent=2))

print("\n=== Share of Positive Sentiment ===")
print(json.dumps(share_of_positive_sentiment(df), indent=2))

print("\n=== Sentiment Trend ===")
st = sentiment_trend(df, freq='D')
for b, g in st.items():
    print(b, g.to_json(orient='split', default_handler=str))

print("\n=== Engagement Metrics ===")
print(json.dumps(engagement_metrics(df), indent=2))

print("\n=== Top Mentions ===")
print(json.dumps(top_mentions(df, top_n=5), indent=2))

print("\n=== Volume vs Sentiment Ratio ===")
print(json.dumps(volume_vs_sentiment_ratio(df), indent=2))

print("\n=== Influencer Impact ===")
print(json.dumps(influencer_impact(df), indent=2))