from reddit_client import fetch_reddit_mentions
import logging

logging.basicConfig(level=logging.INFO)
print("Fetching mentions for 'Python' (no auth)...")
mentions = fetch_reddit_mentions("Python", limit=5)
print(f"Found {len(mentions)} mentions.")
for m in mentions:
    print(f"- {m['title']}")
