import os
import json
import time
import logging

try:
    import praw
except Exception:
    praw = None

logger = logging.getLogger("reddit_stream")

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "offpage-sentiment-analyzer/1.0")

def _get_reddit():
    if praw is None:
        raise RuntimeError("praw is not installed. Run `pip install praw`")
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
        raise RuntimeError("Reddit credentials not set")
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

def _write_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def start_submission_stream(brand, out_path="streamed_mentions.jsonl", sleep_on_error=5):
    reddit = _get_reddit()
    logger.info("🚀 Starting submission stream for '%s' ...", brand)

    seen = set()

    while True:
        try:
            for sub in reddit.subreddit("all").stream.submissions(skip_existing=True):
                if sub is None:
                    continue

                text_blob = f"{sub.title}\n{sub.selftext or ''}".lower()
                if brand.lower() not in text_blob:
                    continue

                if sub.id in seen:
                    continue
                seen.add(sub.id)

                obj = {
                    "platform": "reddit",
                    "type": "submission",
                    "id": sub.id,
                    "title": sub.title,
                    "text": (sub.selftext or "")[:4000],
                    "url": f"https://reddit.com{sub.permalink}",
                    "created_utc": sub.created_utc,
                    "subreddit": sub.subreddit.display_name
                }
                _write_jsonl(out_path, obj)

        except Exception as e:
            logger.error("Submission stream failed: %s. Restarting in %s sec...", e, sleep_on_error)
            time.sleep(sleep_on_error)

def start_comment_stream(brand, out_path="streamed_mentions.jsonl", sleep_on_error=5):
    reddit = _get_reddit()
    logger.info("🚀 Starting comment stream for '%s' ...", brand)

    seen = set()

    while True:
        try:
            for c in reddit.subreddit("all").stream.comments(skip_existing=True):
                if c is None:
                    continue

                if brand.lower() not in c.body.lower():
                    continue

                if c.id in seen:
                    continue
                seen.add(c.id)

                obj = {
                    "platform": "reddit",
                    "type": "comment",
                    "id": c.id,
                    "text": c.body[:4000],
                    "url": f"https://reddit.com{c.permalink}",
                    "created_utc": c.created_utc,
                    "subreddit": c.subreddit.display_name
                }
                _write_jsonl(out_path, obj)

        except Exception as e:
            logger.error("Comment stream failed: %s. Restarting in %s sec...", e, sleep_on_error)
            time.sleep(sleep_on_error)
