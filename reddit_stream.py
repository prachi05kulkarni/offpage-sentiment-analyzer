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
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "offpage-sentiment-analyzer/0.1")

def _get_reddit():
    if praw is None:
        raise RuntimeError("praw is not installed. Install with `pip install praw`")
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
        raise RuntimeError("Reddit credentials not set in environment")
    return praw.Reddit(client_id=REDDIT_CLIENT_ID,
                       client_secret=REDDIT_CLIENT_SECRET,
                       user_agent=REDDIT_USER_AGENT)

def start_submission_stream(brand: str, out_path: str = "streamed_mentions.jsonl", sleep_on_error: int = 5):
    reddit = _get_reddit()
    logger.info("Starting Reddit submission stream for brand=%s", brand)
    with open(out_path, "a", encoding="utf-8") as fh:
        stream = reddit.subreddit("all").stream.submissions(skip_existing=True)
        while True:
            try:
                for submission in stream:
                    if submission is None:
                        continue
                    title = getattr(submission, "title", "") or ""
                    selftext = getattr(submission, "selftext", "") or ""
                    text = f"{title}\n{selftext}"
                    if brand.lower() in text.lower():
                        obj = {
                            "platform": "reddit",
                            "type": "submission",
                            "id": getattr(submission, "id", None),
                            "title": title,
                            "text": text[:4000],
                            "url": f"https://reddit.com{submission.permalink}" if getattr(submission, "permalink", None) else getattr(submission, "url", ""),
                            "created_utc": getattr(submission, "created_utc", None),
                            "subreddit": getattr(submission.subreddit, "display_name", None)
                        }
                        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        fh.flush()
            except Exception as e:
                logger.exception("Stream error, sleeping %s sec: %s", sleep_on_error, e)
                time.sleep(sleep_on_error)

def start_comment_stream(brand: str, out_path: str = "streamed_mentions.jsonl", sleep_on_error: int = 5):
    reddit = _get_reddit()
    logger.info("Starting Reddit comment stream for brand=%s", brand)
    with open(out_path, "a", encoding="utf-8") as fh:
        stream = reddit.subreddit("all").stream.comments(skip_existing=True)
        while True:
            try:
                for comment in stream:
                    if comment is None:
                        continue
                    body = getattr(comment, "body", "") or ""
                    if brand.lower() in body.lower():
                        obj = {
                            "platform": "reddit",
                            "type": "comment",
                            "id": getattr(comment, "id", None),
                            "text": body[:4000],
                            "url": f"https://reddit.com{comment.permalink}" if getattr(comment, "permalink", None) else None,
                            "created_utc": getattr(comment, "created_utc", None),
                            "subreddit": getattr(comment.subreddit, "display_name", None)
                        }
                        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        fh.flush()
            except Exception as e:
                logger.exception("Stream error, sleeping %s sec: %s", sleep_on_error, e)
                time.sleep(sleep_on_error)