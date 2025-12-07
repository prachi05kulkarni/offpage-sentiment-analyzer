import os
import time
import logging
from datetime import datetime, timedelta

try:
    import praw
except Exception:
    praw = None

logger = logging.getLogger("reddit_client")

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "offpage-sentiment-analyzer/0.1")

if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
    logger.warning("Reddit credentials are not set. fetch_reddit_mentions will return []")

def _retry(fn, max_attempts=3, base_delay=1.0):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            wait = base_delay * (2 ** (attempt - 1))
            logger.warning("Attempt %s failed with %s; retrying in %.1fs", attempt, e, wait)
            time.sleep(wait)
    raise RuntimeError("All retry attempts failed")

def _get_reddit():
    if praw is None:
        raise RuntimeError("praw is not installed. Install via `pip install praw`")
    return praw.Reddit(client_id=REDDIT_CLIENT_ID,
                       client_secret=REDDIT_CLIENT_SECRET,
                       user_agent=REDDIT_USER_AGENT)

def fetch_reddit_mentions(brand, competitor=None, days=14, limit=200):
    """
    Uses PRAW search over r/all to fetch recent submissions mentioning brand.
    Returns list of dicts with keys: platform, title, text, url, score, comments, created_utc, subreddit
    """
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
        logger.info("Missing Reddit credentials; returning empty list.")
        return []

    def _do_search():
        reddit = _get_reddit()
        query = f'"{brand}"'
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        results = []
        for submission in reddit.subreddit("all").search(query, limit=limit, sort='new'):
            try:
                created = datetime.utcfromtimestamp(getattr(submission, "created_utc", 0))
                if created and created < start_time:
                    continue
                text = (getattr(submission, "title", "") or "") + "\n" + (getattr(submission, "selftext", "") or "")
                url = ""
                if hasattr(submission, "permalink") and getattr(submission, "permalink", None):
                    url = f"https://reddit.com{submission.permalink}"
                elif getattr(submission, "url", None):
                    url = submission.url
                results.append({
                    "platform": "reddit",
                    "title": getattr(submission, "title", "") or "",
                    "text": text,
                    "url": url,
                    "score": getattr(submission, "score", 0) or 0,
                    "comments": getattr(submission, "num_comments", 0) or 0,
                    "created_utc": getattr(submission, "created_utc", None),
                    "subreddit": getattr(submission.subreddit, "display_name", None) if getattr(submission, "subreddit", None) else None
                })
            except Exception as ex:
                logger.debug("Skipping submission due to extraction error: %s", ex)
        return results

    try:
        return _retry(_do_search, max_attempts=3, base_delay=1.0)
    except Exception as e:
        logger.error("Reddit fetch failed: %s", e)
        return []