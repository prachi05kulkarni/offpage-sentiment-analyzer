import os
import time
import logging
import re
from datetime import datetime, timedelta
import requests
import html

try:
    import praw
except Exception:
    praw = None

logger = logging.getLogger("reddit_client")

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "offpage-sentiment-analyzer/0.1")

if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
    logger.warning("Reddit credentials are not set. fetch_reddit_mentions will use PUBLIC JSON fallback (rate limited).")


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


def _word_match(term, text):
    if not term:
        return False
    try:
        return re.search(rf"\b{re.escape(term)}\b", (text or "").lower(), flags=re.IGNORECASE) is not None
    except Exception:
        return term.lower() in (text or "").lower()

def _fetch_via_praw(brand, competitor, days, limit):
    reddit = _get_reddit()

    # Build query to match either term, if competitor provided.
    if competitor:
        query = f'("{brand}" OR "{competitor}")'
    else:
        query = f'"{brand}"'

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    results = []
    # PRAW search
    for submission in reddit.subreddit("all").search(query, time_filter="month", limit=limit, sort='new'):
        try:
            created = datetime.utcfromtimestamp(getattr(submission, "created_utc", 0))
            if created and created < start_time:
                continue
            title = getattr(submission, "title", "") or ""
            selftext = getattr(submission, "selftext", "") or ""
            text = f"{title}\n{selftext}"
            url = ""
            if hasattr(submission, "permalink") and getattr(submission, "permalink", None):
                url = f"https://reddit.com{submission.permalink}"
            elif getattr(submission, "url", None):
                url = submission.url

            matched_terms = []
            if _word_match(brand, text) or _word_match(brand, title):
                matched_terms.append("brand")
            if competitor and (_word_match(competitor, text) or _word_match(competitor, title)):
                matched_terms.append("competitor")

            results.append({
                "platform": "reddit",
                "title": title,
                "text": text,
                "url": url,
                "score": getattr(submission, "score", 0) or 0,
                "comments": getattr(submission, "num_comments", 0) or 0,
                "created_utc": getattr(submission, "created_utc", None),
                "subreddit": getattr(submission.subreddit, "display_name", None) if getattr(submission, "subreddit", None) else None,
                "matched_terms": matched_terms
            })
        except Exception as ex:
            logger.debug("Skipping PRAW submission: %s", ex)
    return results


def _fetch_via_public_json(brand, competitor, days, limit):
    """
    Fallback method using Reddit's public RSS feeds.
    Strictly read-only, rate-limited, no auth required.
    RSS is more likely to work without keys than JSON.
    """
    import xml.etree.ElementTree as ET
    
    # Build query
    if competitor:
        q = f'("{brand}" OR "{competitor}")'
    else:
        q = f'"{brand}"'
    
    # URL encoded query
    url = "https://old.reddit.com/search.rss"
    # User-Agent still important
    headers = {
        "User-Agent": "offpage-sentiment-test-v1.0 (by /u/tweet_sentiment_bot)"
    }
    params = {
        "q": q,
        "sort": "new",
        "limit": min(limit, 100),
        "t": "month"
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 429:
            logger.warning("Reddit RSS returned 429 (Too Many Requests).")
            return []
        resp.raise_for_status()
        
        # Parse XML
        # Remove namespace prefixes to make find simpler or handle them
        # simple hack: replace xmlns="..." with empty
        content = resp.text
        content = re.sub(r'\sxmlns="[^"]+"', '', content, count=1)
        root = ET.fromstring(content)
        
        results = []
        # atom namespace often used in reddit rss, but we strip default xmlns.
        # Check if it uses 'entry'
        entries = root.findall(".//entry")
        if not entries:
            # try channel/item (RSS 2.0)
            entries = root.findall(".//item")
            
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        for entry in entries:
            # Atom
            title_node = entry.find("title")
            title = title_node.text if title_node is not None else ""
            
            # Content is usually in <content type="html">
            content_node = entry.find("content")
            # if not atom, try description
            if content_node is None:
                content_node = entry.find("description")
                
            raw_html = content_node.text if content_node is not None else ""
            # simple html to text
            text = re.sub(r'<[^>]+>', ' ', html.unescape(raw_html))
            text = f"{title}\n{text}"
            
            # link
            link_node = entry.find("link")
            url_val = ""
            if link_node is not None:
                url_val = link_node.get("href") # atom
                if not url_val:
                    url_val = link_node.text # rss 2.0
            
            # published/updated
            updated_node = entry.find("updated")
            if updated_node is None:
                updated_node = entry.find("pubDate")

            created_utc = 0
            if updated_node is not None and updated_node.text:
                try:
                    # simplistic ISO parsing 2023-10-27T10:00:00+00:00
                    # or util
                    dt = datetime.fromisoformat(updated_node.text.replace('Z', '+00:00'))
                    created_utc = dt.timestamp()
                except:
                    created_utc = time.time() # fallback
            
            created_dt = datetime.utcfromtimestamp(created_utc)
            if created_dt < start_time:
                continue

            matched_terms = []
            if _word_match(brand, text) or _word_match(brand, title):
                matched_terms.append("brand")
            if competitor and (_word_match(competitor, text) or _word_match(competitor, title)):
                matched_terms.append("competitor")
            
            if not matched_terms:
                continue

            results.append({
                "platform": "reddit",
                "title": title,
                "text": text,
                "url": url_val,
                "score": 0, # RSS doesn't give score usually
                "comments": 0,
                "created_utc": created_utc,
                "subreddit": None, # difficult to parse from RSS easily without regex on link
                "matched_terms": matched_terms
            })
            
        return results

    except Exception as e:
        logger.warning("RSS Fallback failed: %s", e)
        return []

def fetch_reddit_mentions(brand, competitor=None, days=14, limit=200):
    """
    Fetches matches using PRAW if available, otherwise falls back to public JSON.
    """
    def _do_search():
        if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
            return _fetch_via_praw(brand, competitor, days, limit)
        else:
            return _fetch_via_public_json(brand, competitor, days, limit)

    try:
        return _retry(_do_search, max_attempts=3, base_delay=1.0)
    except Exception as e:
        logger.error("Reddit fetch failed: %s", e)
        return []
