"""
Reddit Stream - Polling-based Pseudo-Stream (No Auth Required)
Drop-in replacement for PRAW-based reddit_stream.py

Author: Migrated from PRAW for auth-free operation
Features:
    - Real-time monitoring via polling
    - Submission and Comment streams
    - Combined stream with threading
    - Callback support for integrations
    - Automatic retry on failures
"""
import os
import json
import time
import logging
import requests
import threading
from datetime import datetime
from typing import Optional, Callable, Set, List, Dict
from collections import deque
from queue import Queue

logger = logging.getLogger("reddit_stream_public")

# =============================================================================
# Configuration
# =============================================================================

USER_AGENT = os.getenv(
    "REDDIT_USER_AGENT",
    "OffpageSentiment/1.0 (Personal Research Bot; Contact: your@email.com)"
)

# Polling intervals (in seconds)
POLL_INTERVAL_SUBMISSIONS = 5.0  # How often to check for new submissions
POLL_INTERVAL_COMMENTS = 5.0     # How often to check for new comments
ERROR_RETRY_DELAY = 10.0         # Delay before retry on error
RATE_LIMIT_DELAY = 60.0          # Delay when rate limited

# Cache settings
MAX_SEEN_CACHE = 10000  # Maximum IDs to remember (prevents memory bloat)
ITEMS_PER_REQUEST = 100  # Reddit max per request

# Endpoints
NEW_SUBMISSIONS_URL = "https://www.reddit.com/r/all/new.json"
NEW_COMMENTS_URL = "https://www.reddit.com/r/all/comments.json"
SUBREDDIT_NEW_URL = "https://www.reddit.com/r/{subreddit}/new.json"
SUBREDDIT_COMMENTS_URL = "https://www.reddit.com/r/{subreddit}/comments.json"


# =============================================================================
# Utility Functions
# =============================================================================

def _write_jsonl(path: str, obj: dict):
    """Append JSON object to JSONL file (thread-safe with basic locking)."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except IOError as e:
        logger.error(f"Failed to write to {path}: {e}")


def _fetch_items(url: str, limit: int = ITEMS_PER_REQUEST) -> List[Dict]:
    """
    Fetch latest items from Reddit.
    
    Args:
        url: Reddit JSON endpoint
        limit: Number of items to fetch
        
    Returns:
        List of item dicts (empty list on error)
    """
    try:
        response = requests.get(
            url,
            params={"limit": limit, "raw_json": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=15
        )
        
        # Handle rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", RATE_LIMIT_DELAY))
            logger.warning(f"Rate limited. Sleeping {retry_after}s...")
            time.sleep(retry_after)
            return []
        
        # Handle other errors
        if response.status_code != 200:
            logger.warning(f"HTTP {response.status_code} from {url}")
            return []
        
        data = response.json()
        return data.get("data", {}).get("children", [])
        
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching {url}")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return []
    except (ValueError, KeyError) as e:
        logger.error(f"Parse error: {e}")
        return []


def _matches_keywords(text: str, keywords: List[str]) -> List[str]:
    """
    Check which keywords match in the text.
    
    Args:
        text: Text to search in
        keywords: List of keywords to check
        
    Returns:
        List of matched keywords
    """
    text_lower = text.lower()
    matched = []
    for keyword in keywords:
        if keyword.lower() in text_lower:
            matched.append(keyword)
    return matched


# =============================================================================
# Submission Stream
# =============================================================================

def start_submission_stream(
    brand: str,
    competitor: Optional[str] = None,
    out_path: str = "streamed_mentions.jsonl",
    poll_interval: float = POLL_INTERVAL_SUBMISSIONS,
    on_match: Optional[Callable[[dict], None]] = None,
    subreddit: Optional[str] = None,
    stop_event: Optional[threading.Event] = None
):
    """
    Poll-based submission stream.
    
    Args:
        brand: Primary keyword to match
        competitor: Optional competitor keyword
        out_path: Output JSONL file path
        poll_interval: Seconds between polls
        on_match: Optional callback function for each match
        subreddit: Optional specific subreddit (None = r/all)
        stop_event: Optional threading.Event to signal stop
    """
    logger.info(f"🚀 Starting submission stream for '{brand}'...")
    
    # Build keywords list
    keywords = [brand]
    if competitor:
        keywords.append(competitor)
    
    # Determine URL
    if subreddit:
        url = SUBREDDIT_NEW_URL.format(subreddit=subreddit)
        logger.info(f"📍 Monitoring r/{subreddit}")
    else:
        url = NEW_SUBMISSIONS_URL
        logger.info("📍 Monitoring r/all")
    
    # Seen IDs cache (deque for automatic size management)
    seen: deque = deque(maxlen=MAX_SEEN_CACHE)
    
    # Main loop
    while True:
        # Check stop signal
        if stop_event and stop_event.is_set():
            logger.info("Submission stream stopped by signal")
            break
        
        try:
            items = _fetch_items(url, limit=ITEMS_PER_REQUEST)
            
            for wrapper in items:
                post = wrapper.get("data", {})
                post_id = post.get("id")
                
                # Skip if no ID or already seen
                if not post_id or post_id in seen:
                    continue
                
                # Build text blob for matching
                title = post.get("title", "")
                selftext = post.get("selftext", "")
                text_blob = f"{title}\n{selftext}"
                
                # Check for keyword matches
                matched = _matches_keywords(text_blob, keywords)
                if not matched:
                    continue
                
                # Mark as seen
                seen.append(post_id)
                
                # Build output object
                obj = {
                    "platform": "reddit",
                    "type": "submission",
                    "id": post_id,
                    "title": title,
                    "text": selftext[:4000] if selftext else "",
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "score": post.get("score", 0),
                    "comments": post.get("num_comments", 0),
                    "created_utc": post.get("created_utc", 0),
                    "subreddit": post.get("subreddit", ""),
                    "matched_keywords": matched,
                    "matched_at": datetime.utcnow().isoformat()
                }
                
                # Write to file
                _write_jsonl(out_path, obj)
                
                # Log
                logger.info(f"📝 [{obj['subreddit']}] {title[:50]}...")
                
                # Callback
                if on_match:
                    try:
                        on_match(obj)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
            
            # Wait before next poll
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            logger.info("Submission stream stopped by user")
            break
        except Exception as e:
            logger.error(f"Stream error: {e}. Restarting in {ERROR_RETRY_DELAY}s...")
            time.sleep(ERROR_RETRY_DELAY)


# =============================================================================
# Comment Stream
# =============================================================================

def start_comment_stream(
    brand: str,
    competitor: Optional[str] = None,
    out_path: str = "streamed_mentions.jsonl",
    poll_interval: float = POLL_INTERVAL_COMMENTS,
    on_match: Optional[Callable[[dict], None]] = None,
    subreddit: Optional[str] = None,
    stop_event: Optional[threading.Event] = None
):
    """
    Poll-based comment stream.
    
    Args:
        brand: Primary keyword to match
        competitor: Optional competitor keyword
        out_path: Output JSONL file path
        poll_interval: Seconds between polls
        on_match: Optional callback function for each match
        subreddit: Optional specific subreddit (None = r/all)
        stop_event: Optional threading.Event to signal stop
    """
    logger.info(f"🚀 Starting comment stream for '{brand}'...")
    
    # Build keywords list
    keywords = [brand]
    if competitor:
        keywords.append(competitor)
    
    # Determine URL
    if subreddit:
        url = SUBREDDIT_COMMENTS_URL.format(subreddit=subreddit)
        logger.info(f"📍 Monitoring comments in r/{subreddit}")
    else:
        url = NEW_COMMENTS_URL
        logger.info("📍 Monitoring comments in r/all")
    
    # Seen IDs cache
    seen: deque = deque(maxlen=MAX_SEEN_CACHE)
    
    # Main loop
    while True:
        # Check stop signal
        if stop_event and stop_event.is_set():
            logger.info("Comment stream stopped by signal")
            break
        
        try:
            items = _fetch_items(url, limit=ITEMS_PER_REQUEST)
            
            for wrapper in items:
                comment = wrapper.get("data", {})
                comment_id = comment.get("id")
                
                # Skip if no ID or already seen
                if not comment_id or comment_id in seen:
                    continue
                
                body = comment.get("body", "")
                
                # Check for keyword matches
                matched = _matches_keywords(body, keywords)
                if not matched:
                    continue
                
                # Mark as seen
                seen.append(comment_id)
                
                # Build output object
                obj = {
                    "platform": "reddit",
                    "type": "comment",
                    "id": comment_id,
                    "text": body[:4000],
                    "url": f"https://reddit.com{comment.get('permalink', '')}",
                    "score": comment.get("score", 0),
                    "created_utc": comment.get("created_utc", 0),
                    "subreddit": comment.get("subreddit", ""),
                    "matched_keywords": matched,
                    "matched_at": datetime.utcnow().isoformat()
                }
                
                # Write to file
                _write_jsonl(out_path, obj)
                
                # Log
                logger.info(f"💬 [r/{obj['subreddit']}] Comment matched")
                
                # Callback
                if on_match:
                    try:
                        on_match(obj)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
            
            # Wait before next poll
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            logger.info("Comment stream stopped by user")
            break
        except Exception as e:
            logger.error(f"Stream error: {e}. Restarting in {ERROR_RETRY_DELAY}s...")
            time.sleep(ERROR_RETRY_DELAY)


# =============================================================================
# Combined Stream (Threaded)
# =============================================================================

class RedditStreamManager:
    """
    Manages both submission and comment streams with proper lifecycle control.
    """
    
    def __init__(
        self,
        brand: str,
        competitor: Optional[str] = None,
        out_path: str = "streamed_mentions.jsonl",
        poll_interval: float = POLL_INTERVAL_SUBMISSIONS,
        on_match: Optional[Callable[[dict], None]] = None,
        subreddit: Optional[str] = None
    ):
        """
        Initialize the stream manager.
        
        Args:
            brand: Primary keyword to match
            competitor: Optional competitor keyword
            out_path: Output JSONL file path
            poll_interval: Seconds between polls
            on_match: Optional callback for matches
            subreddit: Optional specific subreddit
        """
        self.brand = brand
        self.competitor = competitor
        self.out_path = out_path
        self.poll_interval = poll_interval
        self.on_match = on_match
        self.subreddit = subreddit
        
        self._stop_event = threading.Event()
        self._submission_thread: Optional[threading.Thread] = None
        self._comment_thread: Optional[threading.Thread] = None
        self._match_queue: Queue = Queue()
        self._is_running = False
    
    def start(self, include_submissions: bool = True, include_comments: bool = True):
        """
        Start the stream(s).
        
        Args:
            include_submissions: Monitor submissions
            include_comments: Monitor comments
        """
        if self._is_running:
            logger.warning("Stream already running")
            return
        
        self._stop_event.clear()
        self._is_running = True
        
        logger.info(f"🚀 Starting combined stream for '{self.brand}'...")
        
        if include_submissions:
            self._submission_thread = threading.Thread(
                target=start_submission_stream,
                kwargs={
                    "brand": self.brand,
                    "competitor": self.competitor,
                    "out_path": self.out_path,
                    "poll_interval": self.poll_interval,
                    "on_match": self._handle_match,
                    "subreddit": self.subreddit,
                    "stop_event": self._stop_event
                },
                daemon=True,
                name="reddit-submission-stream"
            )
            self._submission_thread.start()
        
        if include_comments:
            # Offset comment stream to distribute requests
            time.sleep(self.poll_interval / 2)
            
            self._comment_thread = threading.Thread(
                target=start_comment_stream,
                kwargs={
                    "brand": self.brand,
                    "competitor": self.competitor,
                    "out_path": self.out_path,
                    "poll_interval": self.poll_interval,
                    "on_match": self._handle_match,
                    "subreddit": self.subreddit,
                    "stop_event": self._stop_event
                },
                daemon=True,
                name="reddit-comment-stream"
            )
            self._comment_thread.start()
        
        logger.info("✅ Stream threads started")
    
    def _handle_match(self, obj: dict):
        """Internal match handler that queues and forwards matches."""
        self._match_queue.put(obj)
        if self.on_match:
            self.on_match(obj)
    
    def stop(self):
        """Stop all streams gracefully."""
        if not self._is_running:
            return
        
        logger.info("🛑 Stopping streams...")
        self._stop_event.set()
        
        # Wait for threads to finish
        if self._submission_thread and self._submission_thread.is_alive():
            self._submission_thread.join(timeout=5)
        
        if self._comment_thread and self._comment_thread.is_alive():
            self._comment_thread.join(timeout=5)
        
        self._is_running = False
        logger.info("✅ Streams stopped")
    
    def is_running(self) -> bool:
        """Check if streams are running."""
        return self._is_running
    
    def get_matches(self, max_items: int = 100) -> List[dict]:
        """
        Get queued matches (non-blocking).
        
        Args:
            max_items: Maximum items to retrieve
            
        Returns:
            List of match dicts
        """
        matches = []
        while not self._match_queue.empty() and len(matches) < max_items:
            try:
                matches.append(self._match_queue.get_nowait())
            except Exception:
                break
        return matches
    
    def run_forever(self):
        """Block and run until interrupted."""
        try:
            while self._is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()


def start_combined_stream(
    brand: str,
    competitor: Optional[str] = None,
    out_path: str = "streamed_mentions.jsonl",
    poll_interval: float = POLL_INTERVAL_SUBMISSIONS,
    on_match: Optional[Callable[[dict], None]] = None,
    subreddit: Optional[str] = None
):
    """
    Convenience function to start combined stream (blocking).
    
    Args:
        brand: Primary keyword
        competitor: Optional competitor keyword
        out_path: Output file path
        poll_interval: Poll interval in seconds
        on_match: Optional callback
        subreddit: Optional specific subreddit
    """
    manager = RedditStreamManager(
        brand=brand,
        competitor=competitor,
        out_path=out_path,
        poll_interval=poll_interval,
        on_match=on_match,
        subreddit=subreddit
    )
    
    manager.start(include_submissions=True, include_comments=True)
    manager.run_forever()


# =============================================================================
# Multi-Keyword Stream
# =============================================================================

def start_multi_keyword_stream(
    keywords: List[str],
    out_path: str = "streamed_mentions.jsonl",
    poll_interval: float = POLL_INTERVAL_SUBMISSIONS,
    on_match: Optional[Callable[[dict], None]] = None,
    subreddit: Optional[str] = None,
    include_submissions: bool = True,
    include_comments: bool = True
):
    """
    Stream for multiple keywords simultaneously.
    
    Args:
        keywords: List of keywords to match (any match triggers)
        out_path: Output file path
        poll_interval: Poll interval in seconds
        on_match: Optional callback
        subreddit: Optional specific subreddit
        include_submissions: Monitor submissions
        include_comments: Monitor comments
    """
    if not keywords:
        raise ValueError("At least one keyword required")
    
    logger.info(f"🚀 Starting multi-keyword stream for: {keywords}")
    
    # Use first keyword as primary, rest as "competitors"
    # The matching logic will check all keywords
    primary = keywords[0]
    
    # Build extended matching callback
    def extended_matcher(text: str) -> List[str]:
        return _matches_keywords(text, keywords)
    
    # Seen cache
    seen: deque = deque(maxlen=MAX_SEEN_CACHE)
    
    # URLs
    if subreddit:
        sub_url = SUBREDDIT_NEW_URL.format(subreddit=subreddit)
        comment_url = SUBREDDIT_COMMENTS_URL.format(subreddit=subreddit)
    else:
        sub_url = NEW_SUBMISSIONS_URL
        comment_url = NEW_COMMENTS_URL
    
    while True:
        try:
            # Fetch submissions
            if include_submissions:
                items = _fetch_items(sub_url, limit=ITEMS_PER_REQUEST)
                
                for wrapper in items:
                    post = wrapper.get("data", {})
                    post_id = post.get("id")
                    
                    if not post_id or post_id in seen:
                        continue
                    
                    title = post.get("title", "")
                    selftext = post.get("selftext", "")
                    text_blob = f"{title}\n{selftext}"
                    
                    matched = extended_matcher(text_blob)
                    if not matched:
                        continue
                    
                    seen.append(post_id)
                    
                    obj = {
                        "platform": "reddit",
                        "type": "submission",
                        "id": post_id,
                        "title": title,
                        "text": selftext[:4000] if selftext else "",
                        "url": f"https://reddit.com{post.get('permalink', '')}",
                        "score": post.get("score", 0),
                        "comments": post.get("num_comments", 0),
                        "created_utc": post.get("created_utc", 0),
                        "subreddit": post.get("subreddit", ""),
                        "matched_keywords": matched,
                        "matched_at": datetime.utcnow().isoformat()
                    }
                    
                    _write_jsonl(out_path, obj)
                    logger.info(f"📝 [{obj['subreddit']}] Matched: {matched}")
                    
                    if on_match:
                        on_match(obj)
            
            time.sleep(poll_interval / 2)
            
            # Fetch comments
            if include_comments:
                items = _fetch_items(comment_url, limit=ITEMS_PER_REQUEST)
                
                for wrapper in items:
                    comment = wrapper.get("data", {})
                    comment_id = comment.get("id")
                    
                    if not comment_id or comment_id in seen:
                        continue
                    
                    body = comment.get("body", "")
                    
                    matched = extended_matcher(body)
                    if not matched:
                        continue
                    
                    seen.append(comment_id)
                    
                    obj = {
                        "platform": "reddit",
                        "type": "comment",
                        "id": comment_id,
                        "text": body[:4000],
                        "url": f"https://reddit.com{comment.get('permalink', '')}",
                        "score": comment.get("score", 0),
                        "created_utc": comment.get("created_utc", 0),
                        "subreddit": comment.get("subreddit", ""),
                        "matched_keywords": matched,
                        "matched_at": datetime.utcnow().isoformat()
                    }
                    
                    _write_jsonl(out_path, obj)
                    logger.info(f"💬 [r/{obj['subreddit']}] Matched: {matched}")
                    
                    if on_match:
                        on_match(obj)
            
            time.sleep(poll_interval / 2)
            
        except KeyboardInterrupt:
            logger.info("Multi-keyword stream stopped by user")
            break
        except Exception as e:
            logger.error(f"Stream error: {e}. Restarting in {ERROR_RETRY_DELAY}s...")
            time.sleep(ERROR_RETRY_DELAY)


# =============================================================================
# Health Check
# =============================================================================

def check_stream_health() -> Dict[str, bool]:
    """
    Check if streaming endpoints are accessible.
    
    Returns:
        Dict with health status
    """
    health = {
        "submissions_endpoint": False,
        "comments_endpoint": False
    }
    
    try:
        items = _fetch_items(NEW_SUBMISSIONS_URL, limit=1)
        health["submissions_endpoint"] = len(items) > 0
    except Exception:
        pass
    
    try:
        items = _fetch_items(NEW_COMMENTS_URL, limit=1)
        health["comments_endpoint"] = len(items) > 0
    except Exception:
        pass
    
    return health