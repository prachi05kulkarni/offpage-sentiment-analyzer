"""
Reddit Client - Public JSON API (No Auth Required)
Drop-in replacement for PRAW-based reddit_client.py

Author: Migrated from PRAW for auth-free operation
Features:
    - Historical search via Reddit Public JSON API
    - Pullpush.io fallback for deeper archives
    - Combined fetcher with deduplication
    - Exact same output schema as PRAW version
"""
import os
import time
import logging
import re
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from urllib.parse import quote_plus

logger = logging.getLogger("reddit_client_public")

# =============================================================================
# Configuration
# =============================================================================

USER_AGENT = os.getenv(
    "REDDIT_USER_AGENT",
    "OffpageSentiment/1.0 (Personal Research Bot; Contact: your@email.com)"
)
REQUEST_DELAY = 2.0  # Seconds between requests (safe rate limiting)
MAX_RETRIES = 3
RESULTS_PER_PAGE = 100  # Reddit max per request

# Base URLs
REDDIT_SEARCH_URL = "https://www.reddit.com/search.json"
REDDIT_SUBREDDIT_SEARCH_URL = "https://www.reddit.com/r/{subreddit}/search.json"

# Pullpush.io (Pushshift successor) URLs
PULLPUSH_SUBMISSION_URL = "https://api.pullpush.io/reddit/search/submission"
PULLPUSH_COMMENT_URL = "https://api.pullpush.io/reddit/search/comment"


# =============================================================================
# Utility Functions
# =============================================================================

def _ui_log(msg: str):
    """Log to Streamlit session_state and Console."""
    print(msg)
    try:
        from streamlit import session_state
        if "logs" not in session_state:
            session_state["logs"] = []
        session_state["logs"].append(msg)
        session_state["logs"] = session_state["logs"][-300:]
    except Exception:
        pass


def _word_match(term: str, text: str) -> bool:
    """Check if term exists as a whole word in text."""
    if not term:
        return False
    try:
        pattern = rf"\b{re.escape(term)}\b"
        return re.search(pattern, (text or ""), flags=re.IGNORECASE) is not None
    except Exception:
        return term.lower() in (text or "").lower()


def _make_request(
    url: str,
    params: dict,
    retry_count: int = 0,
    timeout: int = 30
) -> Optional[dict]:
    """
    Make HTTP request with rate limiting and retries.
    
    Args:
        url: API endpoint URL
        params: Query parameters
        retry_count: Current retry attempt
        timeout: Request timeout in seconds
        
    Returns:
        JSON response dict or None on failure
    """
    headers = {"User-Agent": USER_AGENT}
    
    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout
        )
        
        # Handle rate limiting (HTTP 429)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            _ui_log(f"[REDDIT] Rate limited. Waiting {retry_after}s...")
            time.sleep(retry_after)
            if retry_count < MAX_RETRIES:
                return _make_request(url, params, retry_count + 1, timeout)
            return None
        
        # Handle server errors with retry
        if response.status_code >= 500:
            if retry_count < MAX_RETRIES:
                wait_time = REQUEST_DELAY * (retry_count + 1)
                _ui_log(f"[REDDIT] Server error {response.status_code}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                return _make_request(url, params, retry_count + 1, timeout)
            return None
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.Timeout:
        logger.warning(f"Request timeout for {url}")
        if retry_count < MAX_RETRIES:
            time.sleep(REQUEST_DELAY * (retry_count + 1))
            return _make_request(url, params, retry_count + 1, timeout)
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        if retry_count < MAX_RETRIES:
            time.sleep(REQUEST_DELAY * (retry_count + 1))
            return _make_request(url, params, retry_count + 1, timeout)
        return None


def _get_time_filter(days: int) -> str:
    """Map days to Reddit's time filter buckets."""
    if days <= 1:
        return "day"
    elif days <= 7:
        return "week"
    elif days <= 30:
        return "month"
    elif days <= 365:
        return "year"
    return "all"


# =============================================================================
# Reddit Public JSON API - Primary Fetcher
# =============================================================================

def fetch_reddit_mentions(
    brand: str,
    competitor: Optional[str] = None,
    days: int = 14,
    limit: int = 500,
    include_comments: bool = False,
    subreddit: Optional[str] = None
) -> List[Dict]:
    """
    Fetches Reddit mentions using Public JSON API.
    
    Args:
        brand: Primary keyword to search
        competitor: Optional competitor keyword
        days: Number of days to look back
        limit: Maximum number of results
        include_comments: Whether to also fetch comments
        subreddit: Optional specific subreddit to search (None = all)
    
    Returns:
        List of mention dicts with schema:
        {
            "platform": "reddit",
            "title": str,
            "text": str,
            "url": str,
            "score": int,
            "comments": int,
            "created_utc": float,
            "subreddit": str,
            "matched_terms": list,
            "id": str
        }
    """
    _ui_log(f"[REDDIT] Fetching mentions for '{brand}' via Public JSON API...")
    
    results = []
    seen_ids = set()
    start_time = datetime.utcnow() - timedelta(days=days)
    time_filter = _get_time_filter(days)
    
    # Determine search URL
    if subreddit:
        base_url = REDDIT_SUBREDDIT_SEARCH_URL.format(subreddit=subreddit)
    else:
        base_url = REDDIT_SEARCH_URL
    
    # Build search queries
    queries = [f'"{brand}"']
    if competitor:
        queries.append(f'"{competitor}"')
    
    for query in queries:
        _ui_log(f"[REDDIT] Searching: {query}")
        
        after = None  # Pagination cursor
        fetched = 0
        consecutive_empty = 0
        
        while fetched < limit:
            params = {
                "q": query,
                "sort": "new",
                "t": time_filter,
                "limit": min(RESULTS_PER_PAGE, limit - fetched),
                "type": "link",  # Submissions only
                "raw_json": 1,
                "restrict_sr": "true" if subreddit else "false"
            }
            if after:
                params["after"] = after
            
            data = _make_request(base_url, params)
            
            if not data or "data" not in data:
                _ui_log(f"[REDDIT] No more results or error occurred")
                break
            
            posts = data["data"].get("children", [])
            
            if not posts:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
                continue
            
            consecutive_empty = 0
            new_items_this_page = 0
            
            for post_wrapper in posts:
                post = post_wrapper.get("data", {})
                post_id = post.get("id")
                
                # Skip duplicates
                if not post_id or post_id in seen_ids:
                    continue
                
                # Check date filter
                created_utc = post.get("created_utc", 0)
                post_date = datetime.utcfromtimestamp(created_utc)
                
                if post_date < start_time:
                    continue
                
                # Extract text content
                title = post.get("title", "")
                selftext = post.get("selftext", "")
                text = f"{title}\n{selftext}"
                
                # Match terms
                matched_terms = []
                if _word_match(brand, text):
                    matched_terms.append("brand")
                if competitor and _word_match(competitor, text):
                    matched_terms.append("competitor")
                
                # Skip if no keyword match (search can be fuzzy)
                if not matched_terms:
                    continue
                
                seen_ids.add(post_id)
                new_items_this_page += 1
                
                item = {
                    "platform": "reddit",
                    "title": title,
                    "text": text,
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "score": post.get("score", 0),
                    "comments": post.get("num_comments", 0),
                    "created_utc": created_utc,
                    "subreddit": post.get("subreddit", ""),
                    "matched_terms": matched_terms,
                    "id": post_id
                }
                results.append(item)
                fetched += 1
                
                # Progress logging
                if fetched % 50 == 0:
                    _ui_log(f"[REDDIT] Fetched {fetched} submissions...")
            
            # Check if we got any new items
            if new_items_this_page == 0:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
            
            # Pagination
            after = data["data"].get("after")
            if not after:
                break
            
            # Rate limiting delay
            time.sleep(REQUEST_DELAY)
    
    # Optional: Fetch comments
    if include_comments:
        remaining_limit = limit - len(results)
        if remaining_limit > 0:
            comment_results = _fetch_comments_public(
                brand=brand,
                competitor=competitor,
                days=days,
                limit=remaining_limit,
                seen_ids=seen_ids,
                start_time=start_time,
                subreddit=subreddit
            )
            results.extend(comment_results)
    
    _ui_log(f"[REDDIT] Finished. Total results: {len(results)}")
    return results


def _fetch_comments_public(
    brand: str,
    competitor: Optional[str],
    days: int,
    limit: int,
    seen_ids: set,
    start_time: datetime,
    subreddit: Optional[str] = None
) -> List[Dict]:
    """
    Fetch comments matching brand/competitor via Public JSON API.
    
    Args:
        brand: Primary keyword
        competitor: Optional competitor keyword
        days: Days to look back
        limit: Max results
        seen_ids: Set of already-seen IDs (modified in place)
        start_time: Datetime cutoff
        subreddit: Optional specific subreddit
        
    Returns:
        List of comment dicts
    """
    _ui_log(f"[REDDIT] Fetching comments...")
    
    results = []
    time_filter = _get_time_filter(days)
    
    # Determine search URL
    if subreddit:
        base_url = REDDIT_SUBREDDIT_SEARCH_URL.format(subreddit=subreddit)
    else:
        base_url = REDDIT_SEARCH_URL
    
    queries = [f'"{brand}"']
    if competitor:
        queries.append(f'"{competitor}"')
    
    fetched = 0
    
    for query in queries:
        if fetched >= limit:
            break
            
        params = {
            "q": query,
            "sort": "new",
            "t": time_filter,
            "limit": min(RESULTS_PER_PAGE, limit - fetched),
            "type": "comment",
            "raw_json": 1,
            "restrict_sr": "true" if subreddit else "false"
        }
        
        data = _make_request(base_url, params)
        
        if not data or "data" not in data:
            continue
        
        comments = data["data"].get("children", [])
        
        for comment_wrapper in comments:
            if fetched >= limit:
                break
                
            comment = comment_wrapper.get("data", {})
            comment_id = comment.get("id")
            
            if not comment_id or comment_id in seen_ids:
                continue
            
            created_utc = comment.get("created_utc", 0)
            comment_date = datetime.utcfromtimestamp(created_utc)
            
            if comment_date < start_time:
                continue
            
            body = comment.get("body", "")
            
            matched_terms = []
            if _word_match(brand, body):
                matched_terms.append("brand")
            if competitor and _word_match(competitor, body):
                matched_terms.append("competitor")
            
            if not matched_terms:
                continue
            
            seen_ids.add(comment_id)
            fetched += 1
            
            item = {
                "platform": "reddit",
                "title": f"Comment in r/{comment.get('subreddit', 'unknown')}",
                "text": body,
                "url": f"https://reddit.com{comment.get('permalink', '')}",
                "score": comment.get("score", 0),
                "comments": 0,
                "created_utc": created_utc,
                "subreddit": comment.get("subreddit", ""),
                "matched_terms": matched_terms,
                "id": comment_id,
                "type": "comment"
            }
            results.append(item)
        
        time.sleep(REQUEST_DELAY)
    
    _ui_log(f"[REDDIT] Fetched {len(results)} comments")
    return results


# =============================================================================
# Pullpush.io API - Fallback for Historical Data
# =============================================================================

def fetch_reddit_mentions_pullpush(
    brand: str,
    competitor: Optional[str] = None,
    days: int = 30,
    limit: int = 500,
    include_comments: bool = True
) -> List[Dict]:
    """
    Fallback: Fetch from Pullpush.io (Pushshift successor).
    Useful for historical data beyond Reddit's search limits.
    
    Note: Pullpush.io may have indexing delays (not real-time).
    
    Args:
        brand: Primary keyword
        competitor: Optional competitor keyword
        days: Days to look back
        limit: Max results
        include_comments: Whether to include comments
        
    Returns:
        List of mention dicts with same schema as fetch_reddit_mentions
    """
    _ui_log(f"[PULLPUSH] Fetching historical mentions for '{brand}'...")
    
    results = []
    seen_ids = set()
    
    # Calculate time range (Unix timestamps)
    end_epoch = int(datetime.utcnow().timestamp())
    start_epoch = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    
    # Build query (Pullpush uses | for OR)
    query = brand
    if competitor:
        query = f"{brand}|{competitor}"
    
    # Fetch submissions
    submission_results = _fetch_pullpush_submissions(
        query=query,
        brand=brand,
        competitor=competitor,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        limit=limit,
        seen_ids=seen_ids
    )
    results.extend(submission_results)
    
    # Fetch comments if requested
    if include_comments:
        remaining = limit - len(results)
        if remaining > 0:
            comment_results = _fetch_pullpush_comments(
                query=query,
                brand=brand,
                competitor=competitor,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                limit=remaining,
                seen_ids=seen_ids
            )
            results.extend(comment_results)
    
    _ui_log(f"[PULLPUSH] Finished. Total results: {len(results)}")
    return results


def _fetch_pullpush_submissions(
    query: str,
    brand: str,
    competitor: Optional[str],
    start_epoch: int,
    end_epoch: int,
    limit: int,
    seen_ids: set
) -> List[Dict]:
    """Fetch submissions from Pullpush.io API."""
    results = []
    
    params = {
        "q": query,
        "after": start_epoch,
        "before": end_epoch,
        "size": min(limit, 100),
        "sort": "desc",
        "sort_type": "created_utc"
    }
    
    try:
        response = requests.get(
            PULLPUSH_SUBMISSION_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=30
        )
        
        if response.status_code != 200:
            _ui_log(f"[PULLPUSH] Submissions API returned {response.status_code}")
            return results
        
        data = response.json().get("data", [])
        
        for post in data:
            post_id = post.get("id")
            
            if not post_id or post_id in seen_ids:
                continue
            
            title = post.get("title", "")
            selftext = post.get("selftext", "")
            text = f"{title}\n{selftext}"
            
            matched_terms = []
            if _word_match(brand, text):
                matched_terms.append("brand")
            if competitor and _word_match(competitor, text):
                matched_terms.append("competitor")
            
            if not matched_terms:
                continue
            
            seen_ids.add(post_id)
            
            subreddit_name = post.get("subreddit", "")
            permalink = post.get("permalink", f"/r/{subreddit_name}/comments/{post_id}")
            
            item = {
                "platform": "reddit",
                "title": title,
                "text": text,
                "url": f"https://reddit.com{permalink}",
                "score": post.get("score", 0),
                "comments": post.get("num_comments", 0),
                "created_utc": post.get("created_utc", 0),
                "subreddit": subreddit_name,
                "matched_terms": matched_terms,
                "id": post_id,
                "source": "pullpush"
            }
            results.append(item)
            
    except requests.exceptions.RequestException as e:
        _ui_log(f"[PULLPUSH] Submissions request error: {e}")
        logger.exception("Pullpush submissions error")
    except (ValueError, KeyError) as e:
        _ui_log(f"[PULLPUSH] Submissions parse error: {e}")
        logger.exception("Pullpush submissions parse error")
    
    return results


def _fetch_pullpush_comments(
    query: str,
    brand: str,
    competitor: Optional[str],
    start_epoch: int,
    end_epoch: int,
    limit: int,
    seen_ids: set
) -> List[Dict]:
    """Fetch comments from Pullpush.io API."""
    results = []
    
    params = {
        "q": query,
        "after": start_epoch,
        "before": end_epoch,
        "size": min(limit, 100),
        "sort": "desc",
        "sort_type": "created_utc"
    }
    
    try:
        response = requests.get(
            PULLPUSH_COMMENT_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=30
        )
        
        if response.status_code != 200:
            _ui_log(f"[PULLPUSH] Comments API returned {response.status_code}")
            return results
        
        data = response.json().get("data", [])
        
        for comment in data:
            comment_id = comment.get("id")
            
            if not comment_id or comment_id in seen_ids:
                continue
            
            body = comment.get("body", "")
            
            matched_terms = []
            if _word_match(brand, body):
                matched_terms.append("brand")
            if competitor and _word_match(competitor, body):
                matched_terms.append("competitor")
            
            if not matched_terms:
                continue
            
            seen_ids.add(comment_id)
            
            subreddit_name = comment.get("subreddit", "unknown")
            permalink = comment.get("permalink", "")
            
            item = {
                "platform": "reddit",
                "title": f"Comment in r/{subreddit_name}",
                "text": body,
                "url": f"https://reddit.com{permalink}" if permalink else "",
                "score": comment.get("score", 0),
                "comments": 0,
                "created_utc": comment.get("created_utc", 0),
                "subreddit": subreddit_name,
                "matched_terms": matched_terms,
                "id": comment_id,
                "type": "comment",
                "source": "pullpush"
            }
            results.append(item)
            
    except requests.exceptions.RequestException as e:
        _ui_log(f"[PULLPUSH] Comments request error: {e}")
        logger.exception("Pullpush comments error")
    except (ValueError, KeyError) as e:
        _ui_log(f"[PULLPUSH] Comments parse error: {e}")
        logger.exception("Pullpush comments parse error")
    
    return results


# =============================================================================
# Combined Fetcher (Best of Both APIs)
# =============================================================================

def fetch_reddit_mentions_combined(
    brand: str,
    competitor: Optional[str] = None,
    days: int = 14,
    limit: int = 500,
    include_comments: bool = True,
    use_pullpush_fallback: bool = True,
    subreddit: Optional[str] = None
) -> List[Dict]:
    """
    Combined fetcher: Public JSON API + Pullpush fallback.
    Automatically deduplicates results.
    
    Args:
        brand: Primary keyword to search
        competitor: Optional competitor keyword
        days: Number of days to look back
        limit: Maximum number of results
        include_comments: Whether to also fetch comments
        use_pullpush_fallback: Whether to use Pullpush for additional data
        subreddit: Optional specific subreddit (None = all)
        
    Returns:
        Deduplicated, sorted list of mention dicts
    """
    _ui_log(f"[REDDIT] Combined fetch for '{brand}'...")
    
    # Primary: Public JSON API (more real-time)
    results = fetch_reddit_mentions(
        brand=brand,
        competitor=competitor,
        days=days,
        limit=limit,
        include_comments=include_comments,
        subreddit=subreddit
    )
    
    seen_ids = {r["id"] for r in results}
    initial_count = len(results)
    
    # Fallback: Pullpush for additional historical data
    if use_pullpush_fallback and len(results) < limit:
        remaining = limit - len(results)
        _ui_log(f"[REDDIT] Using Pullpush fallback for {remaining} more results...")
        
        pullpush_results = fetch_reddit_mentions_pullpush(
            brand=brand,
            competitor=competitor,
            days=days,
            limit=remaining,
            include_comments=include_comments
        )
        
        # Deduplicate and merge
        for item in pullpush_results:
            if item["id"] not in seen_ids:
                results.append(item)
                seen_ids.add(item["id"])
        
        _ui_log(f"[REDDIT] Added {len(results) - initial_count} from Pullpush")
    
    # Sort by created_utc descending (newest first)
    results.sort(key=lambda x: x.get("created_utc", 0), reverse=True)
    
    # Enforce limit
    results = results[:limit]
    
    _ui_log(f"[REDDIT] Combined total: {len(results)}")
    return results


# =============================================================================
# Utility: Subreddit-specific Search
# =============================================================================

def fetch_subreddit_mentions(
    subreddit: str,
    brand: str,
    competitor: Optional[str] = None,
    days: int = 14,
    limit: int = 100,
    include_comments: bool = True
) -> List[Dict]:
    """
    Convenience function to search within a specific subreddit.
    
    Args:
        subreddit: Subreddit name (without r/)
        brand: Primary keyword
        competitor: Optional competitor keyword
        days: Days to look back
        limit: Max results
        include_comments: Include comments
        
    Returns:
        List of mention dicts
    """
    return fetch_reddit_mentions(
        brand=brand,
        competitor=competitor,
        days=days,
        limit=limit,
        include_comments=include_comments,
        subreddit=subreddit
    )


# =============================================================================
# Health Check
# =============================================================================

def check_api_health() -> Dict[str, bool]:
    """
    Check if Reddit and Pullpush APIs are accessible.
    
    Returns:
        Dict with health status for each API
    """
    health = {
        "reddit_public_json": False,
        "pullpush_submissions": False,
        "pullpush_comments": False
    }
    
    # Check Reddit Public JSON
    try:
        response = requests.get(
            "https://www.reddit.com/r/all/new.json",
            params={"limit": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=10
        )
        health["reddit_public_json"] = response.status_code == 200
    except Exception:
        pass
    
    # Check Pullpush Submissions
    try:
        response = requests.get(
            PULLPUSH_SUBMISSION_URL,
            params={"size": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=10
        )
        health["pullpush_submissions"] = response.status_code == 200
    except Exception:
        pass
    
    # Check Pullpush Comments
    try:
        response = requests.get(
            PULLPUSH_COMMENT_URL,
            params={"size": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=10
        )
        health["pullpush_comments"] = response.status_code == 200
    except Exception:
        pass
    
    return health