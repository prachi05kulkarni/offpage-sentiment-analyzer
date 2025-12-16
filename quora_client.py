"""
Quora Client - Uses SerpAPI with pagination to work around free tier 10-result limit
"""
import os
import time
import logging
import requests
from bs4 import BeautifulSoup
import re

logger = logging.getLogger("quora_client")

# Strip whitespace from env var to fix common .env formatting issues
SERPAPI_KEY = (os.getenv("SERPAPI_KEY") or "").strip()
SERPAPI_URL = "https://serpapi.com/search"

# SerpAPI free tier limitation
MAX_RESULTS_PER_PAGE = 10  # Free tier caps at 10 per request


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


def _fetch_page_text(url, timeout=10):
    """Fetch text content from a Quora page."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; offpage-sentiment/1.0)"}
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    return " ".join(paragraphs)


def _word_match(term, text):
    """Check if term appears in text with word boundaries."""
    if not term:
        return False
    try:
        return re.search(rf"\b{re.escape(term)}\b", (text or "").lower(), flags=re.IGNORECASE) is not None
    except Exception:
        return term.lower() in (text or "").lower()


def fetch_quora_mentions(brand, competitor=None, days=14, num=20):
    """
    Uses SerpAPI to search Quora pages for brand.
    
    Due to SerpAPI free tier limit (10 results per request), this function
    uses pagination to fetch more results when num > 10.
    """
    if not SERPAPI_KEY:
        _ui_log("[QUORA] SERPAPI_KEY not set or empty - skipping")
        logger.info("SERPAPI_KEY not set; Quora fetch will return [].")
        return []

    # Build query
    if competitor:
        q = f'site:quora.com ("{brand}" OR "{competitor}")'
    else:
        q = f'site:quora.com "{brand}"'

    _ui_log(f"[QUORA] Searching for: '{brand}'")
    _ui_log(f"[QUORA] Query: {q}")
    _ui_log(f"[QUORA] Requesting {num} results (will paginate if needed)")

    all_results = []
    seen_urls = set()
    start = 0
    pages_fetched = 0
    max_pages = (num // MAX_RESULTS_PER_PAGE) + 1  # Calculate how many pages we need
    max_pages = min(max_pages, 7)  # Cap at 5 pages to avoid too many API calls

    while len(all_results) < num and pages_fetched < max_pages:
        try:
            params = {
                "q": q,
                "engine": "google",
                "api_key": SERPAPI_KEY,
                "num": 10,  # Request 10 per page (free tier max)
                "start": start  # Pagination offset
            }

            _ui_log(f"[QUORA] Fetching page {pages_fetched + 1} (start={start})...")
            
            r = requests.get(SERPAPI_URL, params=params, timeout=15)
            
            if r.status_code != 200:
                _ui_log(f"[QUORA] API Error: {r.status_code}")
                break

            data = r.json()
            
            if "error" in data:
                _ui_log(f"[QUORA] SerpAPI ERROR: {data.get('error')}")
                break

            organic_results = data.get("organic_results", [])
            _ui_log(f"[QUORA] Page {pages_fetched + 1}: got {len(organic_results)} results")

            if not organic_results:
                break  # No more results

            # Process results
            for item in organic_results:
                if len(all_results) >= num:
                    break
                    
                try:
                    url = item.get("link") or item.get("url")
                    
                    # Skip duplicates
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    
                    title = item.get("title") or ""
                    snippet = item.get("snippet") or ""

                    # Try to fetch full page text (with fallback to snippet)
                    text = ""
                    if url:
                        try:
                            text = _fetch_page_text(url, timeout=10)[:4000]
                            time.sleep(0.5)  # Be nice to Quora
                        except Exception:
                            text = snippet  # Fallback to snippet

                    if not text and snippet:
                        text = snippet

                    # Matched terms
                    matched_terms = []
                    if _word_match(brand, text) or _word_match(brand, title):
                        matched_terms.append(brand)
                    if competitor and (_word_match(competitor, text) or _word_match(competitor, title)):
                        matched_terms.append(competitor)

                    all_results.append({
                        "platform": "quora",
                        "title": title,
                        "text": text,
                        "url": url,
                        "matched_terms": matched_terms
                    })

                except Exception as ex:
                    logger.debug("Skipping result: %s", ex)
                    continue

            # Move to next page
            start += 10
            pages_fetched += 1
            time.sleep(1.0)  # Rate limiting between pages

        except Exception as e:
            _ui_log(f"[QUORA] Error during fetch: {e}")
            break

    _ui_log(f"[QUORA] Total results: {len(all_results)} (from {pages_fetched} pages)")
    return all_results