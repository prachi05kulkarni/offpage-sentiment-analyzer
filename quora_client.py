import os
import time
import logging
import requests
from bs4 import BeautifulSoup
import re

logger = logging.getLogger("quora_client")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"

def _retry(fn, max_attempts=3, base_delay=1.0):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            wait = base_delay * (2 ** (attempt - 1))
            logger.warning("Attempt %s failed with %s; retrying in %.1fs", attempt, e, wait)
            time.sleep(wait)
    raise RuntimeError("All attempts failed")

def _fetch_page_text(url, timeout=10):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; offpage-sentiment/1.0)"}
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    return " ".join(paragraphs)

def _word_match(term, text):
    if not term:
        return False
    try:
        return re.search(rf"\b{re.escape(term)}\b", (text or "").lower(), flags=re.IGNORECASE) is not None
    except Exception:
        return term.lower() in (text or "").lower()

def fetch_quora_mentions(brand, competitor=None, days=14, num=20):
    """
    Uses SerpAPI to search Quora pages for brand (and optional competitor).
    Returns list of dicts with keys: platform, title, text, url, matched_terms
    """
    if not SERPAPI_KEY:
        logger.info("SERPAPI_KEY not set; Quora fetch will return [].")
        return []

    def _do_search():
        # Build query to include competitor if provided
        if competitor:
            q = f'site:quora.com ("{brand}" OR "{competitor}")'
        else:
            q = f'site:quora.com "{brand}"'

        params = {
            "q": q,
            "engine": "google",
            "api_key": SERPAPI_KEY,
            "num": num
        }
        r = requests.get(SERPAPI_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("organic_results", []):
            try:
                url = item.get("link") or item.get("url")
                title = item.get("title") or ""
                text = ""
                if url:
                    try:
                        text = _fetch_page_text(url, timeout=10)[:4000]
                        time.sleep(1.0)
                    except Exception as ex:
                        logger.debug("Could not fetch Quora page %s: %s", url, ex)
                        text = ""
                # matched_terms via word-boundary matching
                matched_terms = []
                if _word_match(brand, text) or _word_match(brand, title):
                    matched_terms.append(brand)
                if competitor and (_word_match(competitor, text) or _word_match(competitor, title)):
                    matched_terms.append(competitor)

                results.append({
                    "platform": "quora",
                    "title": title,
                    "text": text,
                    "url": url,
                    "matched_terms": matched_terms
                })
            except Exception as ex:
                logger.debug("Skipping organic result due to error: %s", ex)
        return results

    try:
        return _retry(_do_search, max_attempts=3, base_delay=1.0)
    except Exception as e:
        logger.error("Quora fetch failed: %s", e)
        return []