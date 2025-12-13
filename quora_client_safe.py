import os
import time
import logging
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import urllib.robotparser
from typing import List, Dict, Optional

logger = logging.getLogger("quora_client_safe")
USER_AGENT = os.getenv("REDDIT_USER_AGENT", "offpage-sentiment-analyzer/0.1 (by yourname)")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  #SERPAPI_KEY =b2def6f3eedbf41664b82db9c9a95cb20dae5a8455b4a37f287313a26abb0d80
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX") 
BING_API_KEY = os.getenv("BING_API_KEY") 
POLITE_DELAY = float(os.getenv("QUORA_FETCH_DELAY", "1.0"))  # seconds between page fetches

def _can_fetch_url(url: str, user_agent: str = USER_AGENT) -> bool:
    """
    Check robots.txt for the target host before fetching. Returns True if allowed.
    """
    try:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = urljoin(base, "/robots.txt")
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        can = rp.can_fetch(user_agent, url)
        logger.debug("robots check for %s -> %s", url, can)
        return can
    except Exception as e:
        logger.debug("robots.txt check failed for %s: %s", url, e)
        # If robots can't be read, be conservative and return False
        return False

def polite_fetch_page(url: str, timeout: int = 10) -> str:
    """
    Fetch page HTML with polite headers & delay. Respect robots.txt.
    Returns text content (or empty string on failure).
    """
    try:
        if not _can_fetch_url(url):
            logger.info("Blocked by robots.txt: %s", url)
            return ""
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        time.sleep(POLITE_DELAY)
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        return " ".join(paragraphs)
    except Exception as e:
        logger.debug("Failed to fetch page %s: %s", url, e)
        return ""

def fetch_quora_via_serpapi(brand: str, num: int = 10) -> List[Dict]:
    """
    Discover Quora pages via SerpAPI (recommended if you have a key).
    Returns list of {title, url} for Quora pages.
    """
    if not SERPAPI_KEY:
        logger.info("SERPAPI_KEY not set; SerpAPI disabled")
        return []
    try:
        params = {
            "q": f'site:quora.com "{brand}"',
            "engine": "google",
            "api_key": SERPAPI_KEY,
            "num": num
        }
        r = requests.get("https://serpapi.com/search", params=params, timeout=15, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        data = r.json()
        out = []
        for item in data.get("organic_results", []):
            url = item.get("link") or item.get("url")
            title = item.get("title") or item.get("snippet") or ""
            out.append({"title": title, "url": url})
        return out
    except Exception as e:
        logger.error("SerpAPI search failed: %s", e)
        return []

def fetch_quora_via_google(brand: str, num: int = 10) -> List[Dict]:
    """
    Discover Quora pages via Google Custom Search JSON API.
    Requires GOOGLE_API_KEY and GOOGLE_CX (search engine id).
    """
    if not (GOOGLE_API_KEY and GOOGLE_CX):
        logger.info("Google CSE credentials not set; skipping")
        return []
    try:
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CX,
            "q": f'site:quora.com "{brand}"',
            "num": min(10, num)
        }
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=15, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        data = r.json()
        out = []
        for item in data.get("items", []):
            out.append({"title": item.get("title"), "url": item.get("link")})
        return out
    except Exception as e:
        logger.error("Google CSE search failed: %s", e)
        return []

def fetch_quora_via_bing(brand: str, num: int = 10) -> List[Dict]:
    """
    Discover Quora pages via Bing Web Search API (Azure).
    Requires BING_API_KEY in env.
    """
    if not BING_API_KEY:
        logger.info("Bing API key not set; skipping")
        return []
    try:
        headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY, "User-Agent": USER_AGENT}
        params = {"q": f'site:quora.com "{brand}"', "count": min(10, num)}
        r = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        out = []
        for it in data.get("webPages", {}).get("value", []):
            out.append({"title": it.get("name"), "url": it.get("url")})
        return out
    except Exception as e:
        logger.error("Bing search failed: %s", e)
        return []

def fetch_quora_mentions(brand: str, method_preference: Optional[List[str]] = None, fetch_pages: bool = False, num: int = 10) -> List[Dict]:
    """
    High-level function to discover Quora pages mentioning `brand`.
    - method_preference: list containing any of ["serpapi","google","bing","publicsearch"]
      If None, will try SerpAPI -> Google -> Bing -> public search fallback.
    - fetch_pages: if True, attempts to politely fetch page content (respecting robots.txt).
    - Returns list of dicts: {platform:'quora', title, url, text}
      If fetch_pages=False, text will be empty string and only title+url returned.
    """
    methods = method_preference or ["serpapi", "google", "bing"]
    results = []
    for m in methods:
        if m == "serpapi":
            res = fetch_quora_via_serpapi(brand, num=num)
        elif m == "google":
            res = fetch_quora_via_google(brand, num=num)
        elif m == "bing":
            res = fetch_quora_via_bing(brand, num=num)
        else:
            res = []
        if res:
            results = res
            break

    # fallback: minimal public web search using reddit-like approach (not ideal)
    if not results:
        logger.info("No search results found from APIs; returning empty list")
        return []

    out = []
    for r in results:
        url = r.get("url")
        title = r.get("title") or ""
        text = ""
        if fetch_pages and url:
            text = polite_fetch_page(url)
        out.append({"platform": "quora", "title": title, "text": text, "url": url})
    return out






    