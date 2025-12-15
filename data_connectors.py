# data_connectors.py
# Simple explicit connectors. If a module/function is not present, returns None or empty list.

from typing import List, Dict, Optional, Callable
import importlib
import os

# Helper to log to Streamlit UI and Console
def _ui_log(msg: str):
    """Log to Streamlit session_state and Console."""
    # Print to console for debugging
    print(msg)
    
    try:
        from streamlit import session_state
        if "logs" not in session_state:
            session_state["logs"] = []
        session_state["logs"].append(msg)
        session_state["logs"] = session_state["logs"][-300:]
    except Exception:
        pass  # Fallback silently if not in Streamlit context

def _import_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

# ------------ Reddit ------------
def fetch_reddit_mentions_for_brand(brand: str, limit: int = 200) -> List[Dict]:
    """
    Calls reddit_client.fetch_reddit_mentions(brand, competitor=None, days=14, limit=limit)
    Uses Public JSON method - no API keys required.
    """
    _ui_log(f"[DATA] Fetching Reddit mentions for '{brand}'...")
    
    mod = _import_module("reddit_client")
    if not mod:
        _ui_log("[DATA] reddit_client module not found!")
        return []
    
    fn = getattr(mod, "fetch_reddit_mentions", None)
    if not callable(fn):
        _ui_log("[DATA] fetch_reddit_mentions function not found!")
        return []
    
    try:
        # Call with full signature
        res = fn(brand, None, 14, limit)
        if isinstance(res, list):
            _ui_log(f"[DATA] Reddit returned {len(res)} results")
            return res
        else:
            _ui_log(f"[DATA] Reddit returned unexpected type: {type(res)}")
            return []
    except Exception as e:
        _ui_log(f"[DATA] Reddit error: {str(e)}")
        return []

# ------------ Quora ------------
def fetch_quora_mentions_for_brand(brand: str, limit: int = 200) -> List[Dict]:
    """
    Call quora_client.fetch_quora_mentions(brand) or similar. If not present, return [].
    """
    # Check SERPAPI_KEY first
    serpapi_key = (os.getenv("SERPAPI_KEY") or "").strip()
    if not serpapi_key:
        _ui_log("[QUORA] SERPAPI_KEY not set - skipping Quora")
        return []
    
    _ui_log(f"[QUORA] Fetching mentions for '{brand}'...")
    
    mod = _import_module("quora_client")
    if not mod:
        _ui_log("[QUORA] quora_client module not found")
        return []
    
    for name in ("fetch_quora_mentions", "fetch_quora", "get_quora_mentions", "fetch_mentions"):
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                # Pass limit to Quora (SerpAPI caps at ~100)
                quora_limit = min(limit, 100)
                _ui_log(f"[QUORA] Calling {name}('{brand}', num={quora_limit})")
                res = fn(brand, num=quora_limit)
                if isinstance(res, list):
                    _ui_log(f"[QUORA] Found {len(res)} results for '{brand}'")
                    return res
                else:
                    _ui_log(f"[QUORA] Function returned non-list: {type(res)}")
                    return []
            except Exception as e:
                _ui_log(f"[QUORA] Error in {name}(): {str(e)}")
                continue
    
    _ui_log("[QUORA] No suitable function found in quora_client")
    return []

# ------------ Sentiment ------------
def find_sentiment_fn() -> Optional[Callable]:
    """
    Try to import sentiment.py and return a callable that accepts either:
    - a single text -> label
    - a list of texts -> list of labels
    Preferred function names: predict, analyze_sentiment, classify
    """
    mod = _import_module("sentiment")
    if not mod:
        return None
    for name in ("predict", "analyze_sentiments", "analyze_sentiment", "classify", "predict_sentiment"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    # fallback: any callable with 'sent' in name
    for attr in dir(mod):
        if "sent" in attr.lower():
            fn = getattr(mod, attr)
            if callable(fn):
                return fn
    return None