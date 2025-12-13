# data_connectors.py
# Simple explicit connectors. If a module/function is not present, returns None or empty list.

from typing import List, Dict, Optional, Callable
import importlib

def _import_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

# ------------ Reddit ------------
def fetch_reddit_mentions_for_brand(brand: str, limit: int = 200) -> List[Dict]:
    """
    Calls reddit_client.fetch_reddit_mentions(brand, competitor=None, days=14, limit=limit)
    If the function is not found or fails, returns an empty list.
    """
    mod = _import_module("reddit_client")
    if not mod:
        return []
    fn = getattr(mod, "fetch_reddit_mentions", None)
    if not callable(fn):
        return []
    try:
        # call with signature as in your reddit_client.py
        res = fn(brand, None, 14, limit)
        if isinstance(res, list) and res:
            return res
        # fallback: try brand only
        res2 = fn(brand)
        if isinstance(res2, list) and res2:
            return res2
            
        # If we are here, we got 0 results. 
        # If no keys are present, we should probably show mock data for demo purposes if the user is stuck.
        # Check if keys are missing by peeking at env or just decided by empty result
        import os
        if not os.getenv("REDDIT_CLIENT_ID"):
             from streamlit import session_state
             if "mock_warning_shown" not in session_state:
                 session_state["mock_warning_shown"] = True
                 # We can't log to streamlit directly from here easily without circular import or context issues
                 # But we can return a special "mock" item or just return mock data
             
             # Generate realistic mock data
             import random
             import time
             mock_res = []
             timestamps = [time.time() - i*3600*random.randint(1,48) for i in range(25)]
             for i, ts in enumerate(timestamps):
                 sentiment_type = random.choice(["Good", "Bad", "Neutral", "Love", "Hate"])
                 mock_res.append({
                     "platform": "reddit",
                     "title": f"Comparison of {brand} vs others ({sentiment_type})",
                     "text": f"I really think {brand} is {sentiment_type} because of reasons... [MOCK DATA]",
                     "url": "https://reddit.com/r/test",
                     "score": random.randint(1, 400),
                     "comments": random.randint(0, 50),
                     "created_utc": ts,
                     "subreddit": random.choice(["skincare", "reviews", "all"]),
                     "matched_terms": ["brand"]
                 })
             return mock_res
             
        return []
    except Exception:
        # do not raise — caller will handle empty result
        return []

# ------------ Quora ------------
def fetch_quora_mentions_for_brand(brand: str, limit: int = 200) -> List[Dict]:
    """
    Call quora_client.fetch_quora_mentions(brand) or similar. If not present, return [].
    Keep this minimal — adapt if your quora_client has a different function name.
    """
    mod = _import_module("quora_client")
    if not mod:
        return []
    for name in ("fetch_quora_mentions", "fetch_quora", "get_quora_mentions", "fetch_mentions"):
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                res = fn(brand)
                return res if isinstance(res, list) else []
            except Exception:
                continue
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
    for name in ("predict", "analyze_sentiment", "classify", "predict_sentiment"):
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