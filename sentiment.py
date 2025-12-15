import os
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger("sentiment")

USE_HF = os.getenv("USE_HF", "true").lower() in ("1", "true", "yes")
HF_MODEL = os.getenv("HF_SENTIMENT_MODEL", "textattack/bert-base-uncased-SST-2")
# If the HF model predicts POS/NEG only, treat predictions with max prob < this as NEUTRAL
HF_NEUTRAL_PROB_THRESHOLD = float(os.getenv("HF_NEUTRAL_PROB_THRESHOLD", "0.60"))

nlp_model = None
tokenizer = None
device = "cpu"
_use_pipeline = False
_pipeline = None

# Try to initialize HuggingFace model / pipeline if requested
if USE_HF:
    try:
        # Prefer the transformers pipeline for simplicity if available
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification  # type: ignore

        try:
            _pipeline = pipeline("sentiment-analysis", model=HF_MODEL)
            _use_pipeline = True
            logger.info("Loaded HF sentiment pipeline model %s", HF_MODEL)
        except Exception:
            # fallback to manual model+tokenizer usage
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
            nlp_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
            # move model to CPU/CUDA if torch is available
            try:
                import torch  # type: ignore

                device = "cuda" if torch.cuda.is_available() else "cpu"
                nlp_model.to(device)
            except Exception:
                device = "cpu"
            logger.info("Loaded HF model %s on %s", HF_MODEL, device)
    except Exception as e:
        logger.warning("Failed to initialize HF model/pipeline %s: %s", HF_MODEL, e)
        nlp_model = None
        tokenizer = None
        _pipeline = None
        _use_pipeline = False

# VADER fallback
_vader = None
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore

    _vader = SentimentIntensityAnalyzer()
    logger.info("VADER sentiment available as fallback")
except Exception:
    _vader = None
    logger.debug("VADER not available; install nltk and vader_lexicon for fallback behavior")

def _softmax(logits: List[float]) -> List[float]:
    import math

    exps = [math.exp(x) for x in logits]
    s = sum(exps) if sum(exps) != 0 else 1.0
    return [e / s for e in exps]

def _normalize_label(label: str) -> str:
    lab = (label or "").strip().upper()
    if lab.startswith("POS"):
        return "POSITIVE"
    if lab.startswith("NEG"):
        return "NEGATIVE"
    if lab.startswith("NEU"):
        return "NEUTRAL"
    
    # Handle standard HuggingFace label formats for SST-2
    if lab == "LABEL_1":
        return "POSITIVE"
    if lab == "LABEL_0":
        return "NEGATIVE"
        
    # Some models use 0/1 mapping; handle generically
    if lab in ("0", "1"):
        # simple heuristic for binary classification often 0=neg, 1=pos
        if lab == "1":
            return "POSITIVE"
        if lab == "0":
            return "NEGATIVE"
        return lab
    return lab

def analyze_sentiments(texts: List[str], batch_size: int = 16) -> List[Dict]:
    """
    Analyze a list of texts and return a list of dicts: {"label": <POSITIVE|NEGATIVE|NEUTRAL>, "score": <0..1>}
    - Attempts to use HF pipeline / model when available
    - Falls back to VADER if HF not available
    - Last-resort heuristic keyword matcher if neither HF nor VADER are available
    """
    results: List[Dict] = []
    if not texts:
        return results

    # 1) HuggingFace pipeline (simple, handles batching internally)
    if _use_pipeline and _pipeline is not None:
        try:
            raw_out = _pipeline(texts, truncation=True)
            for o in raw_out:
                label = _normalize_label(o.get("label", "NEUTRAL"))
                score = float(o.get("score", 0.0))
                # if model only returns POS/NEG, allow neutral when confidence is low
                if label in ("POSITIVE", "NEGATIVE") and score < HF_NEUTRAL_PROB_THRESHOLD:
                    results.append({"label": "NEUTRAL", "score": score})
                else:
                    results.append({"label": label, "score": score})
            return results
        except Exception as e:
            logger.warning("HF pipeline inference failed: %s", e)

    # 2) HuggingFace model + tokenizer (manual batching & softmax)
    if nlp_model is not None and tokenizer is not None:
        try:
            import torch  # type: ignore

            model = nlp_model
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits_batch = out.logits.cpu().tolist()
                # map id->label if available
                label_map = getattr(model.config, "id2label", None) or {}
                for logits in logits_batch:
                    probs = _softmax(logits)
                    top_idx = int(max(range(len(probs)), key=lambda j: probs[j]))
                    raw_label = label_map.get(top_idx, str(top_idx))
                    label = _normalize_label(raw_label)
                    score = float(probs[top_idx]) if probs else 0.0
                    if label in ("POSITIVE", "NEGATIVE") and score < HF_NEUTRAL_PROB_THRESHOLD:
                        results.append({"label": "NEUTRAL", "score": score})
                    else:
                        results.append({"label": label, "score": score})
            return results
        except Exception as e:
            logger.warning("HF model inference failed: %s", e)

    # 3) VADER fallback (rule-based, good for social/short-text English)
    if _vader is not None:
        try:
            for t in texts:
                scores = _vader.polarity_scores(t or "")
                comp = scores.get("compound", 0.0)
                # thresholds tuned for social text; adjust via environment if needed
                pos_thr = float(os.getenv("VADER_POS_THRESHOLD", "0.40"))
                neg_thr = float(os.getenv("VADER_NEG_THRESHOLD", "-0.40"))
                if comp >= pos_thr:
                    results.append({"label": "POSITIVE", "score": float(comp)})
                elif comp <= neg_thr:
                    results.append({"label": "NEGATIVE", "score": float(abs(comp))})
                else:
                    results.append({"label": "NEUTRAL", "score": float(abs(comp))})
            return results
        except Exception as e:
            logger.warning("VADER inference failed: %s", e)

    # 4) Very simple heuristic fallback (keywords). Not ideal, but guaranteed to run.
    pos_words = ("good", "great", "love", "liked", "awesome", "excellent", "happy", "amazing")
    neg_words = ("bad", "hate", "terrible", "worst", "awful", "angry", "disappointed", "disappointing")
    for t in texts:
        tl = (t or "").lower()
        pos = sum(1 for w in pos_words if w in tl)
        neg = sum(1 for w in neg_words if w in tl)
        if pos > neg:
            # normalize score to [0,1] (simple scaling)
            score = min(1.0, float(pos - neg) / max(1, len(pos_words)))
            results.append({"label": "POSITIVE", "score": score})
        elif neg > pos:
            score = min(1.0, float(neg - pos) / max(1, len(neg_words)))
            results.append({"label": "NEGATIVE", "score": score})
        else:
            results.append({"label": "NEUTRAL", "score": 0.0})
    return results

def analyze_sentiment(text: str) -> Dict:
    """
    Convenience wrapper for single text -> dict {"label":..., "score":...}
    """
    out = analyze_sentiments([text], batch_size=1)
    return out[0] if out else {"label": "NEUTRAL", "score": 0.0}


def predict(texts):
    """
    Wrapper to handle both single text and list of texts.
    Returns list of label strings for compatibility with streamlit_app.py.
    
    This is the primary function used by data_connectors.find_sentiment_fn()
    """
    # Handle single string input
    if isinstance(texts, str):
        result = analyze_sentiments([texts])
        return [r.get("label", "NEUTRAL") for r in result]
    
    # Handle list of strings
    if isinstance(texts, list):
        # Filter out any non-string items and ensure we have clean strings
        clean_texts = []
        for t in texts:
            if isinstance(t, str):
                clean_texts.append(t)
            elif t is not None:
                clean_texts.append(str(t))
            else:
                clean_texts.append("")
        
        result = analyze_sentiments(clean_texts)
        return [r.get("label", "NEUTRAL") for r in result]
    
    # Fallback for unexpected input types
    return ["NEUTRAL"]