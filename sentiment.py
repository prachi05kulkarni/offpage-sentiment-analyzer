"""
Updated sentiment helpers with preprocessing hook (normalize_text in utils).
Uses HuggingFace pipeline if USE_HF True; falls back to VADER if available.
"""
import os
import logging
from typing import List, Dict, Optional
logger = logging.getLogger("sentiment")

USE_HF = os.getenv("USE_HF", "true").lower() in ("1", "true", "yes")
HF_MODEL = os.getenv("HF_SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment")
HF_NEUTRAL_PROB_THRESHOLD = float(os.getenv("HF_NEUTRAL_PROB_THRESHOLD", "0.60"))

nlp_model = None
tokenizer = None
device = "cpu"
_use_pipeline = False
_pipeline = None

# Try to import utils for preprocessing
try:
    from utils import normalize_text
except Exception:
    def normalize_text(x): return x

# Initialize HF if requested
if USE_HF:
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
        try:
            _pipeline = pipeline("sentiment-analysis", model=HF_MODEL)
            _use_pipeline = True
            logger.info("Loaded HF pipeline model %s", HF_MODEL)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
            nlp_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
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
    if lab in ("0", "1"):
        return lab
    return lab


def analyze_sentiments(texts: List[str], batch_size: int = 16) -> List[Dict]:
    """
    Analyze a list of texts and return list of {"label": <POSITIVE|NEGATIVE|NEUTRAL>, "score": <0..1>}
    - Preprocess each text with utils.normalize_text
    - Use HF pipeline if available, fallback to HF model manual or VADER or a keyword heuristic
    """
    results: List[Dict] = []
    if not texts:
        return results

    # preprocess
    proc_texts = [normalize_text(t or "") for t in texts]

    # 1) HF pipeline
    if _use_pipeline and _pipeline is not None:
        try:
            raw_out = _pipeline(proc_texts, truncation=True)
            for o in raw_out:
                label = _normalize_label(o.get("label", "NEUTRAL"))
                score = float(o.get("score", 0.0))
                if label in ("POSITIVE", "NEGATIVE") and score < HF_NEUTRAL_PROB_THRESHOLD:
                    results.append({"label": "NEUTRAL", "score": score})
                else:
                    results.append({"label": label, "score": score})
            return results
        except Exception as e:
            logger.warning("HF pipeline inference failed: %s", e)

    # 2) HF manual model + tokenizer
    if nlp_model is not None and tokenizer is not None:
        try:
            import torch  # type: ignore
            model = nlp_model
            for i in range(0, len(proc_texts), batch_size):
                batch = proc_texts[i: i + batch_size]
                enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits_batch = out.logits.cpu().tolist()
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
            logger.warning("HF manual inference failed: %s", e)

    # 3) VADER fallback
    if _vader is not None:
        try:
            for t in proc_texts:
                vs = _vader.polarity_scores(t)
                # compound ranges [-1,1]
                c = vs.get("compound", 0.0)
                if c >= 0.05:
                    results.append({"label": "POSITIVE", "score": float(c)})
                elif c <= -0.05:
                    results.append({"label": "NEGATIVE", "score": abs(float(c))})
                else:
                    results.append({"label": "NEUTRAL", "score": float(abs(c))})
            return results
        except Exception as e:
            logger.warning("VADER fallback failed: %s", e)

    # 4) Simple lexicon heuristic as last resort
    POS = {"good","great","love","excellent","best","amazing","happy","like","awesome","positive","win","improve","recommend"}
    NEG = {"bad","terrible","hate","awful","worst","angry","disappointed","poor","negative","problem","issue","complain","complaint","scam"}
    for t in proc_texts:
        lower = (t or "").lower()
        p = sum(1 for w in POS if w in lower)
        n = sum(1 for w in NEG if w in lower)
        if p > n:
            results.append({"label": "POSITIVE", "score": float(min(1.0, p/(p+n if p+n>0 else 1)))})
        elif n > p:
            results.append({"label": "NEGATIVE", "score": float(min(1.0, n/(p+n if p+n>0 else 1)))})
        else:
            results.append({"label": "NEUTRAL", "score": 0.0})
    return results