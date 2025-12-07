import os
import logging
from typing import List, Dict

logger = logging.getLogger("sentiment")

USE_HF = os.getenv("USE_HF", "true").lower() in ("1", "true", "yes")
HF_MODEL = os.getenv("HF_SENTIMENT_MODEL", "textattack/bert-base-uncased-SST-2")

nlp_model = None
tokenizer = None
device = "cpu"

if USE_HF:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        nlp_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
        nlp_model.to(device)
        nlp_model.eval()
        logger.info("Loaded HF model %s on %s", HF_MODEL, device)
    except Exception as e:
        logger.warning("Failed to initialize HF model %s: %s", HF_MODEL, e)
        nlp_model = None
        tokenizer = None

def _softmax(logits):
    import math
    exps = [math.exp(x) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

def analyze_sentiments(texts: List[str], batch_size: int = 16) -> List[Dict]:
    results = []
    if not texts:
        return results

    if nlp_model is not None and tokenizer is not None:
        try:
            import torch
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                with torch.no_grad():
                    out = nlp_model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = out.logits.cpu().tolist()
                label_map = getattr(nlp_model.config, "id2label", None) or {}
                for logit in logits:
                    probs = _softmax(logit)
                    top_idx = int(max(range(len(probs)), key=lambda j: probs[j]))
                    label = label_map.get(top_idx, str(top_idx))
                    score = float(probs[top_idx])
                    results.append({"label": label, "score": score})
            return results
        except Exception as e:
            logger.warning("HF inference failed: %s", e)

    # Basic heuristic fallback
    pos_words = ("good", "great", "love", "liked", "awesome", "excellent", "happy")
    neg_words = ("bad", "hate", "terrible", "worst", "awful", "angry", "disappointed")
    for t in texts:
        tl = (t or "").lower()
        pos = sum(1 for w in pos_words if w in tl)
        neg = sum(1 for w in neg_words if w in tl)
        if pos > neg:
            results.append({"label": "POSITIVE", "score": float(pos - neg)})
        elif neg > pos:
            results.append({"label": "NEGATIVE", "score": float(neg - pos)})
        else:
            results.append({"label": "NEUTRAL", "score": 0.0})
    return results