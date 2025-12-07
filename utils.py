import os
import json
import csv
import logging
from typing import List, Dict

logger = logging.getLogger("utils")

def export_jsonl_to_csv(jsonl_path: str, csv_path: str):
    fieldnames = ["platform","type","id","title","text","url","created_utc","subreddit"]
    with open(jsonl_path, 'r', encoding='utf-8') as inf, open(csv_path, 'w', newline='', encoding='utf-8') as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        for line in inf:
            try:
                obj = json.loads(line)
                writer.writerow({k: obj.get(k, "") for k in fieldnames})
            except Exception:
                continue
    logger.info("Exported %s -> %s", jsonl_path, csv_path)

def read_processed_jsonl_if_exists(path: str = "processed_mentions.jsonl", limit: int = 500) -> List[Dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, 'r', encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def process_streamed_to_processed(streamed_path: str = "streamed_mentions.jsonl",
                                  processed_path: str = "processed_mentions.jsonl",
                                  sentiment_fn=None,
                                  batch_size: int = 32):
    """
    Read new lines from streamed_mentions.jsonl, run sentiment_fn on text, and append to processed_mentions.jsonl.
    sentiment_fn should accept a list[str] and return list[{'label':..., 'score':...}]
    """
    if sentiment_fn is None:
        raise ValueError("Provide sentiment_fn to process_streamed_to_processed")

    if not os.path.exists(streamed_path):
        return 0

    lines = []
    with open(streamed_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                lines.append(json.loads(line))
            except Exception:
                continue

    texts = [l.get("text", "") for l in lines]
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            res = sentiment_fn(batch)
        except Exception as e:
            logger.exception("Sentiment fn failed on batch: %s", e)
            res = [{"label":"NEUTRAL","score":0.0} for _ in batch]
        results.extend(res)

    # Append sentiment results to processed file
    appended = 0
    with open(processed_path, 'a', encoding='utf-8') as outfh:
        for item, sent in zip(lines, results):
            obj = item.copy()
            obj["sentiment_label"] = sent.get("label")
            obj["sentiment_score"] = sent.get("score")
            outfh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            appended += 1
    logger.info("Processed %s streamed items -> appended %d to %s", len(lines), appended, processed_path)
    return appended