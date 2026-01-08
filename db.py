"""
db.py - lightweight Postgres helper for mentions + KPI storage.

Usage:
  from db import ensure_tables, insert_mention, upsert_daily_kpi, query_domain_totals
  ensure_tables()  # run once at startup (safe: IF NOT EXISTS)
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import psycopg2
import psycopg2.extras

logger = logging.getLogger("offpage_db")
DATABASE_URL = os.getenv("DATABASE_URL")

def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set in environment")
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)

def ensure_tables():
    """
    Create mentions and kpis tables if they don't exist.
    Non-destructive: uses IF NOT EXISTS.
    """
    sql = """
    CREATE TABLE IF NOT EXISTS mentions (
      id SERIAL PRIMARY KEY,
      brand TEXT,
      source_platform TEXT NOT NULL,
      source_type TEXT,
      source_id TEXT,
      title TEXT,
      text TEXT,
      url TEXT,
      subreddit TEXT,
      created_utc TIMESTAMP WITH TIME ZONE,
      domain TEXT,
      sentiment_label TEXT,
      sentiment_score REAL,
      raw JSONB,
      ingested_at TIMESTAMP WITH TIME ZONE DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_mentions_platform ON mentions (source_platform);
    CREATE INDEX IF NOT EXISTS idx_mentions_created ON mentions (created_utc);
    CREATE INDEX IF NOT EXISTS idx_mentions_sentiment ON mentions (sentiment_label);

    CREATE TABLE IF NOT EXISTS kpis (
      id SERIAL PRIMARY KEY,
      brand TEXT NOT NULL,
      day_date DATE NOT NULL,
      mentions_total INTEGER DEFAULT 0,
      reddit_count INTEGER DEFAULT 0,
      quora_count INTEGER DEFAULT 0,
      positive_count INTEGER DEFAULT 0,
      neutral_count INTEGER DEFAULT 0,
      negative_count INTEGER DEFAULT 0,
      sov_pct REAL DEFAULT 0.0,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
      UNIQUE (brand, day_date)
    );
    """
    if not DATABASE_URL:
        logger.debug("DATABASE_URL not set, skipping ensure_tables.")
        return
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        logger.info("DB tables ensured")
    except Exception as e:
        logger.exception("ensure_tables error: %s", e)
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def insert_mention(mention: Dict[str, Any]):
    """
    Insert a single mention into mentions table.
    mention is expected to contain keys: brand, platform, title, text, url, subreddit,
    created_utc (timestamp), domain (optional), sentiment_label, sentiment_score, raw.
    """
    if not DATABASE_URL:
        logger.debug("DATABASE_URL not set, skipping DB insert.")
        return None
    sql = """
    INSERT INTO mentions
      (brand, source_platform, source_type, source_id, title, text, url, subreddit, created_utc, domain, sentiment_label, sentiment_score, raw)
    VALUES
      (%(brand)s, %(source_platform)s, %(source_type)s, %(source_id)s, %(title)s, %(text)s, %(url)s, %(subreddit)s, %(created_utc)s, %(domain)s, %(sentiment_label)s, %(sentiment_score)s, %(raw)s)
    RETURNING id;
    """
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(sql, {
            "brand": mention.get("brand"),
            "source_platform": mention.get("platform") or mention.get("source_platform") or "unknown",
            "source_type": mention.get("source_type"),
            "source_id": mention.get("id") or mention.get("source_id"),
            "title": mention.get("title"),
            "text": mention.get("text"),
            "url": mention.get("url"),
            "subreddit": mention.get("subreddit"),
            "created_utc": mention.get("created_utc"),
            "domain": mention.get("domain"),
            "sentiment_label": mention.get("sentiment_label"),
            "sentiment_score": mention.get("sentiment_score"),
            "raw": json.dumps(mention.get("raw") or mention),
        })
        new_id = cur.fetchone()["id"]
        conn.commit()
        return new_id
    except Exception as e:
        logger.exception("insert_mention error: %s", e)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def upsert_daily_kpi(brand: str, day_date, kpi_values: Dict[str, int], sov_pct: Optional[float] = None):
    """
    Insert or update daily KPI row for a brand.
    kpi_values expected keys: mentions_total, reddit_count, quora_count, positive_count, neutral_count, negative_count
    """
    if not DATABASE_URL:
        logger.debug("DATABASE_URL not set, skipping KPI upsert.")
        return
    sql = """
    INSERT INTO kpis (brand, day_date, mentions_total, reddit_count, quora_count, positive_count, neutral_count, negative_count, sov_pct)
    VALUES (%(brand)s, %(day_date)s, %(mentions_total)s, %(reddit_count)s, %(quora_count)s, %(positive_count)s, %(neutral_count)s, %(negative_count)s, %(sov_pct)s)
    ON CONFLICT (brand, day_date) DO UPDATE
      SET mentions_total = EXCLUDED.mentions_total,
          reddit_count = EXCLUDED.reddit_count,
          quora_count = EXCLUDED.quora_count,
          positive_count = EXCLUDED.positive_count,
          neutral_count = EXCLUDED.neutral_count,
          negative_count = EXCLUDED.negative_count,
          sov_pct = EXCLUDED.sov_pct,
          created_at = now();
    """
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        params = {
            "brand": brand,
            "day_date": day_date,
            "mentions_total": int(kpi_values.get("mentions_total", 0)),
            "reddit_count": int(kpi_values.get("reddit_count", 0)),
            "quora_count": int(kpi_values.get("quora_count", 0)),
            "positive_count": int(kpi_values.get("positive_count", 0)),
            "neutral_count": int(kpi_values.get("neutral_count", 0)),
            "negative_count": int(kpi_values.get("negative_count", 0)),
            "sov_pct": float(sov_pct or 0.0),
        }
        cur.execute(sql, params)
        conn.commit()
    except Exception as e:
        logger.exception("upsert_daily_kpi error: %s", e)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def query_domain_totals(domains: List[str], days: int = 30) -> Dict[str,int]:
    """
    Return mapping domain -> mentions count for the last `days` days.
    domains: list of domain names (lowercase expected)
    """
    out = {}
    if not DATABASE_URL:
        logger.debug("DATABASE_URL not set, skipping domain totals query.")
        return {d: 0 for d in domains}
    sql = """
    SELECT domain, count(*) as cnt
    FROM mentions
    WHERE domain = ANY(%(domains)s) AND created_utc >= (now() - (%(days)s || ' days')::interval)
    GROUP BY domain;
    """
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(sql, {"domains": domains, "days": days})
        rows = cur.fetchall()
        for r in rows:
            if isinstance(r, dict):
                out[r.get("domain")] = int(r.get("cnt",0))
            else:
                out[r[0]] = int(r[1])
    except Exception as e:
        logger.exception("query_domain_totals error: %s", e)
    finally:
        if conn:
            conn.close()
    for d in domains:
        out.setdefault(d, 0)
    return out