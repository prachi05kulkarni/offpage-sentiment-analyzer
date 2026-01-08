-- Updated schema: mentions with domain column, plus kpis table for daily aggregates

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