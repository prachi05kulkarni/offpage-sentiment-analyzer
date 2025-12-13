-- Minimal schema for mentions (reference; db.create_tables will create)
CREATE TABLE IF NOT EXISTS mentions (
  id SERIAL PRIMARY KEY,
  source_platform TEXT NOT NULL,
  source_type TEXT,
  source_id TEXT,
  title TEXT,
  text TEXT,
  url TEXT,
  subreddit TEXT,
  created_utc TIMESTAMP WITH TIME ZONE,
  sentiment_label TEXT,
  sentiment_score REAL,
  raw JSONB,
  ingested_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mentions_platform ON mentions (source_platform);
CREATE INDEX IF NOT EXISTS idx_mentions_created ON mentions (created_utc);
CREATE INDEX IF NOT EXISTS idx_mentions_sentiment ON mentions (sentiment_label);