# Offpage Sentiment Analyzer

This tool analyzes "Share of Voice" and Sentiment for your brand and competitors on **Reddit** and **Quora**.

## Prerequisites

Dependencies are already installed. If you need to reinstall:
```bash
pip install -r requirements.txt
```

## Setup API Keys

To fetch real data, you must configure your API keys in the `.env` file.

1.  Open the file named `.env` in this directory.
2.  Add your keys as follows:

```ini
# Reddit Credentials
# Go to https://www.reddit.com/prefs/apps to create an app (script type)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=offpage-sentiment-analyzer/0.1 by your_reddit_username


# Google/SerpAPI Credentials (for Quora)
# Go to https://serpapi.com/ to get a key
SERPAPI_KEY=b2def6f3eedbf41664b82db9c9a95cb20dae5a8455b4a37f287313a26abb0d80
```

> **Note**: Without these keys, the application will return 0 results.

## How to Run

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Features

-   **Share of Voice**: Compares mention volume between your brand and competitors.
-   **Sentiment Analysis**: Basic lexicon-based or HuggingFace model-based sentiment detection.
-   **Platform Breakdown**: Separate metrics for Reddit and Quora.