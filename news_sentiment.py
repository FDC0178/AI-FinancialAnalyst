import os
import requests
from transformers import pipeline
from collections import defaultdict
from typing import List, Dict

# Load NewsAPI key from environment variable (set NEWSAPI_KEY in .env or system env)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWS_ENDPOINT = "https://newsapi.org/v2/everything"

# Sentiment pipeline (shared with ai_analysis.py for efficiency)
sentiment_pipeline = pipeline("sentiment-analysis")

def fetch_news_for_stocks(stocks: List[str], max_articles=5) -> Dict[str, List[str]]:
    """
    Fetch recent news headlines for each stock symbol using NewsAPI.
    Returns dict: {symbol: [headline1, headline2, ...]}
    """
    news_dict = defaultdict(list)
    for symbol in stocks:
        params = {
            "q": symbol,
            "apiKey": NEWSAPI_KEY,
            "pageSize": max_articles,
            "sortBy": "publishedAt",
            "language": "en"
        }
        try:
            resp = requests.get(NEWS_ENDPOINT, params=params, timeout=10)
            if resp.status_code == 200:
                articles = resp.json().get("articles", [])
                headlines = [a["title"] for a in articles if "title" in a]
                news_dict[symbol].extend(headlines)
            else:
                print(f"NewsAPI error for {symbol}: {resp.status_code}")
        except Exception as e:
            print(f"Exception fetching news for {symbol}: {e}")
    return dict(news_dict)

def analyze_news_sentiment(news_dict: Dict[str, List[str]]):
    """
    Run sentiment analysis on all news headlines for each stock.
    Returns dict: {symbol: {"positive": int, "negative": int, "neutral": int, "avg_score": float}}
    """
    sentiment_summary = {}
    for symbol, headlines in news_dict.items():
        pos, neg, neu, total_score = 0, 0, 0, 0.0
        for headline in headlines:
            result = sentiment_pipeline(headline[:512])[0]
            label = result["label"].upper()
            score = float(result["score"])
            total_score += score if label == "POSITIVE" else -score if label == "NEGATIVE" else 0
            if label == "POSITIVE":
                pos += 1
            elif label == "NEGATIVE":
                neg += 1
            else:
                neu += 1
        n = len(headlines)
        sentiment_summary[symbol] = {
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "avg_score": total_score / n if n > 0 else 0.0,
            "total_articles": n
        }
    return sentiment_summary

# Example usage
def _demo():
    stocks = ["AAPL", "GOOGL"]
    news = fetch_news_for_stocks(stocks)
    print(news)
    sentiment = analyze_news_sentiment(news)
    print(sentiment)

if __name__ == "__main__":
    _demo()
