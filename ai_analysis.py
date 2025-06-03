import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler

# Sentiment Analysis using Hugging Face Transformers (distilbert-base-uncased-finetuned-sst-2-english)
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """Analyze sentiment of a news headline or article."""
    result = sentiment_pipeline(text[:512])  # Truncate to model max length
    return result[0]['label'], float(result[0]['score'])

# Technical Indicators
def moving_average(prices, window=14):
    return prices.rolling(window=window).mean()

def rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_volatility(prices, window=30):
    return prices.pct_change().rolling(window=window).std() * np.sqrt(window)

def sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate / 252
    return np.mean(excess_returns) / np.std(excess_returns)

def beta(stock_returns, market_returns):
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance

# Stock Recommendation (Simple Rule-based)
def recommend_stock(sentiment_score, rsi_value):
    """Recommend buy/hold/sell based on sentiment and RSI."""
    if sentiment_score == 'POSITIVE' and rsi_value < 30:
        return 'BUY'
    elif sentiment_score == 'NEGATIVE' and rsi_value > 70:
        return 'SELL'
    else:
        return 'HOLD'

# Example usage:
if __name__ == "__main__":
    # Example: Sentiment analysis
    print(analyze_sentiment("Apple stock surges after earnings beat expectations!"))

    # Example: Technicals
    prices = pd.Series([100, 102, 104, 103, 105, 107, 110, 108, 109, 111, 113, 112, 115, 117, 119])
    print("RSI:", rsi(prices))
    print("MA:", moving_average(prices))
    print("Volatility:", calculate_volatility(prices))
    returns = prices.pct_change().dropna()
    print("Sharpe Ratio:", sharpe_ratio(returns))
