import pandas as pd
import yfinance as yf
import numpy as np

class PortfolioManager:
    def __init__(self):
        # Holdings: {symbol: {'shares': int, 'buy_price': float}}
        self.holdings = {}

    def add_stock(self, symbol: str, shares: int, buy_price: float):
        self.holdings[symbol] = {'shares': shares, 'buy_price': buy_price}

    def remove_stock(self, symbol: str):
        if symbol in self.holdings:
            del self.holdings[symbol]

    def get_symbols(self):
        return list(self.holdings.keys())

    def fetch_prices(self, period="6mo"):
        symbols = self.get_symbols()
        if not symbols:
            return pd.DataFrame()
        data = yf.download(symbols, period=period, group_by='ticker', auto_adjust=True, progress=False)
        if len(symbols) == 1:
            data = {symbols[0]: data}
        return data

    def portfolio_value(self):
        symbols = self.get_symbols()
        if not symbols:
            return 0.0
        prices = yf.download(symbols, period="1d", group_by='ticker', auto_adjust=True, progress=False)
        total = 0.0
        for symbol, info in self.holdings.items():
            price = 0
            # Handle both single and multiple symbol cases
            try:
                if isinstance(prices, pd.DataFrame):
                    # Multiple symbols: MultiIndex columns
                    if ('Close', symbol) in prices.columns:
                        price = prices[('Close', symbol)].iloc[-1]
                    elif 'Close' in prices.columns:
                        price = prices['Close'].iloc[-1]
                elif isinstance(prices, dict) and symbol in prices:
                    # Dict of DataFrames (rare)
                    price = prices[symbol]['Close'].iloc[-1]
            except Exception:
                price = 0
            total += info['shares'] * price
        return total

    def performance(self):
        prices = self.fetch_prices()
        returns = {}
        for symbol in self.holdings:
            if symbol in prices:
                close = prices[symbol]['Close']
                returns[symbol] = close.pct_change().dropna()
        return returns

    def sharpe_ratio(self):
        returns = self.performance()
        ratios = {}
        for symbol, r in returns.items():
            if np.std(r) > 0:
                ratios[symbol] = (np.mean(r) / np.std(r)) * np.sqrt(252)
            else:
                ratios[symbol] = 0
        return ratios

    def volatility(self):
        returns = self.performance()
        vols = {}
        for symbol, r in returns.items():
            vols[symbol] = np.std(r) * np.sqrt(252)
        return vols

    def diversification(self):
        total = self.portfolio_value()
        weights = {}
        if np.isclose(total, 0.0):
            return weights
        prices = yf.download(self.get_symbols(), period="1d", group_by='ticker', auto_adjust=True, progress=False)
        for symbol, info in self.holdings.items():
            price = prices['Close'][symbol][-1] if symbol in prices['Close'] else 0
            weights[symbol] = (info['shares'] * price) / total
        return weights

# Example usage
def _demo():
    pm = PortfolioManager()
    pm.add_stock('AAPL', 10, 150)
    pm.add_stock('GOOGL', 5, 2500)
    print('Portfolio value:', pm.portfolio_value())
    print('Sharpe ratios:', pm.sharpe_ratio())
    print('Volatility:', pm.volatility())
    print('Diversification:', pm.diversification())

if __name__ == "__main__":
    _demo()
