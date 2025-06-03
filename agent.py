import os
from langchain_community.llms import Ollama
from langchain.agents import Tool
from langchain.agents import create_react_agent
try:
    from langchain_community.agents.react.base import AGENT_PROMPT
except ImportError:
    from langchain.prompts import PromptTemplate
    AGENT_PROMPT = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
        template=(
            "You are a helpful AI finance agent.\n"
            "You have access to the following tools: {tool_names}.\n"
            "Tool details:\n{tools}\n"
            "Use them as needed to answer user questions.\n"
            "Previous steps: {agent_scratchpad}\n"
            "User: {input}\n"
            "Answer:"
        )
    )
from langchain_experimental.tools import PythonREPLTool

# Import your modules
import ai_analysis
import news_sentiment
import portfolio_manager

# Set up Ollama LLM (ensure Ollama is running locally with a model downloaded)
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")  # Default to llama3
llm = Ollama(model=ollama_model, base_url="http://localhost:11434")

# Initialize a global portfolio manager instance
pm = portfolio_manager.PortfolioManager()
# (For demo: add some stocks. In production, connect to user/session data)
pm.add_stock('AAPL', 10, 150)
pm.add_stock('GOOGL', 5, 2500)

# Tool: Portfolio Summary
def portfolio_summary_tool(_):
    value = pm.portfolio_value()
    weights = pm.diversification()
    weight_str = ', '.join(f"{k}: {v:.2%}" for k, v in weights.items())
    return f"Portfolio value: ${value:,.2f}\nDiversification: {weight_str}"

# Tool: Sharpe Ratio
def sharpe_ratio_tool(_):
    ratios = pm.sharpe_ratio()
    return '\n'.join(f"{k}: {v:.2f}" for k, v in ratios.items())

# Tool: Volatility
def volatility_tool(_):
    vols = pm.volatility()
    return '\n'.join(f"{k}: {v:.2%}" for k, v in vols.items())

# Tool: Diversification
def diversification_tool(_):
    weights = pm.diversification()
    return '\n'.join(f"{k}: {v:.2%}" for k, v in weights.items())

# Tool: News Sentiment
news_sentiment_tool = Tool(
    name="NewsSentiment",
    func=lambda stock: str(news_sentiment.analyze_news_sentiment(news_sentiment.fetch_news_for_stocks([stock]))),
    description="Analyze recent news sentiment for a given stock symbol (e.g., 'AAPL')."
)

# Tool: Technical Analysis

def technical_analysis_func(symbol):
    import yfinance as yf
    import pandas as pd
    # Fetch historical prices
    data = yf.download(symbol, period="6mo", auto_adjust=True, progress=False)
    if data.empty:
        return f"No data found for {symbol}."
    close = data['Close']
    ma = ai_analysis.moving_average(close, window=14).iloc[-1]
    rsi_val = ai_analysis.rsi(close, window=14).iloc[-1]
    vol = ai_analysis.calculate_volatility(close, window=30).iloc[-1]
    return (
        f"Technical Analysis for {symbol}:\n"
        f"- 14-day Moving Average: {ma:.2f}\n"
        f"- RSI (14): {rsi_val:.2f}\n"
        f"- 30-day Volatility: {vol:.2%}"
    )

technical_analysis_tool = Tool(
    name="TechnicalAnalysis",
    func=technical_analysis_func,
    description="Run technical indicators (moving average, RSI, volatility) for a stock symbol (e.g., 'AAPL')."
)

# Python REPL tool for calculations
python_tool = PythonREPLTool()

# Assemble tools
tools = [
    Tool(name="PortfolioSummary", func=portfolio_summary_tool, description="Get portfolio value and diversification."),
    Tool(name="SharpeRatio", func=sharpe_ratio_tool, description="Get Sharpe ratio for each holding."),
    Tool(name="Volatility", func=volatility_tool, description="Get volatility for each holding."),
    Tool(name="Diversification", func=diversification_tool, description="Get diversification (weights) for each holding."),
    news_sentiment_tool,
    technical_analysis_tool,
    python_tool
]

# Initialize agent (no memory for now to avoid chat_history error)
agent = create_react_agent(llm, tools, AGENT_PROMPT)

if __name__ == "__main__":
    print("Portfolio AI Agent (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent.invoke({"input": user_input, "intermediate_steps": []})
        print("Agent:", response)
