import streamlit as st
import pandas as pd
import portfolio_manager
import ai_analysis
import news_sentiment

st.set_page_config(page_title="Finance AI Agent", layout="wide")
st.title("💹 Portfolio Analysis & Investment Research Agent")

# Initialize portfolio manager (singleton for Streamlit session)
if 'pm' not in st.session_state:
    st.session_state.pm = portfolio_manager.PortfolioManager()
    st.session_state.pm.add_stock('AAPL', 10, 150)
    st.session_state.pm.add_stock('GOOGL', 5, 2500)
pm = st.session_state.pm

# Sidebar navigation
page = st.sidebar.radio("Go to", ["Portfolio Overview", "Technical Analysis", "News Sentiment", "Chat"])

if page == "Portfolio Overview":
    st.header("Portfolio Overview")
    value = pm.portfolio_value()
    st.metric("Total Portfolio Value", f"${value:,.2f}")
    weights = pm.diversification()
    st.subheader("Diversification")
    if weights:
        st.bar_chart(pd.DataFrame.from_dict(weights, orient='index', columns=['Weight']))
    else:
        st.info("No diversification data to display. Add stocks to your portfolio.")
    st.subheader("Add/Remove Holdings")
    with st.form("add_stock_form"):
        symbol = st.text_input("Stock Symbol", max_chars=8)
        shares = st.number_input("Shares", min_value=1, value=1)
        buy_price = st.number_input("Buy Price", min_value=0.0, value=100.0)
        add = st.form_submit_button("Add Stock")
        if add and symbol:
            pm.add_stock(symbol.upper(), shares, buy_price)
            st.success(f"Added {shares} shares of {symbol.upper()} at ${buy_price}")
    with st.form("remove_stock_form"):
        remove_symbol = st.selectbox("Remove Stock", pm.get_symbols())
        remove = st.form_submit_button("Remove")
        if remove and remove_symbol:
            pm.remove_stock(remove_symbol)
            st.warning(f"Removed {remove_symbol} from portfolio")
    st.subheader("Risk Analytics")
    sharpe = pm.sharpe_ratio()
    vol = pm.volatility()
    st.write("**Sharpe Ratio:**")
    st.json(sharpe)
    st.write("**Volatility:**")
    st.json(vol)

elif page == "Technical Analysis":
    st.header("Technical Analysis")
    symbol = st.text_input("Stock Symbol for Analysis", value="AAPL")
    if st.button("Analyze") and symbol:
        import yfinance as yf
        data = yf.download(symbol, period="6mo", auto_adjust=True, progress=False)
        if data.empty:
            st.error(f"No data found for {symbol}")
        else:
            close = data['Close']
            ma = ai_analysis.moving_average(close, window=14)
            rsi_val = ai_analysis.rsi(close, window=14)
            vol = ai_analysis.calculate_volatility(close, window=30)
            st.line_chart(close, height=200)
            ma_val = float(ma.iloc[-1]) if not ma.empty else float('nan')
            rsi_value = float(rsi_val.iloc[-1]) if not rsi_val.empty else float('nan')
            vol_value = float(vol.iloc[-1]) if not vol.empty else float('nan')
            st.write(f"**14-day Moving Average:** {ma_val:.2f}")
            st.write(f"**RSI (14):** {rsi_value:.2f}")
            st.write(f"**30-day Volatility:** {vol_value:.2%}")

elif page == "News Sentiment":
    st.header("News Sentiment Analysis")
    symbol = st.text_input("Stock Symbol for News Sentiment", value="AAPL")
    if st.button("Analyze News Sentiment") and symbol:
        news = news_sentiment.fetch_news_for_stocks([symbol])
        sentiment = news_sentiment.analyze_news_sentiment(news)
        st.write(f"**Sentiment for {symbol}:**")
        st.json(sentiment)
        headlines = news.get(symbol)
        if headlines:
            st.write("**Recent Headlines:**")
            for headline in headlines:
                st.write(f"- {headline}")
        else:
            st.info("No news headlines found for this symbol.")

elif page == "Chat":
    st.header("Conversational AI Agent")
    st.write("Chat with your AI finance agent below. Ask anything about your portfolio, stocks, or the market!")

    # Import the agent (ensure agent.py is importable as a module)
    import agent as ai_agent
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    def get_agent_response(user_input):
        try:
            response = ai_agent.agent.invoke({"input": user_input, "intermediate_steps": []})
            # If response is a dict with 'output', use that, else str
            if isinstance(response, dict) and 'output' in response:
                return response['output']
            return str(response)
        except Exception as e:
            return f"[Agent Error] {str(e)}"

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Agent:** {msg['content']}")

    # Input box
    user_input = st.text_input("Type your message and press Enter", key="chat_input")
    if st.button("Send") and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        agent_reply = get_agent_response(user_input)
        st.session_state.chat_history.append({"role": "agent", "content": agent_reply})
        st.experimental_rerun()
