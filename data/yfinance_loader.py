import yfinance as yf
import pandas as pd
from typing import Optional, Union, List
import logging
from datetime import datetime, timedelta

def load_historical_data(ticker: str, 
                         period: Optional[str] = "5y", 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         interval: str = "1d") -> pd.DataFrame:
    """
    Load historical stock price data using yfinance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        period: Time period to download (e.g., '1d', '5d', '1mo', '3mo', '1y', '5y', 'max')
                Ignored if start_date and end_date are provided
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
        interval: Data interval (e.g., '1d', '1wk', '1mo')
    
    Returns:
        DataFrame with columns 'date' and 'price'
    """
    logging.info(f"Loading historical data for {ticker}")
    
    try:
        if start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        else:
            data = yf.download(ticker, period=period, interval=interval)
        
        if data.empty:
            logging.warning(f"No data found for ticker {ticker}")
            return pd.DataFrame(columns=['date', 'price'])
        
        data = data[['Close']].reset_index()
        data.columns = ['date', 'price']
        
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
            
        return data
        
    except Exception as e:
        logging.error(f"Error loading data for {ticker}: {str(e)}")
        raise

def load_multiple_tickers(tickers: List[str], period: str = "1y") -> dict:
    """
    Load historical data for multiple tickers
    
    Args:
        tickers: List of ticker symbols
        period: Time period to download
        
    Returns:
        Dictionary of DataFrames with ticker symbols as keys
    """
    result = {}
    for ticker in tickers:
        try:
            result[ticker] = load_historical_data(ticker, period)
        except Exception as e:
            logging.error(f"Failed to load {ticker}: {str(e)}")
            result[ticker] = pd.DataFrame() 
            
    return result

def get_latest_prices(tickers: Union[str, List[str]]) -> dict:
    """
    Get the most recent closing prices for one or more tickers
    
    Args:
        tickers: Single ticker string or list of ticker symbols
        
    Returns:
        Dictionary of latest prices with ticker symbols as keys
    """
    if isinstance(tickers, str):
        tickers = [tickers]
        
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    result = {}
    for ticker in tickers:
        try:
            df = load_historical_data(ticker, 
                                     start_date=start_date.strftime('%Y-%m-%d'),
                                     end_date=end_date.strftime('%Y-%m-%d'))
            
            if not df.empty:
                result[ticker] = df['price'].iloc[-1]
            else:
                result[ticker] = None
                
        except Exception as e:
            logging.error(f"Failed to get latest price for {ticker}: {str(e)}")
            result[ticker] = None
            
    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    apple_data = load_historical_data('AAPL', period='1y')
    print(f"Loaded {len(apple_data)} days of data for AAPL")
    
    tech_stocks = load_multiple_tickers(['AAPL', 'MSFT', 'GOOGL'], period='6mo')
    for ticker, data in tech_stocks.items():
        print(f"{ticker}: {len(data)} days of data")
    
    latest_prices = get_latest_prices(['AAPL', 'MSFT', 'GOOGL'])
    for ticker, price in latest_prices.items():
        print(f"{ticker} latest price: ${price:.2f}")
