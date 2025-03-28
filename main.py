import asyncio
import logging
from typing import List, Dict
from agents.news_sentiment import NewsSentimentAgent
from agents.market_predictor import MarketPredictorAgent
from agents.portfolio_manager import PortfolioManagerAgent
from coordinator.coordinator import EnhancedDecisionCoordinator
from data.yfinance_loader import load_historical_data, get_latest_prices

async def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        # Initialize components
        news_agent = NewsSentimentAgent()
        market_agent = MarketPredictorAgent()
        portfolio_agent = PortfolioManagerAgent(initial_capital=100000)
        coordinator = EnhancedDecisionCoordinator(
            news_agent=news_agent,
            market_agent=market_agent,
            portfolio_agent=portfolio_agent
        )
        
        # Example execution
        ticker = "TSLA"
        headlines = ["Tesla announces breakthrough battery technology"]
        market_data = load_historical_data(ticker, period="1y")
        current_prices = get_latest_prices([ticker])
        
        decision = await coordinator.decide(
            ticker=ticker,
            headlines=headlines,
            market_data=market_data,
            current_prices=current_prices
        )
        
        logger.info(f"Final Decision for {ticker}:")
        logger.info(decision['explanation'])
        
        # Simulate a batch decision
        tickers = ["AAPL", "GOOGL", "MSFT"]
        batch_decisions = []
        for tick in tickers:
            batch_decisions.append({
                'ticker': tick,
                'headlines': [f"{tick} stock sees increased trading volume"],
                'market_data': load_historical_data(tick, period="1y"),
                'current_prices': get_latest_prices([tick])
            })
        
        batch_results = await coordinator.batch_decide(batch_decisions)
        
        for result in batch_results:
            logger.info(f"Batch Decision for {result['ticker']}:")
            logger.info(result['explanation'])
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
