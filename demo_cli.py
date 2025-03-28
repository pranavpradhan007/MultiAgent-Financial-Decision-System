import argparse
import asyncio
import logging
import sys
import os
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components
from agents.news_sentiment import NewsSentimentAgent
from agents.market_predictor import MarketPredictorAgent
from agents.portfolio_manager import PortfolioManagerAgent
from coordinator.coordinator import EnhancedDecisionCoordinator
from data.yfinance_loader import load_historical_data, get_latest_prices
from data.loaders import load_mock_headlines

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Sample portfolio for demo
SAMPLE_PORTFOLIO = {
    'cash': 100000,
    'holdings': {
        'AAPL': {'shares': 50, 'avg_price': 150},
        'MSFT': {'shares': 30, 'avg_price': 280},
        'GOOGL': {'shares': 10, 'avg_price': 2500}
    },
    'risk_tolerance': 'medium'
}

class FinancialAdvisorCLI:
    """Interactive CLI for the multi-agent financial decision system"""
    
    def __init__(self):
        """Initialize the CLI with all required components"""
        self.portfolio = SAMPLE_PORTFOLIO.copy()
        self.setup_agents()
        
        
    def setup_agents(self):
        """Initialize all agents and the coordinator"""
        logger.info("Initializing financial decision system...")
        
        # Initialize agents
        self.news_agent = NewsSentimentAgent()
        self.market_agent = MarketPredictorAgent(input_size=7)  # Make sure input_size matches features
        self.portfolio_agent = PortfolioManagerAgent(initial_capital=self.portfolio['cash'])
        
        # Initialize coordinator
        self.coordinator = EnhancedDecisionCoordinator(
            news_agent=self.news_agent,
            market_agent=self.market_agent,
            portfolio_agent=self.portfolio_agent
        )
        
        logger.info("System initialized successfully")
    
    async def get_recommendation(self, ticker: str, headline_count: int = 5) -> Dict:
        """Get a recommendation for a specific ticker"""
        logger.info(f"Analyzing {ticker}...")
        
        # Load market data
        market_data = load_historical_data(ticker, period="1y")
        if market_data.empty:
            return {"error": f"Could not find data for ticker {ticker}"}
        
        # Add technical indicators
        enhanced_data = self.market_agent.add_features(market_data.copy())
        
        # Get current prices
        current_prices = get_latest_prices([ticker])
        if ticker not in current_prices or current_prices[ticker] is None:
            return {"error": f"Could not get current price for {ticker}"}
        
        # Get headlines
        headlines = load_mock_headlines(ticker, count=headline_count)
        
        # Log what we're analyzing
        logger.info(f"Analyzing {len(headlines)} headlines for {ticker}")
        for i, headline in enumerate(headlines):
            logger.info(f"  {i+1}. {headline}")
        
        # Get recommendation
        decision = await self.coordinator.decide(
            ticker=ticker,
            headlines=headlines,
            market_data=enhanced_data,
            current_prices=current_prices
        )
        
        return decision
    
    def display_recommendation(self, decision: Dict):
        """Display a recommendation in a user-friendly format"""
        if "error" in decision:
            print(f"\n❌ ERROR: {decision['error']}")
            return
            
        # Extract key information
        ticker = decision.get('ticker', 'Unknown')
        action = decision.get('action', 'Unknown').upper()
        confidence = decision.get('confidence', 0)
        
        # Format the output
        print("\n" + "="*80)
        print(f"📊 RECOMMENDATION FOR {ticker}: {action} ({confidence:.0%} CONFIDENCE)")
        print("="*80)
        
        # Check if there's an override
        if 'override_reason' in decision:
            print(f"\n⚠️ OVERRIDE ALERT: {decision['override_reason']}")
        
        # Print the full explanation
        print("\n" + decision.get('explanation', 'No explanation available'))
        
    def display_portfolio(self):
        """Display the current portfolio"""
        print("\n" + "="*80)
        print("📈 CURRENT PORTFOLIO")
        print("="*80)
        
        # Cash position
        print(f"\nCash: ${self.portfolio['cash']:,.2f}")
        
        # Holdings
        if self.portfolio['holdings']:
            print("\nHoldings:")
            print(f"{'Ticker':<10}{'Shares':<10}{'Avg Price':<15}{'Current Value':<15}")
            print("-"*50)
            
            # Get current prices for all holdings
            tickers = list(self.portfolio['holdings'].keys())
            current_prices = get_latest_prices(tickers)
            
            total_value = self.portfolio['cash']
            for ticker, position in self.portfolio['holdings'].items():
                current_price = current_prices.get(ticker, 0)
                current_value = position['shares'] * current_price
                total_value += current_value
                
                print(f"{ticker:<10}{position['shares']:<10}{position['avg_price']:<15,.2f}${current_value:<15,.2f}")
            
            print("\n" + "-"*50)
            print(f"Total Portfolio Value: ${total_value:,.2f}")
        else:
            print("\nNo holdings in portfolio")
        
        print(f"\nRisk Tolerance: {self.portfolio['risk_tolerance'].capitalize()}")
    
    def update_portfolio(self, action: str, ticker: str, shares: int, price: float):
        """Update the portfolio based on a transaction"""
        if action.lower() == 'buy':
            # Check if we have enough cash
            cost = shares * price
            if cost > self.portfolio['cash']:
                print(f"\n❌ Not enough cash to buy {shares} shares of {ticker} at ${price:.2f}")
                return False
                
            # Update cash
            self.portfolio['cash'] -= cost
            
            # Update holdings
            if ticker in self.portfolio['holdings']:
                # Calculate new average price
                current_shares = self.portfolio['holdings'][ticker]['shares']
                current_avg_price = self.portfolio['holdings'][ticker]['avg_price']
                new_total_shares = current_shares + shares
                new_avg_price = ((current_shares * current_avg_price) + (shares * price)) / new_total_shares
                
                self.portfolio['holdings'][ticker]['shares'] = new_total_shares
                self.portfolio['holdings'][ticker]['avg_price'] = new_avg_price
            else:
                self.portfolio['holdings'][ticker] = {
                    'shares': shares,
                    'avg_price': price
                }
                
            print(f"\n✅ Bought {shares} shares of {ticker} at ${price:.2f}")
            return True
            
        elif action.lower() == 'sell':
            # Check if we have the shares
            if ticker not in self.portfolio['holdings']:
                print(f"\n❌ No shares of {ticker} in portfolio")
                return False
                
            current_shares = self.portfolio['holdings'][ticker]['shares']
            if shares > current_shares:
                print(f"\n❌ Not enough shares to sell (have {current_shares}, trying to sell {shares})")
                return False
                
            # Update cash
            proceeds = shares * price
            self.portfolio['cash'] += proceeds
            
            # Update holdings
            if shares == current_shares:
                del self.portfolio['holdings'][ticker]
            else:
                self.portfolio['holdings'][ticker]['shares'] -= shares
                
            print(f"\n✅ Sold {shares} shares of {ticker} at ${price:.2f}")
            return True
            
        else:
            print(f"\n❌ Invalid action: {action}")
            return False
    
    async def interactive_loop(self):
        """Run the interactive CLI loop"""
        print("\n🤖 Welcome to the Multi-Agent Financial Decision System 🤖")
        print("\nThis system uses three specialized agents to help you make financial decisions:")
        print("  1. News Sentiment Agent - Analyzes financial news headlines")
        print("  2. Market Predictor Agent - Analyzes technical indicators")
        print("  3. Portfolio Manager Agent - Evaluates your portfolio composition")
        
        while True:
            print("\n" + "-"*80)
            print("COMMANDS:")
            print("  analyze [ticker] - Get a recommendation for a specific stock")
            print("  portfolio - View your current portfolio")
            print("  buy [ticker] [shares] - Buy shares of a stock")
            print("  sell [ticker] [shares] - Sell shares of a stock")
            print("  risk [low/medium/high] - Change your risk tolerance")
            print("  exit - Exit the program")
            print("-"*80)
            
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'exit':
                print("\nThank you for using the Multi-Agent Financial Decision System. Goodbye!")
                break
                
            elif command == 'portfolio':
                self.display_portfolio()
                
            elif command.startswith('analyze '):
                parts = command.split()
                if len(parts) >= 2:
                    ticker = parts[1].upper()
                    decision = await self.get_recommendation(ticker)
                    self.display_recommendation(decision)
                else:
                    print("\n❌ Please specify a ticker (e.g., analyze TSLA)")
                    
            elif command.startswith('buy '):
                parts = command.split()
                if len(parts) >= 3:
                    ticker = parts[1].upper()
                    try:
                        shares = int(parts[2])
                        # Get current price
                        current_prices = get_latest_prices([ticker])
                        if ticker in current_prices and current_prices[ticker] is not None:
                            price = current_prices[ticker]
                            self.update_portfolio('buy', ticker, shares, price)
                        else:
                            print(f"\n❌ Could not get current price for {ticker}")
                    except ValueError:
                        print("\n❌ Shares must be a number")
                else:
                    print("\n❌ Please specify ticker and shares (e.g., buy TSLA 10)")
                    
            elif command.startswith('sell '):
                parts = command.split()
                if len(parts) >= 3:
                    ticker = parts[1].upper()
                    try:
                        shares = int(parts[2])
                        # Get current price
                        current_prices = get_latest_prices([ticker])
                        if ticker in current_prices and current_prices[ticker] is not None:
                            price = current_prices[ticker]
                            self.update_portfolio('sell', ticker, shares, price)
                        else:
                            print(f"\n❌ Could not get current price for {ticker}")
                    except ValueError:
                        print("\n❌ Shares must be a number")
                else:
                    print("\n❌ Please specify ticker and shares (e.g., sell TSLA 10)")
                    
            elif command.startswith('risk '):
                parts = command.split()
                if len(parts) >= 2:
                    risk = parts[1].lower()
                    if risk in ['low', 'medium', 'high']:
                        self.portfolio['risk_tolerance'] = risk
                        print(f"\n✅ Risk tolerance updated to {risk}")
                    else:
                        print("\n❌ Risk must be 'low', 'medium', or 'high'")
                else:
                    print("\n❌ Please specify risk level (e.g., risk medium)")
                    
            else:
                print("\n❌ Unknown command. Type 'exit' to quit.")

def main():
    """Main entry point for the CLI"""
    cli = FinancialAdvisorCLI()
    
    try:
        # Run the interactive loop
        asyncio.run(cli.interactive_loop())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
