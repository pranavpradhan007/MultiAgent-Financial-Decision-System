import unittest
import torch
from agents.market_predictor import MarketPredictorAgent
from agents.news_sentiment import NewsSentimentAgent
from agents.portfolio_manager import PortfolioManager


class TestNewsSentiment(unittest.TestCase):
    def test_prediction(self):
        agent = NewsSentimentAgent()
        self.assertIn(agent.predict("Great earnings report"), [0, 1, 2])

class TestMarketPredictor(unittest.TestCase):
    def test_data_loading(self):
        data = load_historical_data("AAPL")
        self.assertGreater(len(data), 1000)

if __name__ == "__main__":
    unittest.main()
