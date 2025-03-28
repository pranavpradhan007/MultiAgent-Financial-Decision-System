import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Union, Tuple, Optional

class PortfolioManagerAgent:
    """
    Advanced Portfolio Manager Agent that evaluates portfolio composition
    and makes recommendations based on modern portfolio theory principles.
    
    Features:
    - Risk-adjusted position sizing
    - Diversification analysis
    - Stop-loss and take-profit logic
    - Cash management rules
    - Performance tracking
    - Tax-loss harvesting detection
    """
    
    def __init__(self, initial_capital: float = 10000, risk_tolerance: str = 'medium'):
        """
        Initialize the Portfolio Manager
        
        Args:
            initial_capital: Starting cash amount
            risk_tolerance: Risk profile ('conservative', 'medium', 'aggressive')
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("PortfolioManager")
        
        # Validate inputs
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        if risk_tolerance not in ['conservative', 'medium', 'aggressive']:
            raise ValueError("Risk tolerance must be 'conservative', 'medium', or 'aggressive'")
        
        # Initialize portfolio
        self.portfolio = {
            'cash': initial_capital,
            'holdings': {},  # {ticker: {'shares': n, 'avg_price': x, 'purchase_date': date}}
            'risk_tolerance': risk_tolerance,
            'history': [],
            'performance': {
                'starting_value': initial_capital,
                'current_value': initial_capital,
                'roi': 0.0,
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }
        }
        
        # Risk model parameters
        self.risk_models = {
            'conservative': {
                'max_position_size': 0.05,  # Max 5% in any position
                'target_cash': 0.30,        # 30% cash target
                'stop_loss': 0.05,          # 5% stop loss
                'take_profit': 0.15,        # 15% take profit
                'max_sector_exposure': 0.20, # Max 20% in any sector
                'rebalance_threshold': 0.05  # Rebalance when 5% off target
            },
            'medium': {
                'max_position_size': 0.10,   # Max 10% in any position
                'target_cash': 0.20,         # 20% cash target
                'stop_loss': 0.10,           # 10% stop loss
                'take_profit': 0.25,         # 25% take profit
                'max_sector_exposure': 0.30,  # Max 30% in any sector
                'rebalance_threshold': 0.10   # Rebalance when 10% off target
            },
            'aggressive': {
                'max_position_size': 0.20,   # Max 20% in any position
                'target_cash': 0.10,         # 10% cash target
                'stop_loss': 0.15,           # 15% stop loss
                'take_profit': 0.40,         # 40% take profit
                'max_sector_exposure': 0.40,  # Max 40% in any sector
                'rebalance_threshold': 0.15   # Rebalance when 15% off target
            }
        }
        
        # Sector mappings for diversification analysis
        self.sector_mappings = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'META': 'Technology', 'NVDA': 'Technology', 'AMD': 'Technology',
            
            # Consumer
            'AMZN': 'Consumer', 'TSLA': 'Consumer', 'NKE': 'Consumer',
            'SBUX': 'Consumer', 'MCD': 'Consumer', 'WMT': 'Consumer',
            
            # Financial
            'JPM': 'Financial', 'BAC': 'Financial', 'GS': 'Financial',
            'V': 'Financial', 'MA': 'Financial', 'AXP': 'Financial',
            
            # Healthcare
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
            'ABBV': 'Healthcare', 'MRK': 'Healthcare', 'ABT': 'Healthcare',
            
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
            'BP': 'Energy', 'SLB': 'Energy', 'EOG': 'Energy',
            
            # Default for unknown tickers
            'DEFAULT': 'Other'
        }
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value
        
        Args:
            current_prices: Dictionary of current prices {ticker: price}
            
        Returns:
            Total portfolio value
        """
        # Calculate holdings value
        holdings_value = sum(
            self.portfolio['holdings'].get(ticker, {}).get('shares', 0) * price
            for ticker, price in current_prices.items()
            if ticker in self.portfolio['holdings']
        )
        
        # Add cash
        return self.portfolio['cash'] + holdings_value
    
    def get_position_info(self, ticker: str, current_price: float) -> Dict:
        """
        Get detailed information about a specific position
        
        Args:
            ticker: Stock ticker
            current_price: Current price of the stock
            
        Returns:
            Position details dictionary
        """
        if ticker not in self.portfolio['holdings']:
            return {
                'shares': 0,
                'avg_price': 0,
                'market_value': 0,
                'cost_basis': 0,
                'profit_loss': 0,
                'profit_loss_pct': 0,
                'days_held': 0
            }
        
        position = self.portfolio['holdings'][ticker]
        shares = position['shares']
        avg_price = position['avg_price']
        purchase_date = datetime.strptime(position['purchase_date'], '%Y-%m-%d')
        days_held = (datetime.now() - purchase_date).days
        
        market_value = shares * current_price
        cost_basis = shares * avg_price
        profit_loss = market_value - cost_basis
        profit_loss_pct = (profit_loss / cost_basis) if cost_basis > 0 else 0
        
        return {
            'shares': shares,
            'avg_price': avg_price,
            'market_value': market_value,
            'cost_basis': cost_basis,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'days_held': days_held
        }
    
    def get_sector_exposure(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate exposure by sector
        
        Args:
            current_prices: Dictionary of current prices
            
        Returns:
            Dictionary of sector allocations as percentages
        """
        total_value = self.get_portfolio_value(current_prices)
        if total_value == 0:
            return {'Cash': 1.0}
        
        # Initialize with cash
        sector_values = {'Cash': self.portfolio['cash']}
        
        # Add up values by sector
        for ticker, position in self.portfolio['holdings'].items():
            if ticker in current_prices:
                sector = self.sector_mappings.get(ticker, self.sector_mappings['DEFAULT'])
                value = position['shares'] * current_prices[ticker]
                
                if sector in sector_values:
                    sector_values[sector] += value
                else:
                    sector_values[sector] = value
        
        # Convert to percentages
        return {sector: value/total_value for sector, value in sector_values.items()}
    
    def buy(self, ticker: str, shares: float, price: float) -> Dict:
        """
        Execute a buy order
        
        Args:
            ticker: Stock ticker
            shares: Number of shares to buy
            price: Purchase price per share
            
        Returns:
            Transaction result
        """
        cost = shares * price
        
        # Check if enough cash
        if cost > self.portfolio['cash']:
            return {
                'success': False,
                'message': f"Insufficient cash. Required: ${cost:.2f}, Available: ${self.portfolio['cash']:.2f}"
            }
        
        # Execute transaction
        self.portfolio['cash'] -= cost
        
        # Update holdings
        if ticker in self.portfolio['holdings']:
            # Update existing position with new average price
            current_shares = self.portfolio['holdings'][ticker]['shares']
            current_avg_price = self.portfolio['holdings'][ticker]['avg_price']
            
            total_shares = current_shares + shares
            total_cost = (current_shares * current_avg_price) + cost
            new_avg_price = total_cost / total_shares
            
            self.portfolio['holdings'][ticker]['shares'] = total_shares
            self.portfolio['holdings'][ticker]['avg_price'] = new_avg_price
        else:
            # Create new position
            self.portfolio['holdings'][ticker] = {
                'shares': shares,
                'avg_price': price,
                'purchase_date': datetime.now().strftime('%Y-%m-%d')
            }
        
        # Log transaction
        transaction = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'action': 'buy',
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'value': cost
        }
        self.portfolio['history'].append(transaction)
        
        self.logger.info(f"Bought {shares} shares of {ticker} at ${price:.2f}")
        return {
            'success': True,
            'message': f"Bought {shares} shares of {ticker} at ${price:.2f}",
            'transaction': transaction
        }
    
    def sell(self, ticker: str, shares: float, price: float) -> Dict:
        """
        Execute a sell order
        
        Args:
            ticker: Stock ticker
            shares: Number of shares to sell
            price: Selling price per share
            
        Returns:
            Transaction result
        """
        # Check if position exists
        if ticker not in self.portfolio['holdings']:
            return {
                'success': False,
                'message': f"No position in {ticker} to sell"
            }
        
        current_shares = self.portfolio['holdings'][ticker]['shares']
        
        # Check if enough shares
        if shares > current_shares:
            return {
                'success': False,
                'message': f"Insufficient shares. Requested: {shares}, Available: {current_shares}"
            }
        
        # Calculate proceeds and profit/loss
        proceeds = shares * price
        avg_price = self.portfolio['holdings'][ticker]['avg_price']
        cost_basis = shares * avg_price
        profit_loss = proceeds - cost_basis
        
        # Execute transaction
        self.portfolio['cash'] += proceeds
        
        # Update holdings
        if shares == current_shares:
            # Remove position completely
            del self.portfolio['holdings'][ticker]
        else:
            # Reduce position
            self.portfolio['holdings'][ticker]['shares'] -= shares
        
        # Log transaction
        transaction = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'action': 'sell',
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'value': proceeds,
            'profit_loss': profit_loss
        }
        self.portfolio['history'].append(transaction)
        
        self.logger.info(f"Sold {shares} shares of {ticker} at ${price:.2f}, P/L: ${profit_loss:.2f}")
        return {
            'success': True,
            'message': f"Sold {shares} shares of {ticker} at ${price:.2f}, P/L: ${profit_loss:.2f}",
            'transaction': transaction
        }
    
    def update_performance(self, current_prices: Dict[str, float]) -> None:
        """Update portfolio performance metrics"""
        current_value = self.get_portfolio_value(current_prices)
        starting_value = self.portfolio['performance']['starting_value']
        
        self.portfolio['performance'].update({
            'current_value': current_value,
            'roi': (current_value - starting_value) / starting_value if starting_value > 0 else 0,
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        })
    
    def check_stop_loss(self, ticker: str, current_price: float) -> bool:
        """Check if stop loss has been triggered"""
        if ticker not in self.portfolio['holdings']:
            return False
            
        position = self.portfolio['holdings'][ticker]
        avg_price = position['avg_price']
        loss_pct = (avg_price - current_price) / avg_price
        
        stop_loss = self.risk_models[self.portfolio['risk_tolerance']]['stop_loss']
        
        return loss_pct > stop_loss
    
    def check_take_profit(self, ticker: str, current_price: float) -> bool:
        """Check if take profit has been triggered"""
        if ticker not in self.portfolio['holdings']:
            return False
            
        position = self.portfolio['holdings'][ticker]
        avg_price = position['avg_price']
        gain_pct = (current_price - avg_price) / avg_price
        
        take_profit = self.risk_models[self.portfolio['risk_tolerance']]['take_profit']
        
        return gain_pct > take_profit
    
    def calculate_max_shares_to_buy(self, ticker: str, price: float, current_prices: Dict[str, float]) -> int:
        """Calculate maximum shares to buy based on risk model"""
        total_value = self.get_portfolio_value(current_prices)
        max_position_value = total_value * self.risk_models[self.portfolio['risk_tolerance']]['max_position_size']
        
        # Account for existing position
        existing_value = 0
        if ticker in self.portfolio['holdings']:
            existing_value = self.portfolio['holdings'][ticker]['shares'] * price
            
        remaining_allocation = max_position_value - existing_value
        
        # Calculate max shares based on remaining allocation and available cash
        max_shares_by_allocation = remaining_allocation / price if price > 0 else 0
        max_shares_by_cash = self.portfolio['cash'] / price if price > 0 else 0
        
        return max(0, min(int(max_shares_by_allocation), int(max_shares_by_cash)))
    
    def get_tax_loss_harvesting_opportunities(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Identify tax loss harvesting opportunities"""
        opportunities = []
        
        for ticker, position in self.portfolio['holdings'].items():
            if ticker in current_prices:
                avg_price = position['avg_price']
                current_price = current_prices[ticker]
                loss_pct = (avg_price - current_price) / avg_price
                
                # If loss is significant (over 10%)
                if loss_pct > 0.1:
                    purchase_date = datetime.strptime(position['purchase_date'], '%Y-%m-%d')
                    days_held = (datetime.now() - purchase_date).days
                    
                    # Only consider positions held for more than 30 days
                    if days_held > 30:
                        opportunities.append({
                            'ticker': ticker,
                            'shares': position['shares'],
                            'avg_price': avg_price,
                            'current_price': current_price,
                            'loss_pct': loss_pct,
                            'days_held': days_held
                        })
        
        return opportunities
    
    def evaluate_position(self, ticker: str, current_prices: Dict[str, float]) -> Dict:
        """
        Evaluate a position and recommend action
        
        Args:
            ticker: Stock ticker
            current_prices: Dictionary of current prices for all holdings
            
        Returns:
            Recommendation dictionary
        """
        if not current_prices or ticker not in current_prices:
            return {
                'action': 'hold',
                'confidence': 0.5,
                'reasoning': "Missing price data"
            }
        
        current_price = current_prices[ticker]
        total_value = self.get_portfolio_value(current_prices)
        risk_model = self.risk_models[self.portfolio['risk_tolerance']]
        
        # Get position details
        position_info = self.get_position_info(ticker, current_price)
        position_value = position_info['market_value']
        position_ratio = position_value / total_value if total_value > 0 else 0
        
        # Get sector exposure
        sector_exposure = self.get_sector_exposure(current_prices)
        ticker_sector = self.sector_mappings.get(ticker, self.sector_mappings['DEFAULT'])
        sector_allocation = sector_exposure.get(ticker_sector, 0)
        
        # Check cash position
        cash_ratio = self.portfolio['cash'] / total_value if total_value > 0 else 1
        
        # Initialize reasoning components
        reasons = []
        action = 'hold'  # Default action
        confidence = 0.5  # Default confidence
        
        # Check stop loss
        if self.check_stop_loss(ticker, current_price):
            action = 'sell'
            confidence = 0.9
            reasons.append(f"Stop loss triggered (loss exceeds {risk_model['stop_loss']*100:.0f}%)")
        
        # Check take profit
        elif self.check_take_profit(ticker, current_price):
            action = 'sell'
            confidence = 0.8
            reasons.append(f"Take profit triggered (gain exceeds {risk_model['take_profit']*100:.0f}%)")
        
        # Check position size
        elif position_ratio > risk_model['max_position_size']:
            action = 'sell'
            confidence = 0.7
            reasons.append(f"Position size ({position_ratio*100:.1f}%) exceeds maximum ({risk_model['max_position_size']*100:.0f}%)")
        
        # Check sector exposure
        elif sector_allocation > risk_model['max_sector_exposure']:
            action = 'sell'
            confidence = 0.6
            reasons.append(f"Sector exposure ({sector_allocation*100:.1f}%) exceeds maximum ({risk_model['max_sector_exposure']*100:.0f}%)")
        
        # Check if we should buy
        elif cash_ratio > risk_model['target_cash'] + risk_model['rebalance_threshold']:
            # We have excess cash
            if position_ratio < risk_model['max_position_size'] - 0.02:  # Room to grow
                action = 'buy'
                confidence = 0.6
                reasons.append(f"Excess cash ({cash_ratio*100:.1f}% vs target {risk_model['target_cash']*100:.0f}%) and position below maximum")
        
        # If no strong signals, maintain current position
        else:
            action = 'hold'
            confidence = 0.5
        # Add position-specific reasoning
        if ticker in self.portfolio['holdings']:
            # For existing positions
            if position_info['profit_loss_pct'] > 0:
                reasons.append(f"Current gain: {position_info['profit_loss_pct']*100:.1f}%")
            else:
                reasons.append(f"Current loss: {-position_info['profit_loss_pct']*100:.1f}%")
            
            reasons.append(f"Holding for {position_info['days_held']} days")
        else:
            # For potential new positions
            max_shares = self.calculate_max_shares_to_buy(ticker, current_price, current_prices)
            if max_shares > 0:
                reasons.append(f"Can buy up to {max_shares} shares with available cash")
            else:
                reasons.append("Insufficient cash for new position")
        
        # Format reasoning
        reasoning = "; ".join(reasons)
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'position_details': position_info,
            'portfolio_metrics': {
                'total_value': total_value,
                'cash_ratio': cash_ratio,
                'position_ratio': position_ratio,
                'sector_allocation': sector_allocation
            }
        }

    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """
        Generate a comprehensive portfolio summary
        
        Args:
            current_prices: Dictionary of current prices
            
        Returns:
            Portfolio summary dictionary
        """
        total_value = self.get_portfolio_value(current_prices)
        
        # Calculate position values and allocations
        positions = []
        for ticker, position in self.portfolio['holdings'].items():
            if ticker in current_prices:
                price = current_prices[ticker]
                position_info = self.get_position_info(ticker, price)
                
                positions.append({
                    'ticker': ticker,
                    'shares': position['shares'],
                    'avg_price': position['avg_price'],
                    'current_price': price,
                    'market_value': position_info['market_value'],
                    'allocation': position_info['market_value'] / total_value if total_value > 0 else 0,
                    'profit_loss': position_info['profit_loss'],
                    'profit_loss_pct': position_info['profit_loss_pct'],
                    'sector': self.sector_mappings.get(ticker, self.sector_mappings['DEFAULT'])
                })
        
        # Sort by allocation (descending)
        positions.sort(key=lambda x: x['market_value'], reverse=True)
        
        # Calculate sector allocations
        sector_exposure = self.get_sector_exposure(current_prices)
        
        # Calculate performance metrics
        self.update_performance(current_prices)
        performance = self.portfolio['performance']
        
        # Find tax loss harvesting opportunities
        tax_loss_opportunities = self.get_tax_loss_harvesting_opportunities(current_prices)
        
        # Calculate diversification score (0-100)
        sector_count = len([s for s, v in sector_exposure.items() if v > 0.05 and s != 'Cash'])
        position_count = len(positions)
        max_allocation = max([p['allocation'] for p in positions]) if positions else 0
        
        diversification_score = min(100, (
            (min(position_count, 10) / 10) * 40 +  # Number of positions (max 10)
            (min(sector_count, 5) / 5) * 40 +      # Number of sectors (max 5)
            (1 - max(0, max_allocation - 0.1)) * 20  # Penalize for concentrated positions
        ))
        
        return {
            'total_value': total_value,
            'cash': self.portfolio['cash'],
            'cash_allocation': self.portfolio['cash'] / total_value if total_value > 0 else 1,
            'positions': positions,
            'position_count': len(positions),
            'sector_allocation': sector_exposure,
            'diversification_score': diversification_score,
            'risk_profile': self.portfolio['risk_tolerance'],
            'performance': performance,
            'tax_loss_opportunities': tax_loss_opportunities
        }

    def rebalance_recommendations(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Generate portfolio rebalancing recommendations
        
        Args:
            current_prices: Dictionary of current prices
            
        Returns:
            List of rebalancing recommendations
        """
        recommendations = []
        risk_model = self.risk_models[self.portfolio['risk_tolerance']]
        total_value = self.get_portfolio_value(current_prices)
        
        # Check cash position
        cash_ratio = self.portfolio['cash'] / total_value if total_value > 0 else 1
        target_cash = risk_model['target_cash']
        
        if abs(cash_ratio - target_cash) > risk_model['rebalance_threshold']:
            if cash_ratio > target_cash:
                # Too much cash
                excess_cash = self.portfolio['cash'] - (total_value * target_cash)
                recommendations.append({
                    'action': 'deploy_cash',
                    'amount': excess_cash,
                    'reasoning': f"Cash allocation ({cash_ratio*100:.1f}%) exceeds target ({target_cash*100:.0f}%)"
                })
            else:
                # Too little cash
                cash_needed = (total_value * target_cash) - self.portfolio['cash']
                recommendations.append({
                    'action': 'raise_cash',
                    'amount': cash_needed,
                    'reasoning': f"Cash allocation ({cash_ratio*100:.1f}%) below target ({target_cash*100:.0f}%)"
                })
        
        # Check position sizes
        for ticker, position in self.portfolio['holdings'].items():
            if ticker in current_prices:
                price = current_prices[ticker]
                position_value = position['shares'] * price
                position_ratio = position_value / total_value
                
                if position_ratio > risk_model['max_position_size'] + risk_model['rebalance_threshold']:
                    # Position too large
                    excess_value = position_value - (total_value * risk_model['max_position_size'])
                    shares_to_sell = int(excess_value / price) if price > 0 else 0
                    
                    if shares_to_sell > 0:
                        recommendations.append({
                            'action': 'reduce_position',
                            'ticker': ticker,
                            'shares': shares_to_sell,
                            'value': shares_to_sell * price,
                            'reasoning': f"{ticker} position ({position_ratio*100:.1f}%) exceeds maximum ({risk_model['max_position_size']*100:.0f}%)"
                        })
        
        # Check sector exposures
        sector_exposure = self.get_sector_exposure(current_prices)
        
        for sector, allocation in sector_exposure.items():
            if sector != 'Cash' and allocation > risk_model['max_sector_exposure'] + risk_model['rebalance_threshold']:
                # Sector too concentrated
                recommendations.append({
                    'action': 'reduce_sector',
                    'sector': sector,
                    'current_allocation': allocation,
                    'target_allocation': risk_model['max_sector_exposure'],
                    'reasoning': f"{sector} allocation ({allocation*100:.1f}%) exceeds maximum ({risk_model['max_sector_exposure']*100:.0f}%)"
                })
        
        return recommendations

    def simulate_transaction(self, action: str, ticker: str, shares: float, price: float) -> Dict:
        """
        Simulate a transaction without actually executing it
        
        Args:
            action: 'buy' or 'sell'
            ticker: Stock ticker
            shares: Number of shares
            price: Transaction price
            
        Returns:
            Simulated result dictionary
        """
        # Create a copy of the portfolio
        import copy
        portfolio_copy = copy.deepcopy(self.portfolio)
        
        # Temporarily replace the actual portfolio with the copy
        original_portfolio = self.portfolio
        self.portfolio = portfolio_copy
        
        # Execute the simulated transaction
        if action.lower() == 'buy':
            result = self.buy(ticker, shares, price)
        elif action.lower() == 'sell':
            result = self.sell(ticker, shares, price)
        else:
            result = {'success': False, 'message': f"Invalid action: {action}"}
        
        # Restore the original portfolio
        self.portfolio = original_portfolio
        
        # Add simulation flag
        result['simulated'] = True
        
        return result

    def save_portfolio(self, filename: str) -> bool:
        """Save portfolio to file"""
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(self.portfolio, f, indent=4)
            return True
        except Exception as e:
            self.logger.error(f"Error saving portfolio: {str(e)}")
            return False

    def load_portfolio(self, filename: str) -> bool:
        """Load portfolio from file"""
        try:
            import json
            with open(filename, 'r') as f:
                self.portfolio = json.load(f)
            return True
        except Exception as e:
            self.logger.error(f"Error loading portfolio: {str(e)}")
            return False

    def get_recommended_position_size(self, ticker: str, current_price: float, 
                                    volatility: float, current_prices: Dict[str, float]) -> Dict:
        """
        Calculate recommended position size based on volatility and risk tolerance
        
        Args:
            ticker: Stock ticker
            current_price: Current price
            volatility: Stock volatility (standard deviation of returns)
            current_prices: Dictionary of current prices
            
        Returns:
            Position size recommendation
        """
        total_value = self.get_portfolio_value(current_prices)
        risk_model = self.risk_models[self.portfolio['risk_tolerance']]
        
        # Base position size from risk model
        base_size = risk_model['max_position_size']
        
        # Adjust for volatility (reduce size for higher volatility)
        # Assuming volatility is annualized standard deviation
        volatility_factor = 1.0
        if volatility > 0.4:  # Very high volatility (>40%)
            volatility_factor = 0.5
        elif volatility > 0.3:  # High volatility (30-40%)
            volatility_factor = 0.7
        elif volatility > 0.2:  # Moderate volatility (20-30%)
            volatility_factor = 0.9
        
        # Calculate adjusted position size
        adjusted_size = base_size * volatility_factor
        
        # Calculate dollar amount and shares
        position_value = total_value * adjusted_size
        shares = int(position_value / current_price) if current_price > 0 else 0
        
        return {
            'recommended_allocation': adjusted_size,
            'dollar_amount': position_value,
            'shares': shares,
            'reasoning': f"Base allocation ({base_size*100:.0f}%) adjusted for volatility ({volatility*100:.1f}%)"
        }
