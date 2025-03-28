import numpy as np
import logging
from typing import Dict, List, Union, Tuple, Any, Optional
import json
from datetime import datetime
import asyncio

class EnhancedDecisionCoordinator:
    """
    Advanced coordinator for financial decision-making that integrates multiple specialized agents.
    
    Features:
    - Confidence-weighted decision making
    - Adaptive agent weighting based on historical performance
    - Asynchronous agent execution
    - Comprehensive explanation generation
    - Decision overrides for critical scenarios
    - Historical decision tracking
    """
    
    def __init__(self, 
                 news_agent=None, 
                 market_agent=None, 
                 portfolio_agent=None,
                 initial_weights: Dict[str, float] = None,
                 decision_history_size: int = 100):
        """
        Initialize the coordinator with specialized agents
        
        Args:
            news_agent: Agent for analyzing news sentiment
            market_agent: Agent for market prediction
            portfolio_agent: Agent for portfolio management
            initial_weights: Initial weights for each agent
            decision_history_size: Number of past decisions to track
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DecisionCoordinator")
        
        # Initialize agents
        self.news_agent = news_agent
        self.market_agent = market_agent
        self.portfolio_agent = portfolio_agent
        
        # Initialize weights
        self.weights = initial_weights or {
            'news': 0.35,
            'market': 0.40,
            'portfolio': 0.25
        }
        
        # Validate weights
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            self.logger.warning(f"Weights don't sum to 1.0 (sum: {total_weight}). Normalizing.")
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # Decision history for performance tracking
        self.decision_history = []
        self.decision_history_size = decision_history_size
        
        # Agent performance metrics
        self.agent_performance = {
            'news': {'correct': 0, 'total': 0},
            'market': {'correct': 0, 'total': 0},
            'portfolio': {'correct': 0, 'total': 0}
        }
        
        # Critical override rules
        self.override_rules = [
            {'name': 'stop_loss', 'priority': 10},
            {'name': 'take_profit', 'priority': 9},
            {'name': 'market_crash', 'priority': 8},
            {'name': 'breaking_news', 'priority': 7}
        ]
    
    async def get_news_analysis(self, ticker: str, headlines: List[str]) -> Dict:
        """Asynchronously get news sentiment analysis"""
        try:
            if not headlines:
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.5,
                    'reasoning': 'No headlines provided'
                }
            
            # Use the analyze_headlines method for comprehensive analysis
            if hasattr(self.news_agent, 'analyze_headlines'):
                result = self.news_agent.analyze_headlines(ticker, headlines)
            # Fall back to simple prediction if analyze_headlines is not available
            elif hasattr(self.news_agent, 'predict'):
                # Convert prediction to standardized format
                sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                prediction = self.news_agent.predict(headlines[0])
                
                if isinstance(prediction, int):
                    sentiment = sentiment_map.get(prediction, 'neutral')
                else:
                    sentiment = prediction
                
                result = {
                    'sentiment': sentiment,
                    'confidence': 0.7,  # Default confidence
                    'reasoning': f"Based on {len(headlines)} headlines"
                }
            else:
                raise AttributeError("News agent lacks required prediction methods")
            
            # Map sentiment to action
            sentiment_to_action = {
                'positive': 'buy',
                'neutral': 'hold',
                'negative': 'sell'
            }
            
            result['action'] = sentiment_to_action.get(result['sentiment'], 'hold')
            return result
            
        except Exception as e:
            self.logger.error(f"Error in news analysis: {str(e)}")
            return {
                'sentiment': 'neutral',
                'action': 'hold',
                'confidence': 0.5,
                'reasoning': f"Error in news analysis: {str(e)}"
            }
    
    async def get_market_prediction(self, ticker: str, market_data: Any) -> Dict:
        """Asynchronously get market prediction"""
        try:
            result = self.market_agent.predict(market_data)
            
            # Standardize the result format
            if isinstance(result, str):
                # Simple string result
                return {
                    'prediction': result,
                    'action': result,
                    'confidence': 0.7,
                    'reasoning': "Based on market data analysis"
                }
            elif isinstance(result, dict):
                # Already a dictionary result
                if 'prediction' in result and 'action' not in result:
                    result['action'] = result['prediction']
                elif 'action' in result and 'prediction' not in result:
                    result['prediction'] = result['action']
                
                # Ensure confidence exists
                if 'confidence' not in result:
                    result['confidence'] = 0.7
                
                return result
            else:
                raise ValueError(f"Unexpected result type from market agent: {type(result)}")
                
        except Exception as e:
            self.logger.error(f"Error in market prediction: {str(e)}")
            return {
                'prediction': 'hold',
                'action': 'hold',
                'confidence': 0.5,
                'reasoning': f"Error in market prediction: {str(e)}"
            }
    
    async def get_portfolio_recommendation(self, ticker: str, current_prices: Dict[str, float]) -> Dict:
        """Asynchronously get portfolio recommendation"""
        try:
            # Check which method signature the portfolio agent supports
            if hasattr(self.portfolio_agent, 'evaluate_position'):
                # Check if it takes a dictionary or a single price
                import inspect
                sig = inspect.signature(self.portfolio_agent.evaluate_position)
                
                if len(sig.parameters) >= 2 and 'current_prices' in sig.parameters:
                    # Takes dictionary of prices
                    result = self.portfolio_agent.evaluate_position(ticker, current_prices)
                else:
                    # Takes single price
                    result = self.portfolio_agent.evaluate_position(ticker, current_prices.get(ticker, 0))
            else:
                raise AttributeError("Portfolio agent lacks required evaluation method")
            
            # Standardize the result format
            if isinstance(result, str):
                # Simple string result
                return {
                    'recommendation': result,
                    'action': result,
                    'confidence': 0.7,
                    'reasoning': "Based on portfolio analysis"
                }
            elif isinstance(result, dict):
                # Already a dictionary result
                if 'action' not in result:
                    if 'recommendation' in result:
                        result['action'] = result['recommendation']
                    else:
                        result['action'] = 'hold'
                
                # Ensure confidence exists
                if 'confidence' not in result:
                    result['confidence'] = 0.7
                
                return result
            else:
                raise ValueError(f"Unexpected result type from portfolio agent: {type(result)}")
                
        except Exception as e:
            self.logger.error(f"Error in portfolio recommendation: {str(e)}")
            return {
                'recommendation': 'hold',
                'action': 'hold',
                'confidence': 0.5,
                'reasoning': f"Error in portfolio recommendation: {str(e)}"
            }
    
    def _check_for_overrides(self, news_result: Dict, market_result: Dict, portfolio_result: Dict) -> Optional[Dict]:
        """Check for critical scenarios that should override normal decision process"""
        # Check for stop loss in portfolio recommendation
        if portfolio_result.get('reasoning', '').lower().find('stop loss') >= 0:
            return {
                'action': 'sell',
                'confidence': 0.95,
                'override_reason': 'Stop loss triggered',
                'override_rule': 'stop_loss'
            }
        
        # Check for take profit in portfolio recommendation
        if portfolio_result.get('reasoning', '').lower().find('take profit') >= 0:
            return {
                'action': 'sell',
                'confidence': 0.9,
                'override_reason': 'Take profit triggered',
                'override_rule': 'take_profit'
            }
        
        # Check for extreme market volatility
        if market_result.get('reasoning', '').lower().find('volatility is high') >= 0:
            return {
                'action': 'hold',
                'confidence': 0.85,
                'override_reason': 'High market volatility detected',
                'override_rule': 'market_volatility'
            }
        
        # Check for breaking negative news
        if (news_result.get('sentiment') == 'negative' and 
            news_result.get('confidence', 0) > 0.8 and
            news_result.get('headline_count', 0) > 3):
            return {
                'action': 'sell',
                'confidence': 0.85,
                'override_reason': 'Breaking negative news',
                'override_rule': 'breaking_news'
            }
        
        # No override needed
        return None
    
    def _weighted_decision(self, news_result: Dict, market_result: Dict, portfolio_result: Dict) -> Dict:
        """Make a weighted decision based on agent outputs"""
        # Initialize decision matrix
        decision_matrix = {
            'buy': 0,
            'hold': 0,
            'sell': 0
        }
        
        # Get actions and confidences from each agent
        news_action = news_result.get('action', 'hold')
        news_confidence = news_result.get('confidence', 0.5)
        
        market_action = market_result.get('action', 'hold')
        market_confidence = market_result.get('confidence', 0.5)
        
        portfolio_action = portfolio_result.get('action', 'hold')
        portfolio_confidence = portfolio_result.get('confidence', 0.5)
        
        # Add weighted votes to decision matrix
        decision_matrix[news_action] += self.weights['news'] * news_confidence
        decision_matrix[market_action] += self.weights['market'] * market_confidence
        decision_matrix[portfolio_action] += self.weights['portfolio'] * portfolio_confidence
        
        # Get the action with the highest score
        final_action = max(decision_matrix, key=decision_matrix.get)
        confidence = decision_matrix[final_action]
        
        # Calculate agreement level (0-1)
        agreement_level = 0
        if news_action == market_action == portfolio_action:
            agreement_level = 1.0  # Full agreement
        elif (news_action == market_action) or (news_action == portfolio_action) or (market_action == portfolio_action):
            agreement_level = 0.5  # Partial agreement
        
        return {
            'action': final_action,
            'confidence': confidence,
            'agreement_level': agreement_level,
            'decision_matrix': decision_matrix
        }
    
    def _generate_explanation(self, 
                             ticker: str,
                             final_decision: Dict, 
                             news_result: Dict, 
                             market_result: Dict, 
                             portfolio_result: Dict) -> str:
        """Generate a comprehensive explanation for the decision"""
        action = final_decision['action']
        confidence = final_decision['confidence']
        
        # Start with the decision summary
        explanation = [
            f"RECOMMENDATION: {action.upper()} {ticker} with {confidence:.0%} confidence.",
            ""
        ]
        
        # Add override explanation if applicable
        if 'override_reason' in final_decision:
            explanation.append(f"⚠️ OVERRIDE ALERT: {final_decision['override_reason']}")
            explanation.append("")
        
        # Add agent contributions
        explanation.append("AGENT INSIGHTS:")
        
        # News sentiment
        news_sentiment = news_result.get('sentiment', 'neutral')
        news_confidence = news_result.get('confidence', 0.5)
        news_reasoning = news_result.get('reasoning', '')
        
        sentiment_emoji = {
            'positive': '📈',
            'neutral': '➖',
            'negative': '📉'
        }.get(news_sentiment, '❓')
        
        explanation.append(f"{sentiment_emoji} News Sentiment: {news_sentiment.capitalize()} ({news_confidence:.0%} confidence)")
        if news_reasoning:
            explanation.append(f"   {news_reasoning}")
        
        # Market prediction
        market_prediction = market_result.get('prediction', 'hold')
        market_confidence = market_result.get('confidence', 0.5)
        market_reasoning = market_result.get('reasoning', '')
        
        prediction_emoji = {
            'buy': '🔼',
            'hold': '◀▶',
            'sell': '🔽',
            'up': '🔼',
            'down': '🔽'
        }.get(market_prediction, '❓')
        
        explanation.append(f"{prediction_emoji} Market Prediction: {market_prediction.capitalize()} ({market_confidence:.0%} confidence)")
        if market_reasoning:
            explanation.append(f"   {market_reasoning}")
        
        # Portfolio recommendation
        portfolio_recommendation = portfolio_result.get('action', 'hold')
        portfolio_confidence = portfolio_result.get('confidence', 0.5)
        portfolio_reasoning = portfolio_result.get('reasoning', '')
        
        portfolio_emoji = {
            'buy': '💰',
            'hold': '⏹️',
            'sell': '💸'
        }.get(portfolio_recommendation, '❓')
        
        explanation.append(f"{portfolio_emoji} Portfolio Assessment: {portfolio_recommendation.capitalize()} ({portfolio_confidence:.0%} confidence)")
        if portfolio_reasoning:
            explanation.append(f"   {portfolio_reasoning}")
        
        # Add decision weights
        explanation.append("")
        explanation.append("DECISION WEIGHTS:")
        for agent, weight in self.weights.items():
            explanation.append(f"- {agent.capitalize()}: {weight:.0%}")
        
        # Add agreement level
        agreement_level = final_decision.get('agreement_level', 0)
        agreement_desc = {
            0: "No agreement between agents",
            0.5: "Partial agreement between agents",
            1.0: "Full agreement between agents"
        }.get(agreement_level, "Unknown agreement level")
        
        explanation.append("")
        explanation.append(f"AGREEMENT LEVEL: {agreement_desc}")
        
        # Join all parts with newlines
        return "\n".join(explanation)
    
    def _update_decision_history(self, ticker: str, decision: Dict) -> None:
        """Update the decision history"""
        # Add timestamp and ticker
        decision_record = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'decision': decision
        }
        
        # Add to history
        self.decision_history.append(decision_record)
        
        # Trim history if needed
        if len(self.decision_history) > self.decision_history_size:
            self.decision_history = self.decision_history[-self.decision_history_size:]
    
    def _update_agent_weights(self) -> None:
        """Update agent weights based on historical performance"""
        # Only update if we have enough history
        if len(self.decision_history) < 10:
            return
        
        # Calculate performance metrics
        for agent in self.agent_performance:
            if self.agent_performance[agent]['total'] > 0:
                accuracy = self.agent_performance[agent]['correct'] / self.agent_performance[agent]['total']
                
                # Adjust weights based on accuracy
                # This is a simple linear adjustment - could be more sophisticated
                self.weights[agent] = 0.2 + (accuracy * 0.6)  # Range from 0.2 to 0.8
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.logger.info(f"Updated agent weights: {self.weights}")
    
    async def decide(self, 
                    ticker: str, 
                    headlines: List[str], 
                    market_data: Any, 
                    current_prices: Dict[str, float]) -> Dict:
        """
        Make a coordinated financial decision
        
        Args:
            ticker: Stock ticker symbol
            headlines: List of news headlines
            market_data: Historical market data
            current_prices: Dictionary of current prices
            
        Returns:
            Decision dictionary with action, confidence, and explanation
        """
        # Run all agents concurrently
        news_task = self.get_news_analysis(ticker, headlines)
        market_task = self.get_market_prediction(ticker, market_data)
        portfolio_task = self.get_portfolio_recommendation(ticker, current_prices)
        
        # Gather results
        news_result, market_result, portfolio_result = await asyncio.gather(
            news_task, market_task, portfolio_task
        )
        
        # Check for critical overrides
        override = self._check_for_overrides(news_result, market_result, portfolio_result)
        
        if override:
            final_decision = override
            self.logger.warning(f"Decision override: {override['override_reason']}")
        else:
            # Make weighted decision
            final_decision = self._weighted_decision(news_result, market_result, portfolio_result)
        
        # Generate explanation
        explanation = self._generate_explanation(
            ticker, final_decision, news_result, market_result, portfolio_result
        )
        
        # Prepare final result
        result = {
            'ticker': ticker,
            'action': final_decision['action'],
            'confidence': final_decision['confidence'],
            'explanation': explanation,
            'timestamp': datetime.now().isoformat(),
            'agent_results': {
                'news': news_result,
                'market': market_result,
                'portfolio': portfolio_result
            }
        }
        
        # Update decision history
        self._update_decision_history(ticker, result)
        
        return result
    
    def provide_feedback(self, decision_id: str, was_correct: bool) -> None:
        """
        Provide feedback on a past decision to improve agent weighting
        
        Args:
            decision_id: ID of the decision (timestamp)
            was_correct: Whether the decision was correct
        """
        # Find the decision in history
        for decision in self.decision_history:
            if decision['timestamp'] == decision_id:
                # Update agent performance
                for agent, result in decision['decision']['agent_results'].items():
                    agent_action = result.get('action')
                    final_action = decision['decision']['action']
                    if agent_action == final_action:
                        self.agent_performance[agent]['total'] += 1
                        if was_correct:
                            self.agent_performance[agent]['correct'] += 1
            
            # Update weights based on new performance data
            self._update_agent_weights()
            
            self.logger.info(f"Feedback recorded for decision {decision_id}. Correct: {was_correct}")
            return
    
        self.logger.warning(f"Decision {decision_id} not found in history")

    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for each agent"""
        metrics = {}
        for agent, performance in self.agent_performance.items():
            if performance['total'] > 0:
                accuracy = performance['correct'] / performance['total']
                metrics[agent] = {
                    'accuracy': accuracy,
                    'total_decisions': performance['total']
                }
            else:
                metrics[agent] = {
                    'accuracy': None,
                    'total_decisions': 0
                }
        
        return metrics

    def save_state(self, filename: str) -> None:
        """Save the current state of the coordinator"""
        state = {
            'weights': self.weights,
            'decision_history': self.decision_history,
            'agent_performance': self.agent_performance
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f)
        
        self.logger.info(f"Coordinator state saved to {filename}")

    def load_state(self, filename: str) -> None:
        """Load a saved state for the coordinator"""
        with open(filename, 'r') as f:
            state = json.load(f)
        
        self.weights = state['weights']
        self.decision_history = state['decision_history']
        self.agent_performance = state['agent_performance']
        
        self.logger.info(f"Coordinator state loaded from {filename}")

    async def batch_decide(self, decisions: List[Dict]) -> List[Dict]:
        """
        Make decisions for multiple stocks in parallel
        
        Args:
            decisions: List of dictionaries, each containing 'ticker', 'headlines', 'market_data', and 'current_prices'
        
        Returns:
            List of decision results
        """
        tasks = []
        for decision in decisions:
            task = self.decide(
                decision['ticker'],
                decision['headlines'],
                decision['market_data'],
                decision['current_prices']
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

    def analyze_decision_history(self) -> Dict:
        """Analyze the decision history for patterns and insights"""
        if not self.decision_history:
            return {"message": "No decision history available"}
        
        analysis = {
            "total_decisions": len(self.decision_history),
            "action_distribution": {"buy": 0, "hold": 0, "sell": 0},
            "average_confidence": 0,
            "override_frequency": 0,
            "most_common_override": None
        }
        
        override_counts = {}
        total_confidence = 0
        
        for decision in self.decision_history:
            action = decision['decision']['action']
            analysis['action_distribution'][action] += 1
            total_confidence += decision['decision']['confidence']
            
            if 'override_reason' in decision['decision']:
                analysis['override_frequency'] += 1
                override_rule = decision['decision'].get('override_rule', 'unknown')
                override_counts[override_rule] = override_counts.get(override_rule, 0) + 1
        
        # Calculate averages and percentages
        analysis['average_confidence'] = total_confidence / len(self.decision_history)
        analysis['override_frequency'] /= len(self.decision_history)
        
        # Find most common override
        if override_counts:
            analysis['most_common_override'] = max(override_counts, key=override_counts.get)
        
        # Convert action distribution to percentages
        total_actions = sum(analysis['action_distribution'].values())
        analysis['action_distribution'] = {
            action: count / total_actions 
            for action, count in analysis['action_distribution'].items()
        }
        
        return analysis

    def get_decision_explanation(self, decision_id: str) -> Optional[str]:
        """Retrieve the explanation for a specific past decision"""
        for decision in self.decision_history:
            if decision['timestamp'] == decision_id:
                return decision['decision'].get('explanation')
        return None

    def adjust_risk_tolerance(self, new_risk_level: str) -> None:
        """
        Adjust the risk tolerance of the decision-making process
        
        Args:
            new_risk_level: 'low', 'medium', or 'high'
        """
        if new_risk_level not in ['low', 'medium', 'high']:
            raise ValueError("Risk level must be 'low', 'medium', or 'high'")
        
        # Adjust weights based on risk level
        if new_risk_level == 'low':
            self.weights['news'] *= 0.8
            self.weights['market'] *= 0.8
            self.weights['portfolio'] *= 1.4
        elif new_risk_level == 'high':
            self.weights['news'] *= 1.2
            self.weights['market'] *= 1.2
            self.weights['portfolio'] *= 0.6
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.logger.info(f"Risk tolerance adjusted to {new_risk_level}. New weights: {self.weights}")

    def simulate_decision(self, 
                        ticker: str, 
                        headlines: List[str], 
                        market_data: Any, 
                        current_prices: Dict[str, float],
                        custom_weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        Simulate a decision with optional custom weights without affecting the actual system
        
        Args:
            ticker: Stock ticker symbol
            headlines: List of news headlines
            market_data: Historical market data
            current_prices: Dictionary of current prices
            custom_weights: Optional custom weights for simulation
        
        Returns:
            Simulated decision result
        """
        # Store original weights
        original_weights = self.weights.copy()
        
        # Apply custom weights if provided
        if custom_weights:
            self.weights = custom_weights
            # Normalize custom weights
            total_weight = sum(self.weights.values())
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # Make decision
        loop = asyncio.get_event_loop()
        decision = loop.run_until_complete(self.decide(ticker, headlines, market_data, current_prices))
        
        # Restore original weights
        self.weights = original_weights
        
        # Mark as simulation
        decision['simulated'] = True
        
        return decision


