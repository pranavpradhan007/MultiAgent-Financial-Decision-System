# In visualization/charts.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplfinance as mpf
import seaborn as sns
from matplotlib.dates import DateFormatter
import os

# Create directory for saving charts
os.makedirs('charts', exist_ok=True)

def plot_technical_indicators(ticker, market_data, save_path=None):
    """Plot technical indicators used in market prediction"""
    # Convert to DataFrame if it's not already
    if not isinstance(market_data, pd.DataFrame):
        market_data = pd.DataFrame(market_data)
    
    # Ensure date column is datetime
    if 'date' in market_data.columns:
        market_data['date'] = pd.to_datetime(market_data['date'])
        market_data.set_index('date', inplace=True)
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Plot 1: Price and Moving Averages
    axs[0].plot(market_data.index, market_data['price'], label='Price', color='blue')
    if 'sma_5' in market_data.columns:
        axs[0].plot(market_data.index, market_data['sma_5'], label='5-day MA', color='red')
    if 'sma_20' in market_data.columns:
        axs[0].plot(market_data.index, market_data['sma_20'], label='20-day MA', color='green')
    axs[0].set_title(f'{ticker} Price and Moving Averages')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: RSI
    if 'rsi' in market_data.columns:
        axs[1].plot(market_data.index, market_data['rsi'], color='purple')
        axs[1].axhline(y=70, color='r', linestyle='-', alpha=0.3)
        axs[1].axhline(y=30, color='g', linestyle='-', alpha=0.3)
        axs[1].set_title('RSI (Relative Strength Index)')
        axs[1].set_ylim(0, 100)
        axs[1].grid(True)
    
    # Plot 3: MACD
    if 'macd' in market_data.columns:
        axs[2].plot(market_data.index, market_data['macd'], label='MACD', color='blue')
        if 'signal' in market_data.columns:
            axs[2].plot(market_data.index, market_data['signal'], label='Signal', color='red')
        axs[2].bar(market_data.index, market_data['macd'] - market_data.get('signal', 0), 
                  label='Histogram', color='gray', alpha=0.3)
        axs[2].set_title('MACD (Moving Average Convergence Divergence)')
        axs[2].legend()
        axs[2].grid(True)
    
    # Format x-axis
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        return save_path
    else:
        plt.show()
        return None

def plot_sentiment_analysis(ticker, headlines, sentiments, save_path=None):
    """Plot sentiment analysis results"""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count sentiments
    sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    for sentiment in sentiments:
        sentiment_counts[sentiment] += 1
    
    # Plot 1: Pie chart of sentiment distribution
    ax1.pie(
        sentiment_counts.values(),
        labels=sentiment_counts.keys(),
        autopct='%1.1f%%',
        colors=['green', 'gray', 'red'],
        explode=(0.1, 0, 0.1)
    )
    ax1.set_title(f'Sentiment Distribution for {ticker}')
    
    # Plot 2: Bar chart of headlines and sentiments
    colors = ['green' if s == 'positive' else 'red' if s == 'negative' else 'gray' for s in sentiments]
    
    # Truncate headlines for display
    short_headlines = [h[:40] + '...' if len(h) > 40 else h for h in headlines]
    
    # Create bars
    y_pos = np.arange(len(headlines))
    ax2.barh(y_pos, [1] * len(headlines), color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(short_headlines)
    ax2.set_title('Headlines Sentiment')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Positive'),
        Patch(facecolor='gray', label='Neutral'),
        Patch(facecolor='red', label='Negative')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        return save_path
    else:
        plt.show()
        return None

def plot_portfolio_allocation(portfolio_summary, save_path=None):
    """Plot portfolio allocation"""
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot 1: Asset allocation pie chart
    positions = portfolio_summary.get('positions', [])
    if positions:
        labels = [p['ticker'] for p in positions]
        sizes = [p['market_value'] for p in positions]
        
        # Add cash
        labels.append('Cash')
        sizes.append(portfolio_summary.get('cash', 0))
        
        ax1.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90
        )
        ax1.set_title('Portfolio Allocation')
    else:
        ax1.text(0.5, 0.5, 'No positions in portfolio', 
                horizontalalignment='center', verticalalignment='center')
    
    # Plot 2: Sector allocation
    sector_allocation = portfolio_summary.get('sector_allocation', {})
    if sector_allocation:
        sectors = list(sector_allocation.keys())
        allocations = list(sector_allocation.values())
        
        ax2.bar(sectors, allocations)
        ax2.set_ylabel('Allocation (%)')
        ax2.set_title('Sector Allocation')
        plt.xticks(rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No sector data available', 
                horizontalalignment='center', verticalalignment='center')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        return save_path
    else:
        plt.show()
        return None

def plot_decision_history(decisions, save_path=None):
    """Plot decision history"""
    # Extract data
    dates = [d['timestamp'][:10] for d in decisions]
    actions = [d.get('action', 'hold') for d in decisions]  # Use get with default 'hold'
    confidences = [d.get('confidence', 0.5) for d in decisions]  # Use get with default 0.5
    
    # Create color map
    color_map = {'buy': 'green', 'sell': 'red', 'hold': 'blue'}
    colors = [color_map.get(action, 'gray') for action in actions]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot decisions
    ax.scatter(dates, confidences, c=colors, s=100, alpha=0.7)
    
    # Connect points with lines
    ax.plot(dates, confidences, 'k--', alpha=0.3)
    
    # Add labels
    for i, action in enumerate(actions):
        ax.annotate(action, (dates[i], confidences[i]), 
                   xytext=(5, 5), textcoords='offset points')
    
    # Customize plot
    ax.set_xlabel('Date')
    ax.set_ylabel('Confidence')
    ax.set_title('Decision History')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Buy', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Sell', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Hold', markersize=10)
    ]
    ax.legend(handles=legend_elements)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        return save_path
    else:
        plt.show()
        return None
