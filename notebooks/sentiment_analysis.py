# In notebooks/sentiment_analysis.py
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.news_sentiment import NewsSentimentAgent

def visualize_sentiment_analysis(headlines):
    agent = NewsSentimentAgent()
    results = []
    
    for headline in headlines:
        result = agent.predict_single(headline)
        results.append({
            "headline": headline,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"]
        })
    
    # Count sentiments
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    for result in results:
        sentiments[result["sentiment"]] += 1
    
    # Create pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(
        sentiments.values(), 
        labels=sentiments.keys(),
        autopct='%1.1f%%',
        colors=['green', 'gray', 'red']
    )
    plt.title('Sentiment Distribution')
    plt.savefig('sentiment_distribution.png')
    
    # Create bar chart of confidence by headline
    plt.figure(figsize=(12, 8))
    plt.barh(
        [r["headline"][:50] + "..." for r in results],
        [r["confidence"] for r in results],
        color=[
            'green' if r["sentiment"] == "positive" else
            'red' if r["sentiment"] == "negative" else 'gray'
            for r in results
        ]
    )
    plt.xlabel('Confidence')
    plt.ylabel('Headline')
    plt.title('Sentiment Analysis Confidence by Headline')
    plt.tight_layout()
    plt.savefig('sentiment_confidence.png')
    
    # Print detailed results
    for result in results:
        print(f"Headline: {result['headline']}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f})")
        print("-" * 50)

# Test with sample headlines
headlines = [
    "Tesla reports record quarterly deliveries, beating expectations",
    "Tesla stock drops after CEO Elon Musk sells shares",
    "Tesla announces new factory in Europe to boost production",
    "Analysts downgrade Tesla citing increased competition",
    "Tesla unveils new battery technology with longer range"
]

visualize_sentiment_analysis(headlines)
