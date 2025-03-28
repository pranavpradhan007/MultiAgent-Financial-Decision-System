import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.news_sentiment import NewsSentimentAgent

def test_sentiment_analysis():
    agent = NewsSentimentAgent()
    
    test_cases = [
        {"headline": "Company X reports record profits", "expected": "positive"},
        {"headline": "Company X stock plummets after missed earnings", "expected": "negative"},
        {"headline": "Company X announces new product line", "expected": "neutral"}
    ]
    
    print("Sentiment Analysis Test Results:")
    print("-" * 50)
    
    correct = 0
    for case in test_cases:
        result = agent.predict_single(case["headline"])
        is_correct = result["sentiment"] == case["expected"]
        
        if is_correct:
            correct += 1
            
        print(f"Headline: {case['headline']}")
        print(f"Expected: {case['expected']}, Got: {result['sentiment']} (Confidence: {result['confidence']:.2f})")
        print(f"Correct: {'✓' if is_correct else '✗'}")
        print("-" * 50)
    
    print(f"Overall accuracy: {correct/len(test_cases):.0%}")

if __name__ == "__main__":
    test_sentiment_analysis()
