from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import os
import numpy as np
from datasets import Dataset
import logging
from typing import List, Dict, Union, Tuple, Optional
import re

class NewsSentimentAgent:
    """
    Agent for analyzing sentiment in financial news headlines using FinBERT.
    
    This agent can:
    1. Analyze sentiment of headlines (positive, negative, neutral)
    2. Filter headlines relevant to specific tickers
    3. Provide confidence scores and explanations
    4. Fine-tune on custom financial datasets
    """
    
    def __init__(self, model_path: str = "ProsusAI/finbert", use_cached: bool = True):
        """
        Initialize the News Sentiment Agent
        
        Args:
            model_path: Path to pretrained or fine-tuned model
            use_cached: Whether to use cached models
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Set up model paths
        self.model_dir = "models/finbert_finetuned"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load model and tokenizer
        try:
            if use_cached and os.path.exists(f"{self.model_dir}/pytorch_model.bin"):
                logging.info("Loading model from local cache")
                self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
                self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
            else:
                logging.info(f"Loading model from {model_path}")
                self.tokenizer = BertTokenizer.from_pretrained(model_path)
                self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
            
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode by default
            
            # Define label mapping
            self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
            self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
            
            # Financial keyword boosters
            self.positive_keywords = [
                "beat", "exceed", "growth", "profit", "surge", "rally", "bullish", 
                "upgrade", "innovation", "partnership", "launch", "dividend", "expansion"
            ]
            
            self.negative_keywords = [
                "miss", "decline", "drop", "loss", "bearish", "downgrade", "lawsuit", 
                "investigation", "recall", "debt", "bankruptcy", "layoff", "delay"
            ]
            
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Failed to initialize NewsSentimentAgent: {str(e)}")
    
    def train(self, 
              train_texts: List[str], 
              train_labels: List[int], 
              test_texts: List[str], 
              test_labels: List[int],
              epochs: int = 3,
              batch_size: int = 16,
              learning_rate: float = 2e-5):
        """
        Fine-tune the model on custom financial news data
        
        Args:
            train_texts: List of training headlines
            train_labels: List of training labels (0=negative, 1=neutral, 2=positive)
            test_texts: List of test headlines
            test_labels: List of test labels
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
        
        Returns:
            Dict containing training metrics
        """
        logging.info(f"Fine-tuning model on {len(train_texts)} examples")
        
        # Set model to training mode
        self.model.train()
        
        # Encode datasets
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True)
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })
        
        test_dataset = Dataset.from_dict({
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask'],
            'labels': test_labels
        })
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.model_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir='logs',
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            push_to_hub=False,
            report_to="none"  # Disable wandb/tensorboard reporting
        )
        
        # Define metrics computation function
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            accuracy = (preds == labels).mean()
            return {'accuracy': accuracy}
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train model
        try:
            train_result = trainer.train()
            metrics = train_result.metrics
            
            # Evaluate model
            eval_result = trainer.evaluate()
            metrics.update(eval_result)
            
            # Save model
            trainer.save_model(self.model_dir)
            self.tokenizer.save_pretrained(self.model_dir)
            
            # Set back to evaluation mode
            self.model.eval()
            
            logging.info(f"Training complete. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            return {"error": str(e)}
    
    def save_model(self, path: Optional[str] = None):
        """Save the model and tokenizer"""
        save_path = path or self.model_dir
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logging.info(f"Model saved to {save_path}")
    
    def filter_relevant_headlines(self, ticker: str, headlines: List[str]) -> List[str]:
        """
        Filter headlines relevant to a specific ticker
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            headlines: List of news headlines
            
        Returns:
            List of relevant headlines
        """
        ticker = ticker.upper()
        relevant = []
        
        # Company name mappings (could be expanded)
        company_names = {
            'AAPL': ['apple', 'iphone', 'ipad', 'mac', 'tim cook'],
            'MSFT': ['microsoft', 'windows', 'azure', 'satya nadella'],
            'GOOGL': ['google', 'alphabet', 'android', 'sundar pichai'],
            'AMZN': ['amazon', 'aws', 'jeff bezos', 'andy jassy'],
            'TSLA': ['tesla', 'elon musk', 'electric vehicle', 'ev'],
            'META': ['facebook', 'instagram', 'meta', 'zuckerberg'],
            'NFLX': ['netflix', 'streaming'],
            'NVDA': ['nvidia', 'gpu', 'jensen huang'],
        }
        
        # Get company keywords
        company_keywords = company_names.get(ticker, [ticker.lower()])
        
        for headline in headlines:
            headline_lower = headline.lower()
            
            # Check for ticker symbol
            if ticker.lower() in headline_lower:
                relevant.append(headline)
                continue
                
            # Check for company name and related terms
            for keyword in company_keywords:
                if keyword in headline_lower:
                    relevant.append(headline)
                    break
        
        return relevant
    
    def predict_single(self, headline: str) -> Dict[str, Union[str, float]]:
        """
        Predict sentiment for a single headline
        
        Args:
            headline: News headline text
            
        Returns:
            Dict with sentiment prediction and confidence
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(headline, return_tensors="pt", truncation=True, padding=True).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get probabilities with softmax
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            prediction = torch.argmax(probs).item()
            
            # Get label and confidence
            label = self.id2label[prediction]
            confidence = probs[prediction].item()
            
            # Get keyword matches for explanation
            explanation = self._generate_explanation(headline, label)
            
            return {
                "sentiment": label,
                "confidence": confidence,
                "explanation": explanation
            }
            
        except Exception as e:
            logging.error(f"Error predicting sentiment: {str(e)}")
            return {
                "sentiment": "neutral",
                "confidence": 0.33,
                "explanation": f"Error during prediction: {str(e)}"
            }
    
    def _generate_explanation(self, headline: str, sentiment: str) -> str:
        """Generate explanation for sentiment prediction based on keywords"""
        headline_lower = headline.lower()
        
        # Count keyword matches
        pos_matches = [word for word in self.positive_keywords if word in headline_lower]
        neg_matches = [word for word in self.negative_keywords if word in headline_lower]
        
        # Generate explanation
        if sentiment == "positive" and pos_matches:
            return f"Positive sentiment detected with keywords: {', '.join(pos_matches)}"
        elif sentiment == "negative" and neg_matches:
            return f"Negative sentiment detected with keywords: {', '.join(neg_matches)}"
        elif sentiment == "neutral":
            if pos_matches and neg_matches:
                return f"Mixed signals with both positive and negative keywords"
            else:
                return "No strong sentiment indicators detected"
        else:
            return f"Model predicted {sentiment} sentiment"
    
    def analyze_headlines(self, ticker: str, headlines: List[str]) -> Dict[str, Union[str, float, List]]:
        """
        Analyze sentiment for headlines related to a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            headlines: List of news headlines
            
        Returns:
            Dict with aggregated sentiment analysis
        """
        if not headlines:
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "reasoning": "No headlines provided for analysis",
                "headline_count": 0,
                "details": []
            }
        
        # Filter relevant headlines
        relevant_headlines = self.filter_relevant_headlines(ticker, headlines)
        
        if not relevant_headlines:
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "reasoning": f"No headlines relevant to {ticker} found",
                "headline_count": 0,
                "details": []
            }
        
        # Analyze each headline
        results = []
        for headline in relevant_headlines:
            result = self.predict_single(headline)
            results.append({
                "headline": headline,
                "sentiment": result["sentiment"],
                "confidence": result["confidence"]
            })
        
        # Calculate aggregate sentiment
        sentiment_scores = {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }
        
        for result in results:
            sentiment_scores[result["sentiment"]] += result["confidence"]
        
        # Determine overall sentiment
        max_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        total_confidence = sum(sentiment_scores.values())
        
        if total_confidence > 0:
            confidence = sentiment_scores[max_sentiment] / total_confidence
        else:
            confidence = 0.33
            max_sentiment = "neutral"
        
        # Generate reasoning
        pos_count = sum(1 for r in results if r["sentiment"] == "positive")
        neg_count = sum(1 for r in results if r["sentiment"] == "negative")
        neu_count = sum(1 for r in results if r["sentiment"] == "neutral")
        
        reasoning = f"Analysis of {len(relevant_headlines)} headlines for {ticker}: "
        reasoning += f"{pos_count} positive, {neg_count} negative, {neu_count} neutral. "
        
        if max_sentiment == "positive":
            reasoning += f"Overall positive sentiment with {confidence:.0%} confidence."
        elif max_sentiment == "negative":
            reasoning += f"Overall negative sentiment with {confidence:.0%} confidence."
        else:
            reasoning += f"Overall neutral sentiment with {confidence:.0%} confidence."
        
        return {
            "sentiment": max_sentiment,
            "confidence": confidence,
            "reasoning": reasoning,
            "headline_count": len(relevant_headlines),
            "details": results
        }
    
    def batch_predict(self, headlines: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Batch prediction for multiple headlines
        
        Args:
            headlines: List of headlines to analyze
            
        Returns:
            List of prediction results
        """
        # Tokenize all headlines at once
        inputs = self.tokenizer(headlines, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process results
        results = []
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        
        for i, (pred, prob) in enumerate(zip(predictions, probs)):
            label = self.id2label[pred.item()]
            confidence = prob[pred].item()
            explanation = self._generate_explanation(headlines[i], label)
            
            results.append({
                "headline": headlines[i],
                "sentiment": label,
                "confidence": confidence,
                "explanation": explanation
            })
        
        return results
