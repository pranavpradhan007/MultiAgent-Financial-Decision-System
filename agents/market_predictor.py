import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 3)  # Buy/Hold/Sell
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        return self.fc2(out)

class MarketPredictorAgent:
    def __init__(self, input_size=7, lookback=30, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMPredictor(input_size).to(self.device)
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.input_size = input_size
        
        # Load model if path provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        seed = deltas[:window+1]
        up = seed[seed >= 0].sum()/window
        down = -seed[seed < 0].sum()/window
        rs = up/down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:window] = 100. - 100./(1. + rs)
        
        for i in range(window, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
                
            up = (up * (window-1) + upval) / window
            down = (down * (window-1) + downval) / window
            rs = up/down if down != 0 else 0
            rsi[i] = 100. - 100./(1. + rs)
        
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd.values
    
    def create_sequences(self, data):
        """Create sequences for LSTM training"""
        X, y = [], []
        features = data.shape[1]  # Number of features
        
        for i in range(len(data) - self.lookback - 1):
            # Get lookback window
            window = data[i:(i + self.lookback)]
            
            # Current price and next day price for determining direction
            current_close = data[i + self.lookback - 1, 0]  # Assuming price is first column
            next_close = data[i + self.lookback, 0]
            
            # Determine label: 0=Sell, 1=Hold, 2=Buy
            if next_close > current_close * 1.01:  # 1% increase = Buy
                label = 2
            elif next_close < current_close * 0.99:  # 1% decrease = Sell
                label = 0
            else:
                label = 1  # Hold
            
            X.append(window)
            y.append(label)
            
        return np.array(X), np.array(y)
    
    def add_features(self, df):
        """Add technical indicators to the dataframe"""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Simple Moving Averages
        df['sma_5'] = df['price'].rolling(5).mean()
        df['sma_20'] = df['price'].rolling(20).mean()
        
        # Relative Strength Index
        df['rsi'] = self.calculate_rsi(df['price'].values)
        
        # MACD
        df['macd'] = self.calculate_macd(df['price'].values)
        
        # Volatility (20-day standard deviation of returns)
        df['volatility'] = df['price'].pct_change().rolling(20).std()
        
        # Price momentum (percent change)
        df['momentum'] = df['price'].pct_change(5)
        
        # Bollinger Bands
        df['bb_middle'] = df['price'].rolling(20).mean()
        df['bb_std'] = df['price'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Drop NaN values
        return df.dropna()
    
    def prepare_data(self, df):
        """Prepare data for model training/prediction"""
        # Select features
        feature_columns = ['price', 'sma_5', 'sma_20', 'rsi', 'macd', 'volatility', 'momentum']
        data = df[feature_columns].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        return scaled_data
    
    def train(self, ticker_data, epochs=50, batch_size=32, learning_rate=0.001):
        """Train the LSTM model"""
        # Add features
        df = self.add_features(ticker_data)
        
        # Prepare data
        scaled_data = self.prepare_data(df)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Split into train/validation sets (80/20)
        train_size = int(len(X_tensor) * 0.8)
        X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
        y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            accuracy = 100 * correct / total
            
            # Learning rate scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_market_predictor.pth')
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Load best model
        self.model.load_state_dict(torch.load('best_market_predictor.pth'))
        
        # Plot training/validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('market_predictor_training.png')
        
        return train_losses, val_losses
    
    def predict(self, recent_data):
        """Predict market movement based on recent data"""
        # Ensure model is in evaluation mode
        self.model.eval()
        
        try:
            # Add features
            df = self.add_features(recent_data)
            
            # Prepare data
            scaled_data = self.prepare_data(df)
            
            # Create sequence for prediction (last lookback window)
            X = scaled_data[-self.lookback:].reshape(1, self.lookback, -1)
            
            # Convert to PyTorch tensor
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            # Map prediction to action
            action_map = {0: "sell", 1: "hold", 2: "buy"}
            action = action_map[prediction]
            
            # Get latest price and technical indicators for reasoning
            latest_data = df.iloc[-1]
            current_price = latest_data['price']
            
            # Generate reasoning based on technical indicators
            reasoning = self._generate_reasoning(latest_data)
            
            return {
                "prediction": action,
                "confidence": confidence,
                "current_price": current_price,
                "technical_indicators": {
                    "rsi": latest_data['rsi'],
                    "macd": latest_data['macd'],
                    "sma_5": latest_data['sma_5'],
                    "sma_20": latest_data['sma_20'],
                    "volatility": latest_data['volatility']
                },
                "reasoning": reasoning
            }
            
        except Exception as e:
            return {
                "prediction": "hold",
                "confidence": 0.33,
                "reasoning": f"Error in prediction: {str(e)}. Defaulting to hold."
            }
    
    def _generate_reasoning(self, latest_data):
        """Generate human-readable reasoning for the prediction"""
        reasons = []
        
        # RSI analysis
        rsi = latest_data['rsi']
        if rsi > 70:
            reasons.append(f"RSI is overbought at {rsi:.2f}")
        elif rsi < 30:
            reasons.append(f"RSI is oversold at {rsi:.2f}")
        else:
            reasons.append(f"RSI is neutral at {rsi:.2f}")
        
        # Moving average analysis
        if latest_data['sma_5'] > latest_data['sma_20']:
            reasons.append("Short-term trend is above long-term trend (bullish)")
        else:
            reasons.append("Short-term trend is below long-term trend (bearish)")
        
        # MACD analysis
        if latest_data['macd'] > 0:
            reasons.append("MACD is positive (bullish)")
        else:
            reasons.append("MACD is negative (bearish)")
        
        # Volatility analysis
        if latest_data['volatility'] > 0.02:  # 2% daily volatility is high
            reasons.append(f"Market volatility is high at {latest_data['volatility']*100:.2f}%")
        else:
            reasons.append(f"Market volatility is normal at {latest_data['volatility']*100:.2f}%")
        
        # Bollinger Bands analysis
        if latest_data['price'] > latest_data['bb_upper']:
            reasons.append("Price is above upper Bollinger Band (potential reversal)")
        elif latest_data['price'] < latest_data['bb_lower']:
            reasons.append("Price is below lower Bollinger Band (potential reversal)")
        
        return "Market analysis: " + "; ".join(reasons)
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        """Load a trained model"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
