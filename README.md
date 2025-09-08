# Stock Price Prediction with Transformer Model

This project uses a **Time Series Transformer** combined with **sentiment analysis** to predict stock prices. The model analyzes both historical stock data and news sentiment to make predictions.

## 🚀 Quick Start

### 1. Train the Model
```bash
python NLP.py
```
This will:
- Download stock data (TSLA) and news data
- Train a transformer model
- Save the trained model as `stock_prediction_model.pth`

### 2. Make Predictions
```bash
python stock_prediction_example.py
```
This will demonstrate various prediction scenarios with real-time data.

## 📊 How It Works

### Model Architecture
- **Input**: 30 days of historical data (stock prices + sentiment scores)
- **Architecture**: Transformer with positional encoding
- **Output**: Next day's predicted stock price
- **Features**: 
  - Stock closing prices
  - News sentiment scores (-1 to +1)

### Key Components
1. **Data Collection**: Fetches stock data from Yahoo Finance
2. **Sentiment Analysis**: Uses NLTK's VADER sentiment analyzer
3. **Time Series Processing**: Creates 30-day sequences for training
4. **Transformer Model**: Deep learning model for sequence prediction
5. **Prediction Functions**: Various utilities for different prediction scenarios

## 🔮 Prediction Functions

### 1. Predict Next Day
```python
from NLP import load_model, predict_next_day

model, scaler = load_model("stock_prediction_model.pth")
prediction = predict_next_day(model, scaler, recent_data)
print(f"Predicted price: ${prediction:.2f}")
```

### 2. Predict Multiple Days
```python
from NLP import predict_multiple_days

# Predict next 5 days with custom sentiment
predictions = predict_multiple_days(
    model, scaler, recent_data, 
    days_ahead=5,
    sentiment_scores=[0.2, 0.1, -0.1, 0.3, 0.0]  # Custom sentiment
)
```

### 3. Get Prediction Confidence
```python
from NLP import get_prediction_confidence

mean_pred, (lower, upper) = get_prediction_confidence(model, scaler, recent_data)
print(f"95% confidence: ${lower:.2f} - ${upper:.2f}")
```

## 📈 Example Usage Scenarios

### Scenario 1: Daily Trading Decision
```python
# Get today's prediction
current_data = get_latest_stock_data("AAPL", days=60)
tomorrow_price = predict_next_day(model, scaler, current_data)

if tomorrow_price > current_data['Close'].iloc[-1]:
    print("📈 Bullish signal - consider buying")
else:
    print("📉 Bearish signal - consider selling")
```

### Scenario 2: Weekly Portfolio Planning
```python
# Predict next week
weekly_predictions = predict_multiple_days(
    model, scaler, current_data, 
    days_ahead=5,
    sentiment_scores=[0.1, 0.2, 0.0, -0.1, 0.3]  # Weekly sentiment forecast
)
```

### Scenario 3: Risk Assessment
```python
# Get confidence interval
mean_pred, (lower, upper) = get_prediction_confidence(model, scaler, current_data)
risk_level = (upper - lower) / mean_pred * 100
print(f"Prediction uncertainty: {risk_level:.1f}%")
```

## 🛠️ Customization

### Change Stock Symbol
```python
# In stock_prediction_example.py, modify:
stock_data = get_latest_stock_data("AAPL", days=60)  # Change TSLA to AAPL
```

### Adjust Prediction Horizon
```python
# Predict further ahead (note: accuracy decreases with longer horizons)
predictions = predict_multiple_days(model, scaler, data, days_ahead=10)
```

### Custom Sentiment Integration
```python
# Replace synthetic sentiment with real news analysis
def analyze_news_sentiment(news_texts):
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(text)['compound'] for text in news_texts]
```

## 📊 Model Performance

The model uses:
- **Sequence Length**: 30 days
- **Architecture**: 3-layer Transformer (128 d_model, 8 heads)
- **Training**: 10 epochs with AdamW optimizer
- **Validation**: 90/10 train/validation split
- **Features**: Stock prices + sentiment scores

## ⚠️ Important Notes

1. **Not Financial Advice**: This is for educational/research purposes only
2. **Past Performance**: Historical accuracy doesn't guarantee future results
3. **Market Volatility**: Stock markets are inherently unpredictable
4. **Data Quality**: Predictions depend on data quality and sentiment accuracy
5. **Model Limitations**: Predictions become less accurate over longer horizons

## 🔧 Requirements

```
yfinance>=0.2.0
pandas>=1.5.0
matplotlib>=3.5.0
seaborn>=0.11.0
nltk>=3.7
torch>=1.12.0
scikit-learn>=1.1.0
numpy>=1.21.0
```

## 📁 Files

- `NLP.py`: Main training script and model definition
- `stock_prediction_example.py`: Example usage and demonstrations
- `stock_prediction_model.pth`: Trained model (created after running NLP.py)
- `requirements.txt`: Python dependencies

## 🚀 Next Steps

1. **Improve Sentiment Data**: Integrate real-time news sentiment
2. **Add More Features**: Include technical indicators (RSI, MACD, etc.)
3. **Ensemble Methods**: Combine multiple models for better accuracy
4. **Real-time Integration**: Connect to live data feeds
5. **Backtesting**: Test strategy performance on historical data

---

**Happy Predicting! 📈🤖**
