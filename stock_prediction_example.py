#!/usr/bin/env python3
"""
Stock Price Prediction Example
=============================

This script demonstrates how to use the trained transformer model
for stock price prediction with real-world data.

Usage:
    python stock_prediction_example.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Import the prediction functions from NLP.py
from NLP import (
    load_model, predict_next_day, predict_multiple_days, 
    get_prediction_confidence, SentimentIntensityAnalyzer
)

def get_latest_stock_data(symbol="TSLA", days=60):
    """
    Fetch the latest stock data for prediction
    
    Args:
        symbol: Stock symbol (default: TSLA)
        days: Number of days to fetch (default: 60)
    
    Returns:
        DataFrame with Date, Close, and sentiment columns
    """
    print(f"Fetching latest {days} days of data for {symbol}...")
    
    # Get stock data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            raise Exception("No data received")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    
    # Prepare data
    stock_data = stock_data.reset_index()
    stock_data = stock_data[["Date", "Close"]].copy()
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    
    # Generate synthetic sentiment scores (in real use, you'd get this from news analysis)
    sia = SentimentIntensityAnalyzer()
    # For demo purposes, create some realistic sentiment variation
    np.random.seed(42)
    sentiment_scores = np.random.normal(0, 0.3, len(stock_data))
    # Add some correlation with price changes
    price_changes = stock_data["Close"].pct_change().fillna(0)
    sentiment_scores += price_changes * 2  # Sentiment correlates with price changes
    
    stock_data["sentiment"] = np.clip(sentiment_scores, -1, 1)
    
    print(f"Successfully fetched {len(stock_data)} days of data")
    return stock_data

def demonstrate_predictions():
    """Demonstrate various prediction scenarios"""
    
    print("="*60)
    print("STOCK PRICE PREDICTION DEMONSTRATION")
    print("="*60)
    
    # Load the trained model
    try:
        model, scaler = load_model("stock_prediction_model.pth")
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model file not found. Please run NLP.py first to train the model.")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get latest stock data
    stock_data = get_latest_stock_data("TSLA", days=60)
    if stock_data is None:
        print("❌ Could not fetch stock data")
        return
    
    print(f"\n📊 Latest stock price: ${stock_data['Close'].iloc[-1]:.2f}")
    print(f"📈 Price change (last 5 days): {((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-6]) - 1) * 100:.2f}%")
    
    # 1. Predict next day
    print("\n" + "="*40)
    print("1. NEXT DAY PREDICTION")
    print("="*40)
    
    try:
        next_day_pred = predict_next_day(model, scaler, stock_data)
        print(f"🔮 Predicted price for tomorrow: ${next_day_pred:.2f}")
        
        # Calculate expected change
        current_price = stock_data['Close'].iloc[-1]
        change = next_day_pred - current_price
        change_pct = (change / current_price) * 100
        print(f"📊 Expected change: ${change:.2f} ({change_pct:+.2f}%)")
        
    except Exception as e:
        print(f"❌ Error in next day prediction: {e}")
    
    # 2. Predict multiple days
    print("\n" + "="*40)
    print("2. MULTIPLE DAYS PREDICTION")
    print("="*40)
    
    try:
        # Predict next 5 days with different sentiment scenarios
        scenarios = {
            "Neutral": [0.0] * 5,
            "Positive": [0.5, 0.3, 0.4, 0.2, 0.1],
            "Negative": [-0.5, -0.3, -0.4, -0.2, -0.1]
        }
        
        for scenario_name, sentiment_scores in scenarios.items():
            predictions = predict_multiple_days(
                model, scaler, stock_data, 
                days_ahead=5, 
                sentiment_scores=sentiment_scores
            )
            
            print(f"\n📈 {scenario_name} Sentiment Scenario:")
            for i, pred in enumerate(predictions, 1):
                print(f"   Day {i}: ${pred:.2f}")
    
    except Exception as e:
        print(f"❌ Error in multiple days prediction: {e}")
    
    # 3. Get prediction confidence
    print("\n" + "="*40)
    print("3. PREDICTION CONFIDENCE")
    print("="*40)
    
    try:
        mean_pred, (lower, upper) = get_prediction_confidence(model, scaler, stock_data)
        print(f"🎯 Mean prediction: ${mean_pred:.2f}")
        print(f"📊 95% Confidence interval: ${lower:.2f} - ${upper:.2f}")
        print(f"📏 Uncertainty range: ${upper - lower:.2f}")
        
    except Exception as e:
        print(f"❌ Error in confidence estimation: {e}")
    
    # 4. Create visualization
    print("\n" + "="*40)
    print("4. CREATING VISUALIZATION")
    print("="*40)
    
    try:
        create_prediction_plot(stock_data, model, scaler)
        print("📊 Prediction plot saved as 'prediction_analysis.png'")
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE!")
    print("="*60)

def create_prediction_plot(stock_data, model, scaler):
    """Create a comprehensive prediction visualization"""
    
    # Get predictions for the last 10 days of historical data
    historical_predictions = []
    for i in range(10, len(stock_data)):
        recent_data = stock_data.iloc[:i]
        try:
            pred = predict_next_day(model, scaler, recent_data)
            historical_predictions.append(pred)
        except:
            historical_predictions.append(np.nan)
    
    # Get future predictions
    future_predictions = predict_multiple_days(model, scaler, stock_data, days_ahead=5)
    
    # Create dates for future predictions
    last_date = stock_data['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 6)]
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(stock_data['Date'], stock_data['Close'], 'b-', label='Actual Price', linewidth=2)
    
    # Plot historical predictions (last 10 days)
    hist_dates = stock_data['Date'].iloc[-10:]
    hist_preds = historical_predictions[-10:]
    plt.plot(hist_dates, hist_preds, 'r--', label='Historical Predictions', alpha=0.7)
    
    # Plot future predictions
    plt.plot(future_dates, future_predictions, 'g-', label='Future Predictions', linewidth=2, marker='o')
    
    # Add vertical line to separate historical and future
    plt.axvline(x=last_date, color='gray', linestyle=':', alpha=0.7, label='Prediction Point')
    
    plt.title('Stock Price Prediction Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add prediction details as text
    current_price = stock_data['Close'].iloc[-1]
    next_pred = future_predictions[0]
    change = next_pred - current_price
    change_pct = (change / current_price) * 100
    
    textstr = f'Current: ${current_price:.2f}\nNext Day: ${next_pred:.2f}\nChange: {change_pct:+.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.savefig('prediction_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    demonstrate_predictions()
