# ================== INSTALL LIBRARIES ==================
# Run this only once in your terminal or notebook:
# pip install yfinance pandas matplotlib seaborn nltk torch datasets scikit-learn

# ================== IMPORT LIBRARIES ==================
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import urllib.request
import contextlib
import logging
import warnings


def has_internet() -> bool:
    try:
        with contextlib.closing(urllib.request.urlopen("https://www.google.com", timeout=3)):
            return True
    except Exception:
        return False


def can_reach_yahoo() -> bool:
    try:
        with contextlib.closing(urllib.request.urlopen(
            "https://query1.finance.yahoo.com/v7/finance/quote?symbols=TSLA", timeout=3
        )):
            return True
    except Exception:
        return False


# Quiet down noisy logs/warnings from network libs
warnings.filterwarnings("ignore")
for name in ("yfinance", "urllib3", "fsspec", "numexpr"):
    logging.getLogger(name).setLevel(logging.ERROR)

# ================== DOWNLOAD REQUIRED NLTK DATA ==================
try:
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
except Exception:
    class _NeutralSIA:
        def polarity_scores(self, _):
            return {"compound": 0.0}
    sia = _NeutralSIA()

# ================== STEP 1: LOAD STOCK DATA ==================
START_DATE = "2018-01-01"
END_DATE = "2023-12-31"
stock_data = pd.DataFrame()
if has_internet() and can_reach_yahoo():
    try:
        stock_data = yf.download("TSLA", start=START_DATE, end=END_DATE, progress=False, threads=False)
    except Exception:
        stock_data = pd.DataFrame()
if stock_data is None or stock_data.empty:
    # try local CSV fallback
    csv_path = "stock_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "Date" in df and "Close" in df:
            stock_data = df
        else:
            stock_data = pd.DataFrame()
    else:
        stock_data = pd.DataFrame()
if not stock_data.empty and "Date" not in stock_data.columns:
    stock_data = stock_data.reset_index()
if not stock_data.empty:
    stock_data = stock_data[["Date", "Close"]]
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])  # ensure datetime

if stock_data.empty:
    print("Stock Data: using synthetic series (offline/unavailable)")
else:
    print("Stock Data: fetched online (showing first 5 rows)")
    print(stock_data.head())

# ================== STEP 2: LOAD NEWS DATA ==================
try:
    dataset = load_dataset("ag_news", split="train[:5000]")  # smaller subset for demo
except Exception:
    # minimal offline fallback
    dataset = {"text": ["Market is stable", "Stocks fall", "Great earnings", "Tech rally", "Inflation fears" ]}
    dataset = pd.DataFrame(dataset)
    # mimic HF dataset to DataFrame conversion below
    news_df = dataset.copy()
    news_df["sentiment"] = news_df["text"].apply(lambda x: sia.polarity_scores(x)["compound"])
    news_df["Date"] = pd.date_range(start=START_DATE, periods=len(news_df), freq="D")
    news_df = news_df[["Date", "sentiment"]]
else:
    news_df = pd.DataFrame(dataset)
if isinstance(news_df, pd.DataFrame) and "sentiment" not in news_df.columns:
    news_df["sentiment"] = news_df["text"].apply(lambda x: sia.polarity_scores(x)["compound"])
if "Date" not in news_df.columns:
    news_df["Date"] = pd.date_range(start=START_DATE, periods=len(news_df), freq="D")
news_df = news_df[["Date", "sentiment"]]

print("\nNews Data:\n", news_df.head())

# ================== STEP 3: MERGE ==================
if stock_data is None or stock_data.empty:
    # synthesize stock series aligned to news dates to allow training offline
    rng = news_df["Date"].sort_values()
    base = 200.0
    noise = np.random.default_rng(42).normal(0, 1, size=len(rng)).cumsum()
    close = base + noise + (news_df["sentiment"].rolling(7, min_periods=1).mean().fillna(0).values * 5.0)
    stock_data = pd.DataFrame({"Date": rng.values, "Close": close})
merged = pd.merge(stock_data, news_df, on="Date", how="inner")

print("\nMerged Data:\n", merged.head())
if merged.empty:
    raise RuntimeError("Merged dataset is empty. Ensure either internet access or provide 'stock_data.csv' with Date,Close columns.")

# ================== STEP 4: VISUALIZATION ==================
plt.figure(figsize=(12,6))
sns.lineplot(data=merged, x="Date", y="Close", label="Stock Price")
sns.lineplot(data=merged, x="Date", y="sentiment", label="Sentiment")
plt.title("Stock Price vs Sentiment")
plt.legend()
plt.tight_layout()
plt.savefig("plot_stock_vs_sentiment.png", dpi=120)
plt.close()

# ================== STEP 5: PREPARE DATA FOR MODEL ==================
# Scale features
scaler = MinMaxScaler()
scaled = scaler.fit_transform(merged[["Close", "sentiment"]])
scaled_df = pd.DataFrame(scaled, columns=["Close", "sentiment"])

# Convert to sequences
def create_sequences(data, seq_length=30):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, 0]  # predict stock price
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LEN = 30
X, y = create_sequences(scaled, SEQ_LEN)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=2, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        h = self.encoder(x)
        return self.head(h[:, -1, :])


# ================== STEP 6: TRAIN TRANSFORMER MODEL ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(d_model=128, nhead=8, num_layers=3, dim_ff=256, dropout=0.1).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# train/val split and mini-batching
num_samples = X.shape[0]
split_idx = int(num_samples * 0.9)
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128, shuffle=False)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb).squeeze()
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    scheduler.step()
    train_loss = running / len(train_loader.dataset)

    model.eval()
    with torch.no_grad():
        vloss = 0.0
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).squeeze()
            vloss += loss_fn(pred, yb).item() * xb.size(0)
        val_loss = vloss / max(1, len(val_loader.dataset))
    print(f"Epoch {epoch+1}/{EPOCHS}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")

# ================== STEP 8: PREDICTION ==================
model.eval()
with torch.no_grad():
    preds = []
    for i in range(0, X.shape[0], 256):
        xb = X[i:i+256].to(device)
        pb = model(xb).squeeze().detach().cpu().numpy()
        preds.append(pb)
    preds = np.concatenate(preds, axis=0)

# Inverse transform to original scale
preds_rescaled = scaler.inverse_transform(
    np.hstack((preds.reshape(-1,1), np.zeros((len(preds),1))))
)[:,0]

true_rescaled = scaler.inverse_transform(
    np.hstack((y.reshape(-1,1), np.zeros((len(y),1))))
)[:,0]

plt.figure(figsize=(12,6))
plt.plot(true_rescaled, label="True Stock Price")
plt.plot(preds_rescaled, label="Predicted Stock Price")
plt.legend()
plt.title("Stock Price Prediction with Sentiment")
plt.tight_layout()
plt.savefig("plot_predictions.png", dpi=120)
plt.close()

# ================== ENHANCED PREDICTION FUNCTIONS ==================

def save_model(model, scaler, filepath="stock_prediction_model.pth"):
    """Save the trained model and scaler for future use"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_data': scaler.data_min_,
        'scaler_scale': scaler.scale_,
        'model_config': {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'dim_ff': 256,
            'dropout': 0.1
        }
    }, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath="stock_prediction_model.pth"):
    """Load a previously saved model and scaler"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Recreate scaler
    scaler = MinMaxScaler()
    scaler.data_min_ = checkpoint['scaler_data']
    scaler.scale_ = checkpoint['scaler_scale']
    
    # Recreate model
    config = checkpoint['model_config']
    model = TimeSeriesTransformer(
        input_size=2,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_ff=config['dim_ff'],
        dropout=config['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, scaler

def predict_next_day(model, scaler, recent_data, sentiment_score=0.0):
    """
    Predict the next day's stock price using the last 30 days of data
    
    Args:
        model: Trained transformer model
        scaler: Fitted MinMaxScaler
        recent_data: DataFrame with 'Close' and 'sentiment' columns (last 30+ days)
        sentiment_score: Sentiment score for the prediction day (default: 0.0 for neutral)
    
    Returns:
        predicted_price: Predicted stock price for the next day
    """
    model.eval()
    
    # Ensure we have at least 30 days of data
    if len(recent_data) < 30:
        raise ValueError("Need at least 30 days of historical data for prediction")
    
    # Get the last 30 days
    last_30_days = recent_data.tail(30).copy()
    
    # Scale the data
    scaled_data = scaler.transform(last_30_days[["Close", "sentiment"]])
    
    # Convert to tensor and add batch dimension
    X = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(X).squeeze().cpu().numpy()
    
    # Inverse transform to get actual price
    pred_rescaled = scaler.inverse_transform(
        np.array([[prediction, 0.0]])
    )[0, 0]
    
    return pred_rescaled

def predict_multiple_days(model, scaler, recent_data, days_ahead=5, sentiment_scores=None):
    """
    Predict multiple days ahead using iterative prediction
    
    Args:
        model: Trained transformer model
        scaler: Fitted MinMaxScaler
        recent_data: DataFrame with 'Close' and 'sentiment' columns
        days_ahead: Number of days to predict ahead
        sentiment_scores: List of sentiment scores for future days (default: neutral)
    
    Returns:
        predictions: List of predicted prices for the next N days
    """
    if sentiment_scores is None:
        sentiment_scores = [0.0] * days_ahead  # Neutral sentiment
    
    predictions = []
    current_data = recent_data.copy()
    
    for i in range(days_ahead):
        # Predict next day
        pred_price = predict_next_day(model, scaler, current_data, sentiment_scores[i])
        predictions.append(pred_price)
        
        # Add the prediction to current data for next iteration
        next_date = current_data['Date'].iloc[-1] + pd.Timedelta(days=1)
        new_row = pd.DataFrame({
            'Date': [next_date],
            'Close': [pred_price],
            'sentiment': [sentiment_scores[i]]
        })
        current_data = pd.concat([current_data, new_row], ignore_index=True)
    
    return predictions

def get_prediction_confidence(model, scaler, recent_data, num_samples=100):
    """
    Get prediction confidence using Monte Carlo dropout
    
    Args:
        model: Trained transformer model
        scaler: Fitted MinMaxScaler
        recent_data: DataFrame with 'Close' and 'sentiment' columns
        num_samples: Number of samples for confidence estimation
    
    Returns:
        mean_prediction: Average predicted price
        confidence_interval: (lower_bound, upper_bound) for 95% confidence
    """
    model.train()  # Enable dropout for uncertainty estimation
    
    predictions = []
    for _ in range(num_samples):
        pred = predict_next_day(model, scaler, recent_data)
        predictions.append(pred)
    
    model.eval()  # Disable dropout
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    
    # 95% confidence interval
    lower_bound = mean_pred - 1.96 * std_pred
    upper_bound = mean_pred + 1.96 * std_pred
    
    return mean_pred, (lower_bound, upper_bound)

# Save the trained model
save_model(model, scaler)

print("\n" + "="*60)
print("STOCK PRICE PREDICTION GUIDE")
print("="*60)
print("\nYour model is now ready for predictions! Here's how to use it:")
print("\n1. PREDICT NEXT DAY:")
print("   pred_price = predict_next_day(model, scaler, recent_data)")
print("\n2. PREDICT MULTIPLE DAYS:")
print("   predictions = predict_multiple_days(model, scaler, recent_data, days_ahead=5)")
print("\n3. GET PREDICTION CONFIDENCE:")
print("   mean_pred, (lower, upper) = get_prediction_confidence(model, scaler, recent_data)")
print("\n4. SAVE/LOAD MODEL:")
print("   save_model(model, scaler, 'my_model.pth')")
print("   model, scaler = load_model('my_model.pth')")
print("\n" + "="*60)