# 🕵️‍♂️ Stock Holmes: XAUUSD 5-Minute Ahead Prediction

Stock Holmes is a machine learning system designed to ingest, cache, analyze, and predict the short-horizon price direction (UP, DOWN, FLAT) of spot Gold (**XAU/USD**) five minutes into the future. 

The system leverages historical time-series split walk-forward validation, custom market volatility/session features, and a LightGBM multiclass classifier, outputting directional signals and model confidence probabilities.

---

## ⚡ Key Features

* **Incremental SQLite Caching**: Automatically queries local candles before requesting new candles from Twelve Data, staying well within free API key rate limits.
* **Stationarized Volatility & Session Features**: Computes rolling standard deviations, ATR percentages, cyclic hours, and active market sessions (London, New York, and session overlaps).
* **3-Way Direction Target**: Predicts whether Gold returns over the next 5 minutes will be **UP (1)**, **DOWN (-1)**, or **FLAT (0)** (based on threshold parameter $\epsilon$).
* **Model Confidence**: Outputs predicted probability distribution over the target classes to filter out low-confidence "guessing" states.
* **Skill vs. Luck Benchmarking**: Compares the model's accuracy and cumulative hypothetical strategy returns directly against a naive last-value persistence baseline.

---

## 📂 Project Structure

```
D:\csd 231017\Stock Holmes\
├── .github/
│   └── workflows/
│       └── ingestion_inference.yml  # Automated pipeline running on schedule
├── app/
│   └── dashboard.py                 # Streamlit Web Dashboard
├── data/
│   └── stock_holmes.db              # Local cached SQLite database
├── src/
│   ├── ingestion/
│   │   └── fetcher.py               # Twelve Data fetch client
│   ├── features/
│   │   └── builder.py               # Session features & indicators
│   ├── models/
│   │   ├── train.py                 # LightGBM training & walk-forward pipeline
│   │   └── predict.py               # Live prediction inference
│   └── serving/
│       └── db_utils.py              # SQLite storage layers
├── tests/
│   ├── test_features.py             # Feature builder tests
│   └── test_ingestion.py            # SQLite cache logic tests
├── requirements.txt                 # Packages list
└── README.md
```

---

## 🚀 Setup and Installation

### 1. Clone the repository and navigate to folder
```powershell
cd "D:\csd 231017\Stock Holmes"
```

### 2. Install requirements
```powershell
pip install -r requirements.txt
```

### 3. Configure Secrets
Create a `.env` file from the example:
```powershell
copy .env.example .env
```
Open `.env` and fill in your Twelve Data API key:
```env
TWELVE_DATA_API_KEY=your_twelve_data_api_key_here
```

---

## 🛠️ Usage

### 1. Ingest Market Data
To backfill 5,000 recent 1-minute candles from Twelve Data:
```powershell
python -m src.ingestion.fetcher --backfill 5000
```

### 2. Train Model
Run the walk-forward validation and LightGBM model training:
```powershell
python -m src.models.train
```

### 3. Generate Prediction (Inference)
Fetch the latest candles, compute features, perform inference, and save the prediction to database:
```powershell
python -m src.models.predict
```

### 4. Launch Dashboard
Run the Streamlit visualization app locally:
```powershell
streamlit run app/dashboard.py
```

---

## 📊 Skill vs. Luck: Evaluation Protocol

Short-horizon forex/commodity spot markets are highly noisy and resemble random walks. Stock Holmes avoids overhyped point-regression estimates by using a strict walk-forward validation framework:

1. **Stationary Features**: The features are strictly relative returns and ratios—never raw prices—preventing drift leak.
2. **Expanding Window Backtest**: The model trains on a rolling historical subset (e.g. first 70%), validates hyperparameters on 10%, and is evaluated on the final 20% unseen test window.
3. **Naive Baselines**: Performance is compared to:
   * *Naive Flat*: Constantly predicting 0 (FLAT).
   * *Naive Sign*: Predicting that the next 5 minutes will continue the direction of the last 1-minute close return.
4. **Cumulative Equity Curve**: The Streamlit dashboard plots the cumulative returns of trading on the model's signals vs. trading on the naive sign baseline. This visually demonstrates whether the model's performance is sustained over time ("skill") or concentrated in short, lucky windows ("luck").
