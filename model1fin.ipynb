!pip install pandas-ta yfinance

import yfinance as yf

# 1. Load Data
df = yf.download("RELIANCE.NS", start="2011-01-01", end="2024-01-01")


import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns


# --- FIX: Flatten MultiIndex columns if they exist ---
# The yfinance.download function can return a DataFrame with MultiIndex columns,
# which causes issues when assigning new single-level named columns.
# This converts columns like ('Close', 'RELIANCE.NS') to 'Close'.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# --- FEATURE ENGINEERING ---

# A. Moving Average Convergence / Basics
# Distance from SMA (50-day) - Measures mean reversion
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['Dist_SMA_50'] = (df['Close'] / df['SMA_50']) - 1

# B. Momentum & Trend
# RSI (14-day)
df['RSI'] = ta.rsi(df['Close'], length=14)

# ROC (Rate of Change - 5-day)
# Captures the percentage change over the prediction horizon
df['ROC_5'] = ta.roc(df['Close'], length=5)

# C. Volatility & Risk
# Bollinger Band Width
# (Upper - Lower) / Mid. A low value indicates a "Squeeze"
bbands = ta.bbands(df['Close'], length=20, std=2)
df['BB_Width'] = (bbands['BBU_20_2.0_2.0'] - bbands['BBL_20_2.0_2.0']) / bbands['BBM_20_2.0_2.0']

# ATR Ratio (Normalized Volatility)
# Dividing by Close makes it comparable across 13 years of price action
df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
df['ATR_Ratio'] = df['ATR'] / df['Close']

# D. Volume & Flow
# Chaikin Money Flow (CMF) - Measures accumulation vs distribution
df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)

# Rolling Log Volume (5-day)
# We take the log to normalize spikes, then calculate a rolling sum/mean
df['Log_Vol'] = np.log(df['Volume'])
df['Rolling_Log_Vol_5'] = df['Log_Vol'].rolling(window=5).mean()

# E. Statistical Stationary Features
# Z-Score (20-day) - How many std devs is current price from the mean?
df['Z_Score_20'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()

# F. Time-Based Features
# Day of the Week (Monday=0, Sunday=6)
df['Day_of_Week'] = df.index.dayofweek

df['EMA_20'] = ta.ema(df['Close'], length=20)
df['EMA_50'] = ta.ema(df['Close'], length=50)
df['Trend'] = df['EMA_20'] - df['EMA_50']

df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
df['Momentum'] = ta.mom(df['Close'], length=10)

# --- Create Log Returns and Target Variable ---
df['Log_Ret'] = np.log(df['Close']).diff()
df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)

# --- CLEANUP ---
# Drop rows with NaNs created by rolling windows (e.g., SMA_50 needs 50 days) and new features
df.dropna(inplace=True)

# Select features for correlation map
features_to_correlate = [
    'Volume',
    'Close',
    'Dist_SMA_50',
    'RSI',
    'ROC_5',
    'BB_Width',
    'ATR_Ratio',
    'CMF',
    'Rolling_Log_Vol_5',
    'Z_Score_20',
    'Day_of_Week',
    'Log_Ret',
    'EMA_50',
    'EMA_20',
    'Trend',
    'Target'
]

# Calculate the correlation matrix
correlation_matrix = df[features_to_correlate].corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Map of Engineered Features')
plt.show()

!pip uninstall numpy scikit-learn
!pip install numpy==1.26.4

!pip install numpy scikit-learn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler

# 1. Select only our engineered features (exclude OHLC raw prices)
feature_cols = [
    'Log_Ret', 'Z_Score_20', 'RSI', 'ROC_5', 'Dist_SMA_50',
    'BB_Width', 'ATR_Ratio', 'CMF', 'Rolling_Log_Vol_5','Day_of_Week', 'Trend'
]


# 2. Chronological Split
train_df = df[:'2020-12-31']
test_df = df['2021-01-01':]

X_train, y_train = train_df[feature_cols], train_df['Target']
X_test, y_test = test_df[feature_cols], test_df['Target']

# 3. Scaling (The right way)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Learn mean/std from TRAIN only
X_test_scaled = scaler.transform(X_test)       # Apply TRAIN mean/std to TEST

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# Refined XGBoost with Regularization
model = xgb.XGBClassifier(
    n_estimators=120,
    max_depth=3,            # Shallow trees to prevent overfitting
    learning_rate=0.05,     # Slow learning is better for noisy data
    gamma=2,                # Minimum loss reduction to make a split
    subsample=0.7,
    colsample_bytree=0.8,
    eval_metric='logloss',
    use_label_encoder=False, # Suppress warning
    objective='binary:logistic' # Explicitly set for binary classification
)

model.fit(X_train_scaled, y_train)

print("XGBoost model training complete.")


from scipy.stats import uniform, randint

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 200),  # Number of boosting rounds
    'max_depth': randint(3,6),      # Maximum tree depth
    'learning_rate': uniform(0.01, 0.05), # Step size shrinkage to prevent overfitting
    'gamma': uniform(0, 2),         # Minimum loss reduction required to make a further partition
    'subsample': uniform(0.7, 0.3),   # Subsample ratio of the training instance
    'colsample_bytree': uniform(0.6, 0.4), # Subsample ratio of columns when constructing each tree
    'lambda': uniform(1, 7),          # L2 regularization term on weights
    'alpha': uniform(0.1, 1)            # L1 regularization term on weights
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='precision',
    cv=tscv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit RandomizedSearchCV to the training data
random_search.fit(X_train_scaled, y_train)

print("RandomizedSearchCV complete.")
print(f"Best parameters found: {random_search.best_params_}")


model = random_search.best_estimator_

# Retrain the model with the best parameters (optional, as best_estimator_ is already fitted)
# If you want to train on the entire training set with the best params, you can refit it like this:
model.fit(X_train_scaled, y_train)

print("XGBoost model retrained with best parameters.")

y_probs = model.predict_proba(X_test_scaled)[:, 1]

# Higher threshold = Higher quality trades
threshold = 0.51
y_pred_high_conf = (y_probs >= threshold).astype(int)

print("Classification Report (High Confidence):")
print(classification_report(y_test, y_pred_high_conf))

print("\nConfusion Matrix (Tuned Model):")
print(confusion_matrix(y_test, y_pred_high_conf))

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred_high_conf)

# Create a heatmap for better visualization
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()



capital = 1_000_000
capital_history = []

i = 0
indices = X_test.index

while i < len(indices) - 5:

    prob = y_probs[i]

    if prob > 0.55:
        signal = 1
    elif prob < 0.45:
        signal = 0
    else:
        i += 1
        continue

    entry_idx = indices[i+1]
    exit_idx = indices[i+5]

    entry_price = df.loc[entry_idx, 'Open']
    exit_price = df.loc[exit_idx, 'Close']

    if signal == 1:
        trade_return = (exit_price / entry_price) - 1
    else:
        trade_return = (entry_price / exit_price) - 1

    capital *= (1 + trade_return)
    capital_history.append(capital)

    i += 5

capital_series = pd.Series(capital_history)

returns = capital_series.pct_change().dropna()

sharpe = np.sqrt(252/5) * returns.mean() / returns.std()

print("Sharpe Ratio:", sharpe)
