import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Warning: Prophet not installed. Prophet model will be skipped.")
    PROPHET_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

# Load training data and ground truth
print("Loading data.")
train_df = pd.read_csv('data/alkohol_training_data.csv')
truth_df = pd.read_csv('data/alkohol_ground_truth_2021.csv')

# Data preprocessing
train_df['date'] = pd.to_datetime(train_df['MONAT'], format='%Y%m')
train_df = train_df.sort_values('date')
truth_df['date'] = pd.to_datetime(truth_df['MONAT'], format='%Y%m')
truth_df = truth_df.sort_values('date')

print(f"Training data: {len(train_df)} records from {train_df['JAHR'].min()} to {train_df['JAHR'].max()}")
print(f"Ground truth: {len(truth_df)} records for 2021")
print(f"Training value range: {train_df['WERT'].min()} - {train_df['WERT'].max()}")
print(f"Training mean: {train_df['WERT'].mean():.2f}, Std: {train_df['WERT'].std():.2f}")

print("\nTIME SERIES PREDICTION MODELS")

print("Predicting all 12 months of 2021\n")

# Prepare time series data
ts_data = train_df.set_index('date')['WERT']
n_forecast = 12  # Predict all 12 months of 2021

# Model 1: Linear Regression (Trend + Seasonality)
print("Linear Regression (Trend + Seasonality)")
try:
    # Create features for training
    train_df['month_index'] = range(len(train_df))
    train_df['month_of_year'] = train_df['date'].dt.month
    train_df['year'] = train_df['date'].dt.year

    X_train = train_df[['month_index', 'month_of_year', 'year']].values
    y_train = train_df['WERT'].values

    # Fit model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Create features for 2021 predictions
    X_2021 = []
    for month in range(1, 13):
        month_idx = len(train_df) + month - 1
        X_2021.append([month_idx, month, 2021])
    X_2021 = np.array(X_2021)

    lr_predictions = lr_model.predict(X_2021)
    print(f"   Model fitted successfully")
    print(f"   All 12 months: {lr_predictions.round(2)}")
except Exception as e:
    print(f"   Model failed: {e}")
    lr_predictions = None

# Model 2: ARIMA
print("2. ARIMA(1,1,1) Model")
try:
    arima_model = ARIMA(ts_data, order=(1, 1, 1))
    arima_fitted = arima_model.fit()
    arima_forecast = arima_fitted.forecast(steps=n_forecast)
    arima_predictions = arima_forecast.values
    print(f"   Model fitted successfully")
    print(f"   All 12 months: {arima_predictions.round(2)}")
except Exception as e:
    print(f"   Model failed: {e}")
    arima_predictions = None

# Model 3: SARIMA with seasonal component
print("3. SARIMA(1,1,1)(1,1,1,12) Model")
try:
    sarima_model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fitted = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fitted.forecast(steps=n_forecast)
    sarima_predictions = sarima_forecast.values
    print(f"   Model fitted successfully")
    print(f"   All 12 months: {sarima_predictions.round(2)}")
except Exception as e:
    print(f"   Model failed: {e}")
    sarima_predictions = None

# Model 4: Prophet
prophet_predictions = None
if PROPHET_AVAILABLE:
    print("4. Prophet Model (Facebook)")
    try:
        prophet_df = train_df[['date', 'WERT']].copy()
        prophet_df.columns = ['ds', 'y']

        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )
        prophet_model.fit(prophet_df)

        # Create future dates for 2021
        future_dates = pd.date_range(start='2021-01-01', periods=12, freq='MS')
        future_df = pd.DataFrame({'ds': future_dates})
        prophet_forecast = prophet_model.predict(future_df)
        prophet_predictions = prophet_forecast['yhat'].values

        print(f"   Model fitted successfully")
        print(f"   All 12 months: {prophet_predictions.round(2)}")
    except Exception as e:
        print(f"   Model failed: {e}")
        prophet_predictions = None

print("\nMODEL EVALUATION (Full Year 2021)")


# Get actual values for all 12 months
actual_values = truth_df['WERT'].values

# Collect all models
models = {}
if lr_predictions is not None:
    models['Linear_Regression'] = lr_predictions
if arima_predictions is not None:
    models['ARIMA'] = arima_predictions
if sarima_predictions is not None:
    models['SARIMA'] = sarima_predictions
if prophet_predictions is not None:
    models['Prophet'] = prophet_predictions

# Calculate errors for each model
print(f"Actual values for 2021: {actual_values}")
print("Model Performance Metrics:")
print("-" * 80)
print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'MAPE (%)':<12} {'Total Error':<12}")
print("-" * 80)

best_model = None
best_mae = float('inf')
model_metrics = {}

for model_name, predictions in models.items():
    # Calculate metrics
    mae = np.mean(np.abs(predictions - actual_values))
    rmse = np.sqrt(np.mean((predictions - actual_values)**2))
    mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
    total_error = np.sum(np.abs(predictions - actual_values))

    model_metrics[model_name] = {
        'predictions': predictions,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'total_error': total_error
    }

    print(f"{model_name:<20} {mae:>8.2f}   {rmse:>8.2f}   {mape:>10.2f}   {total_error:>10.2f}")

    if mae < best_mae:
        best_mae = mae
        best_model = model_name

print("-" * 80)
print(f"\nBest Model: {best_model}")

# Best model visualization (recent history 2016-2020 + best prediction vs actual 2021)
if best_model is not None:
    best_preds = models[best_model]
    fig_best, ax_best = plt.subplots(figsize=(16, 6))

    # Only show recent historical data (2016-2020) for better clarity
    historical_recent = train_df[train_df['date'] >= '2016-01-01']
    ax_best.plot(historical_recent['date'], historical_recent['WERT'], 'b-o',
                 markersize=3, alpha=0.6, linewidth=1.5, label='Historical (2016-2020)')
    ax_best.plot(truth_df['date'], actual_values, 'ro-', markersize=6, linewidth=2, label='Actual 2021', zorder=5)
    ax_best.plot(truth_df['date'], best_preds, 'gs--', markersize=5, linewidth=2, label=f'Best Model ({best_model})', zorder=4)
    ax_best.axvline(x=pd.to_datetime('2021-01-01'), color='gray', linestyle='--', alpha=0.5)
    ax_best.set_xlabel('Date', fontsize=12)
    ax_best.set_ylabel('Number of Accidents', fontsize=12)
    ax_best.set_title('Alcohol-related Accidents: Best Model Predictions vs Actual 2021',
                      fontsize=14, fontweight='bold')
    ax_best.grid(True, alpha=0.3)
    ax_best.legend(loc='best', fontsize=10)
    ax_best.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('prediction/best_model_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close(fig_best)
    print("\nBest model visualization saved to 'prediction/best_model_vs_actual.png'")

# Save detailed results
results_list = []
months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

for i in range(12):
    row = {
        'Month': i + 1,
        'Month_Name': months[i],
        'Actual': actual_values[i]
    }
    for model_name in models.keys():
        row[f'Pred_{model_name}'] = models[model_name][i]
        row[f'Error_{model_name}'] = abs(models[model_name][i] - actual_values[i])
    results_list.append(row)

results_df = pd.DataFrame(results_list)
results_df.to_csv('prediction/prediction_results_2021.csv', index=False)
print("Detailed results saved to 'prediction_results_2021.csv'")

# Save summary for ALL models
summary_list = []
for model_name, metrics in model_metrics.items():
    summary_list.append({
        'Model': model_name,
        'MAE': metrics['mae'],
        'RMSE': metrics['rmse'],
        'MAPE': metrics['mape'],
        'Total_Error': metrics['total_error'],
        'Is_Best': 'Yes' if model_name == best_model else 'No'
    })

summary_df = pd.DataFrame(summary_list)
summary_df = summary_df.sort_values('MAE')  # Sort by MAE (best first)
summary_df.to_csv('prediction/model_summary.csv', index=False)
print("Model summary (all models) saved to 'prediction/model_summary.csv'")


# Create visualization - Historical trend + 2021 predictions
fig, ax = plt.subplots(figsize=(16, 6))

# Plot historical data (2016-2020)
historical_recent = train_df[train_df['date'] >= '2016-01-01']
ax.plot(historical_recent['date'], historical_recent['WERT'], 'b-o',
        markersize=3, alpha=0.6, linewidth=1.5, label='Historical (2016-2020)')

# Plot 2021 predictions and actual
dates_2021 = truth_df['date']
ax.plot(dates_2021, actual_values, 'ro-', markersize=6, linewidth=2,
        label='Actual 2021', zorder=5)

colors = ['green', 'orange', 'purple', 'brown', 'cyan']
for i, (model_name, preds) in enumerate(models.items()):
    ax.plot(dates_2021, preds, marker='s', markersize=4, alpha=0.7,
            linewidth=1.5, label=f'{model_name}', color=colors[i % len(colors)])

ax.axvline(x=pd.to_datetime('2021-01-01'), color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Number of Accidents', fontsize=11)
ax.set_title('Alkoholunfälle: Historical Trend and 2021 Predictions', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('prediction/alkohol_prediction_analysis_2021.png', dpi=300, bbox_inches='tight')
print("Visualization saved to 'prediction/alkohol_prediction_analysis_2021.png'")