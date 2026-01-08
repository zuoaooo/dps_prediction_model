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
print("Loading data...")
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

print("\n" + "="*60)
print("TIME SERIES PREDICTION MODELS")
print("="*60)
print("Predicting all 12 months of 2021...\n")

# Prepare time series data
ts_data = train_df.set_index('date')['WERT']
n_forecast = 12  # Predict all 12 months of 2021

# Model 1: Linear Regression (Trend + Seasonality)
print("1. Linear Regression (Trend + Seasonality)")
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
    print(f"   ✓ Model fitted successfully")
    print(f"   Predictions: {lr_predictions[:3].round(2)}... (first 3 months)")
except Exception as e:
    print(f"   ✗ Model failed: {e}")
    lr_predictions = None

# Model 2: ARIMA
print("\n2. ARIMA(1,1,1) Model")
try:
    arima_model = ARIMA(ts_data, order=(1, 1, 1))
    arima_fitted = arima_model.fit()
    arima_forecast = arima_fitted.forecast(steps=n_forecast)
    arima_predictions = arima_forecast.values
    print(f"   ✓ Model fitted successfully")
    print(f"   Predictions: {arima_predictions[:3].round(2)}... (first 3 months)")
except Exception as e:
    print(f"   ✗ Model failed: {e}")
    arima_predictions = None

# Model 3: SARIMA with seasonal component
print("\n3. SARIMA(1,1,1)(1,1,1,12) Model")
try:
    sarima_model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fitted = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fitted.forecast(steps=n_forecast)
    sarima_predictions = sarima_forecast.values
    print(f"   ✓ Model fitted successfully")
    print(f"   Predictions: {sarima_predictions[:3].round(2)}... (first 3 months)")
except Exception as e:
    print(f"   ✗ Model failed: {e}")
    sarima_predictions = None

# Model 4: SARIMA with optimized parameters
print("\n4. SARIMA(2,1,2)(1,1,1,12) Model (Optimized)")
try:
    sarima_opt_model = SARIMAX(ts_data, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
    sarima_opt_fitted = sarima_opt_model.fit(disp=False)
    sarima_opt_forecast = sarima_opt_fitted.forecast(steps=n_forecast)
    sarima_opt_predictions = sarima_opt_forecast.values
    print(f"   ✓ Model fitted successfully")
    print(f"   Predictions: {sarima_opt_predictions[:3].round(2)}... (first 3 months)")
except Exception as e:
    print(f"   ✗ Model failed: {e}")
    sarima_opt_predictions = None

# Model 5: Prophet
prophet_predictions = None
if PROPHET_AVAILABLE:
    print("\n5. Prophet Model (Facebook)")
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

        print(f"   ✓ Model fitted successfully")
        print(f"   Predictions: {prophet_predictions[:3].round(2)}... (first 3 months)")
    except Exception as e:
        print(f"   ✗ Model failed: {e}")
        prophet_predictions = None

print("\n" + "="*60)
print("MODEL EVALUATION (Full Year 2021)")
print("="*60)

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
if sarima_opt_predictions is not None:
    models['SARIMA_Optimized'] = sarima_opt_predictions
if prophet_predictions is not None:
    models['Prophet'] = prophet_predictions

# Calculate errors for each model
print(f"\nActual values for 2021: {actual_values}")
print("\nModel Performance Metrics:")
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
print(f"\n✓ Best Model: {best_model}")
print(f"  Mean Absolute Error: {model_metrics[best_model]['mae']:.2f}")
print(f"  Root Mean Squared Error: {model_metrics[best_model]['rmse']:.2f}")
print(f"  Mean Absolute Percentage Error: {model_metrics[best_model]['mape']:.2f}%")

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
results_df.to_csv('prediction_results_2021.csv', index=False)
print("\n✓ Detailed results saved to 'prediction_results_2021.csv'")

# Save summary
summary_df = pd.DataFrame([{
    'Best_Model': best_model,
    'MAE': model_metrics[best_model]['mae'],
    'RMSE': model_metrics[best_model]['rmse'],
    'MAPE': model_metrics[best_model]['mape'],
    'Total_Error': model_metrics[best_model]['total_error']
}])
summary_df.to_csv('model_summary.csv', index=False)

print("\n" + "="*60)
print("VISUALIZATION")
print("="*60)

# Create visualizations (4 essential plots)
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Plot 1: Historical trend (2016-2020) + 2021 predictions (MAIN PLOT)
ax1 = fig.add_subplot(gs[0, :])
historical_recent = train_df[train_df['date'] >= '2016-01-01']
ax1.plot(historical_recent['date'], historical_recent['WERT'], 'b-o',
         markersize=3, alpha=0.6, linewidth=1.5, label='Historical (2016-2020)')

# Plot 2021 predictions and actual
dates_2021 = truth_df['date']
ax1.plot(dates_2021, actual_values, 'ro-', markersize=6, linewidth=2,
         label='Actual 2021', zorder=5)

colors = ['green', 'orange', 'purple', 'brown', 'cyan']
for i, (model_name, preds) in enumerate(models.items()):
    ax1.plot(dates_2021, preds, marker='s', markersize=4, alpha=0.7,
             linewidth=1.5, label=f'{model_name}', color=colors[i % len(colors)])

ax1.axvline(x=pd.to_datetime('2021-01-01'), color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Number of Accidents', fontsize=11)
ax1.set_title('Alkoholunfälle: Historical Trend and 2021 Predictions', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=9)
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Model performance comparison
ax2 = fig.add_subplot(gs[1, 0])
model_names = list(model_metrics.keys())
maes = [model_metrics[m]['mae'] for m in model_names]
colors_bar = ['green' if m == best_model else 'steelblue' for m in model_names]
bars = ax2.barh(model_names, maes, color=colors_bar, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Mean Absolute Error (MAE)', fontsize=10)
ax2.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

for i, (bar, mae) in enumerate(zip(bars, maes)):
    ax2.text(mae + 0.5, bar.get_y() + bar.get_height()/2, f'{mae:.2f}',
             va='center', fontsize=9, fontweight='bold')

# Plot 3: Error distribution over months
ax3 = fig.add_subplot(gs[1, 1])
for i, model_name in enumerate(models.keys()):
    errors = [abs(models[model_name][j] - actual_values[j]) for j in range(12)]
    ax3.plot(range(1, 13), errors, marker='o', label=model_name, linewidth=2,
             color=colors[i % len(colors)])

ax3.set_xlabel('Month', fontsize=10)
ax3.set_ylabel('Absolute Error', fontsize=10)
ax3.set_title('Prediction Error by Month', fontsize=12, fontweight='bold')
ax3.set_xticks(range(1, 13))
ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=8)

plt.savefig('alkohol_prediction_analysis_2021.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to 'alkohol_prediction_analysis_2021.png'")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\nBest performing model: {best_model}")
print(f"Average monthly error: {model_metrics[best_model]['mae']:.2f} accidents")
print(f"This represents a {model_metrics[best_model]['mape']:.1f}% average error rate")