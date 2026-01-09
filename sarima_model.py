import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
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

print("\nSARIMA MODELS WITH OPTIMIZED PARAMETERS")

print("Predicting all 12 months of 2021\n")

# Prepare time series data
ts_data = train_df.set_index('date')['WERT']
n_forecast = 12
actual_values = truth_df['WERT'].values

# Optimized parameters based on ACF/PACF analysis

p_range = [0, 1, 2]
d_range = [1]
q_range = [1, 2]
P_range = [0, 1, 2]
D_range = [1]
Q_range = [1, 2]
s = 12

param_combinations = list(product(p_range, d_range, q_range))
seasonal_combinations = list(product(P_range, D_range, Q_range))

results = []
for param in param_combinations:
    for seasonal_param in seasonal_combinations:
        try:
            model = SARIMAX(ts_data, order=param, seasonal_order=seasonal_param + (s,),
                          enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False, maxiter=200)
            results.append({
                'order': param,
                'seasonal_order': seasonal_param + (s,),
                'AIC': fitted.aic,
                'params': f"SARIMA{param}x{seasonal_param + (s,)}"
            })
        except:
            continue

results_df = pd.DataFrame(results).sort_values('AIC')
print(f"Evaluated {len(results)} models, top 3 selected:\n")

top_models = results_df.head(3)
models_predictions = {}

for idx, row in top_models.iterrows():
    model_name = row['params']
    print(f"{results_df.index.get_loc(idx) + 1}. {model_name}")
    try:
        model = SARIMAX(ts_data, order=row['order'], seasonal_order=row['seasonal_order'],
                       enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=200)
        forecast = fitted.forecast(steps=n_forecast).values
        models_predictions[model_name] = forecast
        print(f"   Model fitted successfully")
        print(f"   All 12 months: {forecast.round(2)}")
    except Exception as e:
        print(f"   Model failed: {e}")

print("4. SARIMA(1,1,1)(1,1,1,12) Baseline")
try:
    baseline = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    baseline_fitted = baseline.fit(disp=False)
    baseline_forecast = baseline_fitted.forecast(steps=n_forecast).values
    models_predictions['SARIMA(1,1,1)(1,1,1,12)'] = baseline_forecast
    print(f"   Model fitted successfully")
    print(f"   All 12 months: {baseline_forecast.round(2)}")
except Exception as e:
    print(f"   Model failed: {e}")

print("\nMODEL EVALUATION (Full Year 2021)")

print(f"Actual values for 2021: {actual_values}")
print("Model Performance Metrics:")
print("-" * 80)
print(f"{'Model':<42} {'MAE':<10} {'RMSE':<10} {'MAPE (%)':<12} {'Total Error':<12}")
print("-" * 80)

best_model = None
best_mae = float('inf')
model_metrics = {}

for model_name, predictions in models_predictions.items():
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

    print(f"{model_name:<42} {mae:>8.2f}   {rmse:>8.2f}   {mape:>10.2f}   {total_error:>10.2f}")

    if mae < best_mae:
        best_mae = mae
        best_model = model_name

print("-" * 80)
print(f"\nBest Model: {best_model}")

# Best model visualization
if best_model is not None:
    best_preds = models_predictions[best_model]
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
    ax_best.set_title(f'SARIMA: Best Model ({best_model}) Predictions vs Actual 2021',
                      fontsize=14, fontweight='bold')
    ax_best.grid(True, alpha=0.3)
    ax_best.legend(loc='best', fontsize=10)
    ax_best.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('prediction/sarima_best_model_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close(fig_best)
    print("\nBest model visualization saved to 'prediction/sarima_best_model_vs_actual.png'")

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
    for model_name in models_predictions.keys():
        row[f'Pred_{model_name}'] = models_predictions[model_name][i]
        row[f'Error_{model_name}'] = abs(models_predictions[model_name][i] - actual_values[i])
    results_list.append(row)

results_df_detailed = pd.DataFrame(results_list)
results_df_detailed.to_csv('prediction/sarima_prediction_results_2021.csv', index=False)
print("Detailed results saved to 'prediction/sarima_prediction_results_2021.csv'")

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
summary_df.to_csv('prediction/sarima_model_summary.csv', index=False)
print("Model summary (all models) saved to 'prediction/sarima_model_summary.csv'")

# Create visualization - All models comparison
fig, ax = plt.subplots(figsize=(16, 6))

# Plot historical data (2016-2020)
historical_recent = train_df[train_df['date'] >= '2016-01-01']
ax.plot(historical_recent['date'], historical_recent['WERT'], 'b-o',
        markersize=3, alpha=0.6, linewidth=1.5, label='Historical (2016-2020)')

# Plot 2021 predictions and actual
dates_2021 = truth_df['date']
ax.plot(dates_2021, actual_values, 'ro-', markersize=6, linewidth=2,
        label='Actual 2021', zorder=5)

colors = ['green', 'orange', 'purple', 'brown', 'cyan', 'pink']
for i, (model_name, preds) in enumerate(models_predictions.items()):
    ax.plot(dates_2021, preds, marker='s', markersize=4, alpha=0.7,
            linewidth=1.5, label=f'{model_name}', color=colors[i % len(colors)])

ax.axvline(x=pd.to_datetime('2021-01-01'), color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Number of Accidents', fontsize=11)
ax.set_title('SARIMA Models: 2021 Predictions Comparison', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('prediction/sarima_all_models_comparison.png', dpi=300, bbox_inches='tight')
print("Visualization saved to 'prediction/sarima_all_models_comparison.png'")
