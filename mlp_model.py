import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
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

print("\nMLP NEURAL NETWORK - PARAMETER SELECTION")

print("Testing different architectures to find optimal configuration\n")
print("This script compares various MLP architectures.")
print("The best model will be saved for use in app.py\n")

# Feature engineering
def create_features(df, lags=[1, 2, 3, 6, 12], rolling_windows=[3, 6, 12]):
    df = df.copy()
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    for lag in lags:
        df[f'lag_{lag}'] = df['WERT'].shift(lag)

    for window in rolling_windows:
        df[f'rolling_mean_{window}'] = df['WERT'].shift(1).rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['WERT'].shift(1).rolling(window=window).std()

    df['trend'] = range(len(df))
    return df

train_features = create_features(train_df)
train_features = train_features.dropna()

feature_cols = [col for col in train_features.columns
                if col not in ['WERT', 'date', 'MONAT', 'AUSPRAEGUNG', 'MONATSZAHL', 'JAHR']]

X_train = train_features[feature_cols].values
y_train = train_features['WERT'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

n_forecast = 12
actual_values = truth_df['WERT'].values

# Parameter grid for optimization
print("Optimizing MLP hyperparameters:\n")

configs = {
    'MLP_Deep_4Layer': {
        'hidden_layer_sizes': (120, 80, 40, 20),
        'activation': 'relu',
        'alpha': 0.001,
        'learning_rate': 'adaptive',
        'max_iter': 1000,
    },
    'MLP_Small': {
        'hidden_layer_sizes': (50,),
        'activation': 'relu',
        'alpha': 0.001,
        'learning_rate': 'adaptive',
        'max_iter': 1000,
    },
    'MLP_Deep_3Layer': {
        'hidden_layer_sizes': (100, 50, 25),
        'activation': 'relu',
        'alpha': 0.001,
        'learning_rate': 'adaptive',
        'max_iter': 1000,
    },
    'MLP_Wide': {
        'hidden_layer_sizes': (150, 75),
        'activation': 'relu',
        'alpha': 0.001,
        'learning_rate': 'adaptive',
        'max_iter': 1000,
    },
    'MLP_Deep_5Layer': {
        'hidden_layer_sizes': (120, 90, 60, 30, 15),
        'activation': 'relu',
        'alpha': 0.001,
        'learning_rate': 'adaptive',
        'max_iter': 1000,
    },
}

models_predictions = {}

for idx, (name, params) in enumerate(configs.items(), 1):
    print(f"{idx}. {name}")
    try:
        mlp_model = MLPRegressor(
            solver='adam',
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            **params
        )
        mlp_model.fit(X_train_scaled, y_train)

        # Multi-step forecasting
        predictions = []
        temp_df = train_df.copy()

        for month in range(1, 13):
            new_date = pd.Timestamp(f'2021-{month:02d}-01')
            new_row = pd.DataFrame({
                'date': [new_date],
                'WERT': [np.nan],
                'MONAT': [int(f'2021{month:02d}')],
                'JAHR': [2021]
            })
            temp_df = pd.concat([temp_df, new_row], ignore_index=True)
            temp_df = create_features(temp_df)
            last_row = temp_df.iloc[-1:][feature_cols].values
            last_row_scaled = scaler.transform(last_row)
            pred = mlp_model.predict(last_row_scaled)[0]
            predictions.append(pred)
            temp_df.iloc[-1, temp_df.columns.get_loc('WERT')] = pred

        predictions = np.array(predictions)
        models_predictions[name] = predictions
        print(f"   Model trained successfully")
        print(f"   All 12 months: {predictions.round(2)}")
    except Exception as e:
        print(f"   Model failed: {e}")

print("\nMODEL EVALUATION (Full Year 2021)")

print(f"Actual values for 2021: {actual_values}")
print("Model Performance Metrics:")
print("-" * 80)
print(f"{'Model':<25} {'MAE':<10} {'RMSE':<10} {'MAPE (%)':<12} {'Total Error':<12}")
print("-" * 80)

best_model = None
best_mae = float('inf')
model_metrics = {}

for model_name, predictions in models_predictions.items():
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

    print(f"{model_name:<25} {mae:>8.2f}   {rmse:>8.2f}   {mape:>10.2f}   {total_error:>10.2f}")

    if mae < best_mae:
        best_mae = mae
        best_model = model_name

print("-" * 80)
print(f"\nBest Model: {best_model}")

# Save best model configuration
print(f"\nRecommended configuration for app.py:")
print(f"  Architecture: {best_model}")
best_config = None
for name, config in configs.items():
    if name == best_model:
        best_config = config
        break
if best_config:
    print(f"  hidden_layer_sizes: {best_config['hidden_layer_sizes']}")
    print(f"  activation: {best_config['activation']}")
    print(f"  alpha: {best_config['alpha']}")
    print(f"  learning_rate: {best_config['learning_rate']}")
    print(f"  MAE: {best_mae:.2f}")

# Best model visualization
if best_model is not None:
    best_preds = models_predictions[best_model]
    fig_best, ax_best = plt.subplots(figsize=(16, 6))

    historical_recent = train_df[train_df['date'] >= '2016-01-01']
    ax_best.plot(historical_recent['date'], historical_recent['WERT'], 'b-o',
                 markersize=3, alpha=0.6, linewidth=1.5, label='Historical (2016-2020)')
    ax_best.plot(truth_df['date'], actual_values, 'ro-', markersize=6, linewidth=2,
                 label='Actual 2021', zorder=5)
    ax_best.plot(truth_df['date'], best_preds, 'gs--', markersize=5, linewidth=2,
                 label=f'Best Model ({best_model})', zorder=4)
    ax_best.axvline(x=pd.to_datetime('2021-01-01'), color='gray', linestyle='--', alpha=0.5)
    ax_best.set_xlabel('Date', fontsize=12)
    ax_best.set_ylabel('Number of Accidents', fontsize=12)
    ax_best.set_title(f'MLP Neural Network: Best Model ({best_model}) Predictions vs Actual 2021',
                      fontsize=14, fontweight='bold')
    ax_best.grid(True, alpha=0.3)
    ax_best.legend(loc='best', fontsize=10)
    ax_best.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('prediction/mlp_best_model_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close(fig_best)
    print("\nBest model visualization saved to 'prediction/mlp_best_model_vs_actual.png'")

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

results_df = pd.DataFrame(results_list)
results_df.to_csv('prediction/mlp_prediction_results_2021.csv', index=False)
print("Detailed results saved to 'prediction/mlp_prediction_results_2021.csv'")

# Save summary
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
summary_df = summary_df.sort_values('MAE')
summary_df.to_csv('prediction/mlp_model_summary.csv', index=False)
print("Model summary (all models) saved to 'prediction/mlp_model_summary.csv'")

# All models comparison
fig, ax = plt.subplots(figsize=(16, 6))

historical_recent = train_df[train_df['date'] >= '2016-01-01']
ax.plot(historical_recent['date'], historical_recent['WERT'], 'b-o',
        markersize=3, alpha=0.6, linewidth=1.5, label='Historical (2016-2020)')

dates_2021 = truth_df['date']
ax.plot(dates_2021, actual_values, 'ro-', markersize=6, linewidth=2,
        label='Actual 2021', zorder=5)

colors = ['green', 'orange', 'purple', 'brown', 'cyan', 'pink', 'olive', 'navy']
for i, (model_name, preds) in enumerate(models_predictions.items()):
    ax.plot(dates_2021, preds, marker='s', markersize=4, alpha=0.7,
            linewidth=1.5, label=f'{model_name}', color=colors[i % len(colors)])

ax.axvline(x=pd.to_datetime('2021-01-01'), color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Number of Accidents', fontsize=11)
ax.set_title('MLP Neural Network Models: 2021 Predictions Comparison', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('prediction/mlp_all_models_comparison.png', dpi=300, bbox_inches='tight')
print("Visualization saved to 'prediction/mlp_all_models_comparison.png'")
