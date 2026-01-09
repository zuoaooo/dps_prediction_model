# Model Experiments

This folder contains experimental results from various models tested during the model selection process.

## Contents

Results from each model type include:
- `*_all_models_comparison.png` - Comparison of different configurations
- `*_best_model_vs_actual.png` - Best configuration vs actual data
- `*_model_summary.csv` - Performance metrics
- `*_prediction_results_2021.csv` - 2021 predictions

## Models Tested

1. **Holt-Winters** - Exponential smoothing (MAE: 7.03)
2. **Prophet** - Facebook's time series model (MAE: 9.49)
3. **SARIMA** - Statistical autoregressive model (MAE: 12.92)
4. **SVR** - Support Vector Regression (MAE: 6.37)

## Final Selection

The **MLP (Multi-Layer Perceptron)** model was selected as the best performer with MAE: 6.26.
Its results are stored in the `../prediction/` folder.