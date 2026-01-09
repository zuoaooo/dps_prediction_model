# Final Prediction Results

This folder contains the results from the **selected MLP (Multi-Layer Perceptron)** model.

## Model Performance

- **Model**: MLP Neural Network (120-80-40-20 architecture)
- **MAE**: 6.26 (best among all tested models)
- **Training Period**: Up to 2020-12

## Files

- `mlp_all_models_comparison.png` - Comparison of different MLP architectures
- `mlp_best_model_vs_actual.png` - Best MLP model predictions vs actual data
- `mlp_model_summary.csv` - Detailed performance metrics
- `mlp_prediction_results_2021.csv` - 2021 monthly predictions

## Usage in Production

This model is deployed in the FastAPI application (`app_fastapi.py`) and is accessible via:

```bash
POST /predict
{
  "year": 2021,
  "month": 1
}
```

## Other Model Experiments

Results from other tested models (Holt-Winters, Prophet, SARIMA, SVR) can be found in the `../model_experiments/` folder.