# Model Selection Scripts

This folder contains all the Python scripts used for model selection and comparison.

## Files

- `mlp_model.py` - Multi-Layer Perceptron (Neural Network) model - **BEST MODEL** (MAE: 6.26)
- `svr_model.py` - Support Vector Regression model (MAE: 6.37)
- `holt_winters_model.py` - Holt-Winters exponential smoothing model (MAE: 7.03)
- `prophet_model.py` - Facebook Prophet time series model (MAE: 9.49)
- `sarima_model.py` - SARIMA statistical model (MAE: 12.92)

## Usage

Each script can be run independently to train and evaluate the respective model:

```bash
python model_selection/mlp_model.py
```

## Results

The MLP model was selected as the final model and is used in the FastAPI application (`app_fastapi.py`).

All model comparison results are stored in:
- Final MLP results: `../prediction/`
- Other model experiments: `../model_experiments/`