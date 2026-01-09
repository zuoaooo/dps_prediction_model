# Munich Alcohol-Related Traffic Accidents Prediction

This project predicts monthly alcohol-related traffic accidents (Alkoholunfälle) in Munich using machine learning. The API is powered by a **Multi-Layer Perceptron (MLP) Neural Network** trained on historical data from the [Munich Open Data Portal](https://opendata.muenchen.de/dataset/monatszahlen-verkehrsunfaelle/resource/40094bd6-f82d-4979-949b-26c8dc00b9a7).

## 🚀 Quick Start

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the FastAPI server:
```bash
python3 app_fastapi.py
```

3. Test locally:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"year": 2021, "month": 1}'
```

Or use the test script:
```bash
python test_api.py
```

### API Documentation

FastAPI provides interactive API documentation when running locally:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🎯 Model & Performance

### MLP Neural Network

The prediction model is a **Multi-Layer Perceptron (MLP)** with the following configuration:

- **Architecture**: 4 hidden layers (120-80-40-20 neurons)
- **Features**: Lag values (1,2,3,6,12), rolling statistics (3,6,12), cyclic month encoding (sin/cos), quarter, trend
- **Training**: Adam optimizer, adaptive learning rate, early stopping
- **Performance**: MAE 6.26, RMSE 7.59, MAPE 26.53% (validated on 2021 data)

### Prediction Results

![MLP Predictions](prediction/mlp_best_model_vs_actual.png)
*MLP model predictions vs actual 2021 values*

Detailed prediction results are available in [mlp_prediction_results_2021.csv](prediction/mlp_prediction_results_2021.csv)

## 📁 Project Structure

```
├── data/                          # Training data from Munich Open Data Portal
│   ├── alkohol_training_data.csv
│   └── alkohol_ground_truth_2021.csv
├── model_selection/               # Model training scripts
│   ├── mlp_model.py              # MLP neural network
│   ├── sarima_model.py           # SARIMA time series model
│   ├── prophet_model.py          # Prophet forecasting
│   ├── holt_winters_model.py     # Holt-Winters exponential smoothing
│   └── svr_model.py              # Support Vector Regression
├── model_experiments/             # All model results and comparisons
│   ├── *_model_summary.csv       # Performance metrics
│   ├── *_prediction_results_2021.csv
│   └── *_best_model_vs_actual.png
├── prediction/                    # Production model results (MLP)
│   ├── mlp_best_model_vs_actual.png
│   └── mlp_prediction_results_2021.csv
├── scripts/                       # EDA and visualization scripts
│   ├── monthly_visualization.py
│   ├── filter.py
├── visualization/                 # EDA output visualizations
├── app_fastapi.py                # FastAPI application (Production)
├── app.py                        # Flask API (Legacy)
├── test_api.py                   # API testing script
└── requirements.txt              # Python dependencies
```

## 🔌 API Reference

### Endpoints

#### GET /
Get API information and model details.

```bash
curl http://localhost:8000/
```

#### GET /health
Health check endpoint.

```bash
curl http://localhost:8000/health
```

#### POST /predict
Get prediction for a specific month and year.

**Request:**
```json
{
  "year": 2021,
  "month": 1
}
```

**Response:**
```json
{
  "prediction": 22.88
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"year": 2021, "month": 1}'
```

### Error Handling

The API returns appropriate error messages for invalid inputs:

- Missing fields: `{"error": "Both 'year' and 'month' are required"}`
- Invalid month: `{"error": "Month must be between 1 and 12"}`
- Past dates: `{"error": "Cannot predict for past dates. Training data ends at 2020-12"}`
- Future limit: `{"error": "Predictions only available up to 5 years in the future"}`

## 🚀 Deployment

The application can be deployed to cloud services like Render, Heroku, or AWS.

## 📝 License

This project is for educational and research purposes.
