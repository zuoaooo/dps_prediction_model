from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Union
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PredictionRequest(BaseModel):
    year: Union[int, str] = Field(...)
    month: Union[int, str] = Field(...)

class PredictionResponse(BaseModel):
    prediction: float

class APIInfo(BaseModel):
    message: str
    model: str
    training_period: str

app = FastAPI(
    title="Munich Traffic Accidents Forecasting API",
    description="Predict alcohol-related traffic accidents using MLP Neural Network",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

train_df = pd.read_csv('data/alkohol_training_data.csv')
train_df['date'] = pd.to_datetime(train_df['MONAT'], format='%Y%m')
train_df = train_df.sort_values('date')

train_features = create_features(train_df)
train_features = train_features.dropna()

feature_cols = [col for col in train_features.columns
                if col not in ['WERT', 'date', 'MONAT', 'AUSPRAEGUNG', 'MONATSZAHL', 'JAHR']]

X_train = train_features[feature_cols].values
y_train = train_features['WERT'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(120, 80, 40, 20),
    activation='relu',
    alpha=0.001,
    solver='adam',
    learning_rate='adaptive',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
mlp_model.fit(X_train_scaled, y_train)

first_train_year = train_df['date'].iloc[0].year
first_train_month = train_df['date'].iloc[0].month
last_train_year = train_df['date'].iloc[-1].year
last_train_month = train_df['date'].iloc[-1].month

prediction_cache = {}

@app.get("/", response_model=APIInfo)
async def home():
    return APIInfo(
        message="Munich Alcohol-Related Traffic Accidents Forecasting API",
        model="MLP Neural Network",
        training_period=f"{first_train_year}-{first_train_month:02d} to {last_train_year}-{last_train_month:02d}"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        try:
            year = int(request.year)
            month = int(request.month)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Year and month must be valid numbers")

        if not (2021 <= year <= 2030):
            raise HTTPException(status_code=400, detail="Year must be between 2021 and 2030")

        if not (1 <= month <= 12):
            raise HTTPException(status_code=400, detail="Month must be between 1 and 12")

        if year < last_train_year or (year == last_train_year and month <= last_train_month):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot predict for past dates. Training data ends at {last_train_year}-{last_train_month:02d}"
            )

        if year > last_train_year + 5:
            raise HTTPException(
                status_code=400,
                detail="Predictions only available up to 5 years in the future"
            )

        cache_key = f"{year}-{month}"
        if cache_key in prediction_cache:
            return PredictionResponse(
                prediction=prediction_cache[cache_key],
            )

        temp_df = train_df.copy()
        target_date = pd.Timestamp(f'{year}-{month:02d}-01')
        last_date = train_df['date'].iloc[-1]
        months_to_predict = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)

        for i in range(1, months_to_predict + 1):
            pred_year = last_date.year + (last_date.month + i - 1) // 12
            pred_month = (last_date.month + i - 1) % 12 + 1
            new_date = pd.Timestamp(f'{pred_year}-{pred_month:02d}-01')

            new_row = pd.DataFrame({
                'date': [new_date],
                'WERT': [np.nan],
                'MONAT': [int(f'{pred_year}{pred_month:02d}')],
                'JAHR': [pred_year]
            })
            temp_df = pd.concat([temp_df, new_row], ignore_index=True)
            temp_df = create_features(temp_df)

            last_row = temp_df.iloc[-1:][feature_cols].values
            last_row_scaled = scaler.transform(last_row)
            pred = mlp_model.predict(last_row_scaled)[0]

            temp_df.iloc[-1, temp_df.columns.get_loc('WERT')] = pred

            pred_cache_key = f"{pred_year}-{pred_month}"
            prediction_cache[pred_cache_key] = round(float(pred), 2)

        prediction_value = prediction_cache[cache_key]

        return PredictionResponse(
            prediction=prediction_value,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "MLP Neural Network",
        "model_loaded": mlp_model is not None,
        "training_data_end": f"{last_train_year}-{last_train_month:02d}",
        "cache_size": len(prediction_cache)
    }

if __name__ == '__main__':
    import uvicorn
    import os
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
