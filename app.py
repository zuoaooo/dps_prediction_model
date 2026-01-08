from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Load and prepare training data once when the app starts
print("Loading training data...")
train_df = pd.read_csv('data/alkohol_training_data.csv')
train_df['date'] = pd.to_datetime(train_df['MONAT'], format='%Y%m')
train_df = train_df.sort_values('date')
ts_data = train_df.set_index('date')['WERT']

# Train the best performing model (SARIMA)
print("Training SARIMA model...")
model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
fitted_model = model.fit(disp=False)
print("Model trained successfully!")

# Get the last training date for calculating prediction steps
last_train_date = ts_data.index[-1]
last_train_year = last_train_date.year
last_train_month = last_train_date.month
print(f"Training data ends at: {last_train_year}-{last_train_month:02d}")

def calculate_months_ahead(target_year, target_month):
    """Calculate how many months ahead from the last training date"""
    months_ahead = (target_year - last_train_year) * 12 + (target_month - last_train_month)
    return months_ahead

# Cache for predictions to avoid recalculating
prediction_cache = {}

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Munich Traffic Accidents Prediction API (Alkoholunfälle - insgesamt)",
        "usage": "POST /predict with JSON body: {\"year\": 2021, \"month\": 01}",
        "model": "SARIMA(1,1,1)(1,1,1,12)",
        "training_period": f"up to {last_train_year}-{last_train_month:02d}",
        "note": "Can predict any future month/year"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        year = data.get('year')
        month = data.get('month')

        if year is None or month is None:
            return jsonify({"error": "Both 'year' and 'month' are required"}), 400

        # Validate input
        year = int(year)
        month = int(month)

        if month < 1 or month > 12:
            return jsonify({"error": "Month must be between 1 and 12"}), 400

        # Calculate how many months ahead to predict
        months_ahead = calculate_months_ahead(year, month)

        if months_ahead <= 0:
            return jsonify({"error": f"Cannot predict for past dates. Training data ends at {last_train_year}-{last_train_month:02d}"}), 400

        # Limit predictions to reasonable future (e.g., 10 years)
        if months_ahead > 120:
            return jsonify({"error": "Predictions only available up to 10 years in the future"}), 400

        # Check cache first
        cache_key = f"{year}-{month}"
        if cache_key in prediction_cache:
            return jsonify({"prediction": prediction_cache[cache_key]})

        # Generate prediction - forecast up to the requested month
        forecast = fitted_model.forecast(steps=months_ahead)
        prediction_value = float(forecast.iloc[-1])  # Get the last value (the target month)

        # Round to 2 decimal places
        prediction_value = round(prediction_value, 2)

        # Cache the result
        prediction_cache[cache_key] = prediction_value

        return jsonify({"prediction": prediction_value})

    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    # Use PORT environment variable for deployment (Render, Heroku, etc.)
    import os
    port = int(os.environ.get('PORT', 8000))  # Changed default to 8000 to avoid macOS AirPlay
    app.run(host='0.0.0.0', port=port)
