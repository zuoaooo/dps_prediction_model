#!/usr/bin/env python3
"""
Create simplified prediction results for production:
- Read detailed results from model_experiments/mlp_prediction_results_2021.csv
- Extract only the best model (MLP_Deep_4Layer)
- Save simplified version to prediction/mlp_prediction_results_2021.csv
"""
import pandas as pd
from pathlib import Path

# Paths
project_root = Path(__file__).parent.parent
prediction_dir = project_root / "prediction"
experiments_dir = project_root / "model_experiments"

# Read the full results from model_experiments
full_results_path = experiments_dir / "mlp_prediction_results_2021.csv"
df = pd.read_csv(full_results_path)

# Extract only the best model (MLP_Deep_4Layer)
simplified_df = df[['Month', 'Month_Name', 'Actual', 'Pred_MLP_Deep_4Layer', 'Error_MLP_Deep_4Layer']].copy()

# Rename columns
simplified_df.columns = ['Month', 'Month_Name', 'Actual', 'Prediction', 'Error']

# Round values to 2 decimal places
simplified_df['Prediction'] = simplified_df['Prediction'].round(2)
simplified_df['Error'] = simplified_df['Error'].round(2)

# Save simplified version to prediction/
simplified_df.to_csv(prediction_dir / "mlp_prediction_results_2021.csv", index=False)

print("✓ Simplified prediction file created successfully")