import pandas as pd

# Load CSV
df = pd.read_csv('/Users/zuoao/Desktop/monatszahlen2510_verkehrsunfaelle_30_10_25.csv', encoding='utf-8')

# Filter: only 'insgesamt' type, exclude null values
filtered = df[
    (df['AUSPRAEGUNG'] == 'insgesamt') &
    (df['WERT'].notna())
][['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT', 'WERT']].copy()

filtered['WERT'] = filtered['WERT'].astype(int)

# Training data: 2000-2020 (for building prediction model)
training_data = filtered[filtered['JAHR'] <= 2020].copy()
print(f"Training data: {len(training_data)} rows (years {training_data['JAHR'].min()}-{training_data['JAHR'].max()})")
print(training_data.tail(10).to_string(index=False))
training_data.to_csv('training_data.csv', index=False, encoding='utf-8')

# Ground truth: 2021 actual values (for validation)
ground_truth = filtered[filtered['JAHR'] == 2021].copy()
print(f"\nGround truth 2021: {len(ground_truth)} rows")
print(ground_truth.to_string(index=False))
ground_truth.to_csv('ground_truth_2021.csv', index=False, encoding='utf-8')

# Prediction target: Alkoholunfälle 2021-01
target = ground_truth[
    (ground_truth['MONATSZAHL'] == 'Alkoholunfälle') &
    (ground_truth['MONAT'] == '202101')
]
if not target.empty:
    print(f"\nPrediction target (Alkoholunfälle 2021-01): Actual value = {target['WERT'].values[0]}")
