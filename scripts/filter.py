import pandas as pd

# LOAD RAW DATA
print("LOADING RAW DATA")

df = pd.read_csv('/Users/zuoao/Desktop/monatszahlen2510_verkehrsunfaelle_30_10_25.csv', encoding='utf-8')

print(f"Total raw records: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Unique MONATSZAHL categories: {df['MONATSZAHL'].unique()}")
print(f"Unique AUSPRAEGUNG types: {df['AUSPRAEGUNG'].unique()}")

# DATASET 1: ALKOHOLUNFÄLLE ONLY (Single Category - for current prediction task)
print("\nDATASET 1: ALKOHOLUNFÄLLE and insgesamt ONLY")

alkohol_filtered = df[
    (df['MONATSZAHL'] == 'Alkoholunfälle') &
    (df['AUSPRAEGUNG'] == 'insgesamt') &
    (df['MONAT'] != 'Summe') &
    (df['WERT'].notna())
][['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT', 'WERT']].copy()

alkohol_filtered['WERT'] = alkohol_filtered['WERT'].astype(int)

# Split into training (2000-2020) and ground truth (2021)
alkohol_training = alkohol_filtered[alkohol_filtered['JAHR'] <= 2020].copy()
alkohol_truth = alkohol_filtered[alkohol_filtered['JAHR'] == 2021].copy()
print(f"Training data (2000-2020): {len(alkohol_training)} records")
print(f"Ground truth (2021): {len(alkohol_truth)} records")

# Save to CSV
alkohol_training.to_csv('data/alkohol_training_data.csv', index=False, encoding='utf-8')
alkohol_truth.to_csv('data/alkohol_ground_truth_2021.csv', index=False, encoding='utf-8')

print("Saved: data/alkohol_training_data.csv, data/alkohol_ground_truth_2021.csv")



# DATASET 2: ALL CATEGORIES (Multi-category - for future experiments)
print("\nDATASET 2: ALL CATEGORIES (ALL MONATSZAHL & ALL AUSPRAEGUNG)")

all_categories = df[
    (df['MONAT'] != 'Summe') &
    (df['WERT'].notna())
][['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT', 'WERT']].copy()

all_categories['WERT'] = all_categories['WERT'].astype(int)

# Show categories and types
unique_cats = all_categories['MONATSZAHL'].unique()
unique_ausp = all_categories['AUSPRAEGUNG'].unique()
print(f"MONATSZAHL categories: {list(unique_cats)}")
print(f"AUSPRAEGUNG types: {list(unique_ausp)}")

# Split into training (2000-2020) and ground truth (2021)
all_training = all_categories[all_categories['JAHR'] <= 2020].copy()
all_truth = all_categories[all_categories['JAHR'] == 2021].copy()
print(f"Training data (2000-2020): {len(all_training)} records")
print(f"Ground truth (2021): {len(all_truth)} records")

# Save to CSV
all_training.to_csv('data/all_categories_training_data.csv', index=False, encoding='utf-8')
all_truth.to_csv('data/all_categories_ground_truth_2021.csv', index=False, encoding='utf-8')

print("Saved: data/all_categories_training_data.csv, data/all_categories_ground_truth_2021.csv")

