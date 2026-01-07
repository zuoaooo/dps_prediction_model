import pandas as pd

# Load CSV file
df = pd.read_csv('/Users/zuoao/Desktop/monatszahlen2510_verkehrsunfaelle_30_10_25.csv', encoding='utf-8')

print("=== CSV Overview ===")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}\n")

print("=== First 15 rows ===")
print(df.head(15))

# Filter for Alkoholunfälle + insgesamt
alkohol = df[(df['MONATSZAHL'] == 'Alkoholunfälle') & (df['AUSPRAEGUNG'] == 'insgesamt')]
print(f"\n=== Alkoholunfälle + insgesamt ({len(alkohol)} rows) ===")
print(alkohol.head(15))
