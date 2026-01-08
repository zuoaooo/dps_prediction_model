import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'data/alkohol_training_data.csv'
SAVE_PATH = 'visualization/acf_pacf.png'
LAGS = 36
SHOW_PLOTS = False  # set True to show plots interactively

print("Loading data...")
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
df.columns = df.columns.str.strip()

# detect date column (prefer MONAT)
if 'MONAT' in df.columns:
    date_col = 'MONAT'
else:
    date_col = next((c for c in ['date', 'Date', 'ds'] if c in df.columns), None)
if date_col is None:
    raise ValueError("No date column found (expected MONAT/date/Date/ds).")

# parse date
if date_col == 'MONAT':
    monat_str = df[date_col].astype(str).str.zfill(6)
    df['date'] = pd.to_datetime(monat_str, format='%Y%m', errors='coerce')
else:
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
dropped = df['date'].isna().sum()
df = df.dropna(subset=['date']).sort_values('date').set_index('date')
if len(df) == 0:
    raise ValueError("No valid date rows after parsing; check MONAT/date values.")
if dropped:
    print(f"Dropped {dropped} rows due to unparsable dates.")

# pick value column (prefer WERT)
if 'WERT' in df.columns and pd.api.types.is_numeric_dtype(df['WERT']):
    value_col = 'WERT'
else:
    value_col = next((c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])), None)
if value_col is None:
    raise ValueError("No numeric value column found for ACF/PACF.")

value_label = "Alcohol-related accidents" if value_col == 'WERT' else value_col
ts = df[value_col].dropna()

print(f"Plotting ACF/PACF for '{value_label}' with {LAGS} lags...")
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(ts, lags=LAGS, ax=axes[0])
axes[0].set_title(f"ACF ({value_label})")
plot_pacf(ts, lags=LAGS, ax=axes[1], method='ywm')
axes[1].set_title(f"PACF ({value_label})")

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {SAVE_PATH}")

if SHOW_PLOTS:
    plt.show()
