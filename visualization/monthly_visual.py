import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Load training data
df = pd.read_csv('data/training_data.csv', encoding='utf-8')

# Filter out 'Summe' rows
df_monthly = df[df['MONAT'] != 'Summe'].copy()

# Parse date
df_monthly['MONAT_NUM'] = df_monthly['MONAT'].astype(str).str[-2:].astype(int)
df_monthly['DATE'] = df_monthly['JAHR'].astype(str) + '-' + df_monthly['MONAT_NUM'].astype(str).str.zfill(2)
df_monthly['DATE'] = pd.to_datetime(df_monthly['DATE'])

# Get unique categories
categories = df_monthly['MONATSZAHL'].unique()
print(f"Found {len(categories)} categories: {list(categories)}\n")

# Create 3 separate plots
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, category in enumerate(categories):
    cat_data = df_monthly[df_monthly['MONATSZAHL'] == category].sort_values('DATE')

    axes[idx].plot(cat_data['DATE'], cat_data['WERT'],
                   linewidth=1.5, alpha=0.8, color=colors[idx])
    
    # Get year range dynamically
    year_min = int(cat_data['JAHR'].min())
    year_max = int(cat_data['JAHR'].max())

    axes[idx].set_title(f'{category} - Monthly Trend ({year_min}-{year_max})',
                       fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Date', fontsize=11)
    axes[idx].set_ylabel('Number of Accidents', fontsize=11)
    axes[idx].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('monthly_trends.png', dpi=300, bbox_inches='tight')
print("Visualization saved to 'monthly_trends.png'")

# Show statistics
print("\n=== Statistics by Category (2000-2020) ===")
for category in categories:
    cat_data = df_monthly[df_monthly['MONATSZAHL'] == category]
    print(f"\n{category}:")
    print(f"  Mean: {cat_data['WERT'].mean():.2f}")
    print(f"  Min: {cat_data['WERT'].min()}, Max: {cat_data['WERT'].max()}")
