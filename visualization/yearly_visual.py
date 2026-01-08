import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Load training data
df = pd.read_csv('data/training_data.csv', encoding='utf-8')

# Filter: only 'Summe' rows (yearly totals)
df_yearly = df[df['MONAT'] == 'Summe'].copy()

# Get unique categories
categories = df_yearly['MONATSZAHL'].unique()
print(f"Found {len(categories)} categories: {list(categories)}\n")

# Create 3 separate plots (one for each category)
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, category in enumerate(categories):
    cat_data = df_yearly[df_yearly['MONATSZAHL'] == category].sort_values('JAHR')

    axes[idx].plot(cat_data['JAHR'], cat_data['WERT'],
                   marker='o', markersize=6, linewidth=2,
                   alpha=0.8, color=colors[idx])

    # Get year range dynamically
    year_min = int(cat_data['JAHR'].min())
    year_max = int(cat_data['JAHR'].max())

    axes[idx].set_title(f'{category} - Yearly Trend ({year_min}-{year_max})',
                       fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Year', fontsize=11)
    axes[idx].set_ylabel('Total Accidents per Year', fontsize=11)
    axes[idx].grid(True, alpha=0.3, linestyle='--')

    # Set x-axis dynamically
    axes[idx].set_xticks(range(year_min, year_max + 1, 2))

    # Add some padding to y-axis for better visibility
    y_min, y_max = cat_data['WERT'].min(), cat_data['WERT'].max()
    y_margin = (y_max - y_min) * 0.1
    axes[idx].set_ylim(y_min - y_margin, y_max + y_margin)

plt.tight_layout()
plt.savefig('yearly_trends.png', dpi=300, bbox_inches='tight')
print("Visualization saved to 'yearly_trends.png'\n")