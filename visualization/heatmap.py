import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load training data
df = pd.read_csv('data/training_data.csv', encoding='utf-8')

# Filter out 'Summe' rows
df_monthly = df[df['MONAT'] != 'Summe'].copy()

# Parse month number
df_monthly['MONAT_NUM'] = df_monthly['MONAT'].astype(str).str[-2:].astype(int)

# Get unique categories
categories = df_monthly['MONATSZAHL'].unique()
print(f"Creating heatmaps for {len(categories)} categories: {list(categories)}\n")

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create separate heatmap for each category
for category in categories:
    # Filter data for this category
    cat_data = df_monthly[df_monthly['MONATSZAHL'] == category].copy()

    # Create pivot table: rows=years, columns=months, values=accidents
    heatmap_data = cat_data.pivot_table(
        values='WERT',
        index='JAHR',
        columns='MONAT_NUM',
        aggfunc='sum'
    )

    # Create individual figure
    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(
        heatmap_data,
        annot=True,          # Show numbers in cells
        fmt='.0f',           # Format as integer
        cmap='YlOrRd',       # Yellow-Orange-Red
        cbar_kws={'label': 'Number of Accidents'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    ax.set_title(f'{category} - Monthly Heatmap (2000-2020)',
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)
    ax.set_xticklabels(month_labels, rotation=0)

    plt.tight_layout()

    # Save with category name in filename
    filename = f'heatmap_{category.replace(" ", "_").replace("ä", "a").replace("ü", "u")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

print("\n=== Key Insights from Heatmaps ===\n")
