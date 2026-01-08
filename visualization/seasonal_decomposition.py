import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')
SHOW_PLOTS = False  # save figures only; set True to show
VERBOSE = False     # set True to print detailed stats
PRINT_SAVE_PATHS = False  # set True if you want to print save paths
GENERATE_MONTHLY_PLOT = False  # set True to save seasonal_pattern_monthly.png

print("Loading data and running seasonal decomposition...")

# Load training data
train_df = pd.read_csv('data/alkohol_training_data.csv')
train_df['date'] = pd.to_datetime(train_df['MONAT'], format='%Y%m')
train_df = train_df.sort_values('date')
train_df = train_df.set_index('date')

print(f"Data range: {train_df.index[0].strftime('%Y-%m')} - {train_df.index[-1].strftime('%Y-%m')} (months: {len(train_df)})")

print("Decomposing (12-month period, additive)...")
# Perform seasonal decomposition
decomposition = seasonal_decompose(
    train_df['WERT'],
    model='additive',  # or 'multiplicative' depending on data characteristics
    period=12,  # 12 months = 1 year cycle
    extrapolate_trend='freq'  # Handle edge cases
)

# Extract components
observed = decomposition.observed
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
seasonal_pattern = seasonal.groupby(seasonal.index.month).mean()

# Print statistics
if VERBOSE:
    print("\n--- COMPONENT STATISTICS ---")
    print("-" * 70)

    print(f"\n1. OBSERVED (Original Data):")
    print(f"   Mean: {observed.mean():.2f}")
    print(f"   Std:  {observed.std():.2f}")
    print(f"   Min:  {observed.min():.2f}")
    print(f"   Max:  {observed.max():.2f}")

    print(f"\n2. TREND (Long-term Direction):")
    print(f"   Start: {trend.dropna().iloc[0]:.2f}")
    print(f"   End:   {trend.dropna().iloc[-1]:.2f}")
    print(f"   Change: {trend.dropna().iloc[-1] - trend.dropna().iloc[0]:.2f}")
    if trend.dropna().iloc[-1] > trend.dropna().iloc[0]:
        print(f"   → Upward trend")
    else:
        print(f"   → Downward trend")

    print(f"\n3. SEASONAL (12-month Cycle):")
    print(f"   Amplitude: {seasonal.max() - seasonal.min():.2f}")
    print(f"   Peak month: {seasonal.idxmax().strftime('%B')}")
    print(f"   Lowest month: {seasonal.idxmin().strftime('%B')}")

    print(f"\n   Monthly Pattern (deviation from trend):")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for i, month in enumerate(months, 1):
        print(f"   {month}: {seasonal_pattern[i]:+.2f}")

    print(f"\n4. RESIDUAL (Random Noise):")
    print(f"   Mean: {residual.dropna().mean():.2f} (should be ~0)")
    print(f"   Std:  {residual.dropna().std():.2f}")

    # Calculate variance explained
    total_var = observed.var()
    trend_var = trend.dropna().var()
    seasonal_var = seasonal.var()
    residual_var = residual.dropna().var()

    print(f"\n" + "-" * 70)
    print("VARIANCE EXPLAINED")
    print("-" * 70)
    print(f"Trend explains:    {trend_var/total_var*100:.1f}% of variance")
    print(f"Seasonal explains: {seasonal_var/total_var*100:.1f}% of variance")
    print(f"Residual:          {residual_var/total_var*100:.1f}% (unexplained)")

print("Saving figures...")

# Create visualization
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
fig.suptitle('Seasonal Decomposition of Alcohol-Related Accidents',
             fontsize=14, fontweight='bold', y=0.995)

# 1. Observed
axes[0].plot(observed.index, observed.values, color='#2E86AB', linewidth=1.5)
axes[0].set_ylabel('Accidents', fontsize=10, fontweight='bold')
axes[0].set_title('1. Observed (Original Data)', fontsize=11, loc='left')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(observed.index[0], observed.index[-1])

# 2. Trend
axes[1].plot(trend.index, trend.values, color='#A23B72', linewidth=2)
axes[1].set_ylabel('Accidents', fontsize=10, fontweight='bold')
axes[1].set_title('2. Trend (Long-term Direction)', fontsize=11, loc='left')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(observed.index[0], observed.index[-1])

# 3. Seasonal
axes[2].plot(seasonal.index, seasonal.values, color='#F18F01', linewidth=1.5)
axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[2].set_ylabel('Deviation', fontsize=10, fontweight='bold')
axes[2].set_title('3. Seasonal (12-month Cycle)', fontsize=11, loc='left')
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(observed.index[0], observed.index[-1])

# Highlight the repeating pattern
for year in range(observed.index[0].year, observed.index[-1].year, 2):
    axes[2].axvspan(pd.Timestamp(f'{year}-01-01'),
                    pd.Timestamp(f'{year}-12-31'),
                    alpha=0.1, color='orange')

# 4. Residual
axes[3].plot(residual.index, residual.values, color='#6A994E', linewidth=1, alpha=0.7)
axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[3].set_ylabel('Deviation', fontsize=10, fontweight='bold')
axes[3].set_xlabel('Date', fontsize=10, fontweight='bold')
axes[3].set_title('4. Residual (Random Noise)', fontsize=11, loc='left')
axes[3].grid(True, alpha=0.3)
axes[3].set_xlim(observed.index[0], observed.index[-1])

# Adjust layout
plt.tight_layout()
output_path = 'visualization/seasonal_decomposition.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)
if PRINT_SAVE_PATHS:
    print(f"Saved: {output_path}")

# Create additional visualization: Monthly seasonal pattern
if GENERATE_MONTHLY_PLOT:
    fig2, ax = plt.subplots(figsize=(10, 6))

    months_full = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    seasonal_values = [seasonal_pattern[i] for i in range(1, 13)]

    colors = ['#E63946' if v > 0 else '#06A77D' for v in seasonal_values]
    bars = ax.bar(months_full, seasonal_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, value in zip(bars, seasonal_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:+.1f}',
                ha='center', va='bottom' if value > 0 else 'top',
                fontsize=9, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=1.5)
    ax.set_ylabel('Average Deviation from Trend (accidents)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax.set_title('Average Seasonal Pattern (12-month Cycle)\n(Quick view of monthly deviation from trend)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E63946', alpha=0.8, label='Above trend (more accidents)'),
        Patch(facecolor='#06A77D', alpha=0.8, label='Below trend (fewer accidents)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    output_path2 = 'visualization/seasonal_pattern_monthly.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    if PRINT_SAVE_PATHS:
        print(f"Saved: {output_path2}")

if SHOW_PLOTS:
    plt.show()
