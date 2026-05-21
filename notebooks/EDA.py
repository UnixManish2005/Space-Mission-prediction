# ═══════════════════════════════════════════════════════
# notebooks/EDA.py  –  Exploratory Data Analysis Script
# Run: python notebooks/EDA.py   (or open as notebook)
# ═══════════════════════════════════════════════════════

import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data, clean_data, create_target, feature_engineering

warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid', palette='muted')

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Space_Corrected.csv')

# ── Load ──────────────────────────────────────────────
df_raw   = load_data(DATA_PATH)
df_clean = clean_data(df_raw)
df_clean = create_target(df_clean)
df       = feature_engineering(df_clean)

print("="*60)
print("EXPLORATORY DATA ANALYSIS — SPACE MISSIONS")
print("="*60)
print(f"\nShape: {df_raw.shape}")
print(f"\nColumns:\n{list(df_raw.columns)}")
print(f"\nDtypes:\n{df_raw.dtypes}")
print(f"\nMissing values:\n{df_raw.isnull().sum()}")
print(f"\nTarget distribution:\n{df['target'].value_counts()}")
print(f"\nDescriptive stats:\n{df[['rocket_cost','launch_year']].describe()}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Space Mission EDA', fontsize=16, y=1.01)

# 1. Target distribution
ax = axes[0, 0]
df['target'].value_counts().plot.bar(ax=ax, color=['#e74c3c','#2ecc71'])
ax.set_title('Success (1) vs Failure (0)')
ax.set_xticklabels(['Failure','Success'], rotation=0)

# 2. Launches by year
ax = axes[0, 1]
df.groupby('launch_year')['target'].count().plot(ax=ax, color='steelblue')
ax.set_title('Launches by Year')
ax.set_xlabel('Year'); ax.set_ylabel('Count')

# 3. Success rate by year
ax = axes[0, 2]
yr = df.groupby('launch_year')['target'].mean() * 100
yr.plot(ax=ax, color='orange')
ax.set_title('Success Rate by Year (%)')
ax.set_xlabel('Year'); ax.set_ylabel('%')

# 4. Top companies
ax = axes[1, 0]
if 'Company Name' in df.columns:
    df['Company Name'].value_counts().head(10).plot.barh(ax=ax, color='mediumpurple')
    ax.set_title('Top 10 Companies by Launches')

# 5. Rocket cost distribution
ax = axes[1, 1]
cost_data = df['rocket_cost'].dropna()
cost_data = cost_data[cost_data > 0]
sns.histplot(cost_data, ax=ax, bins=40, color='teal')
ax.set_title('Rocket Cost Distribution (M USD)')
ax.set_xlabel('Cost (M USD)')

# 6. Correlation heatmap
ax = axes[1, 2]
num_cols = ['rocket_cost','launch_year','launch_month','rocket_status_encoded','company_success_rate','target']
corr = df[[c for c in num_cols if c in df.columns]].corr()
sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
ax.set_title('Feature Correlation Heatmap')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'EDA_plots.png')
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\n[✓] EDA plots saved → {out_path}")
plt.show()
