# Rainfall by City Map Visualization
# Fetches data from SZ_rainstormWarningData.csv and plots rainfall by city on a map.

# Requirements: pandas, matplotlib, geopandas (optional for map), or basemap

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
import os
csv_path = os.path.join(os.path.dirname(__file__), '..', 'SZ_rainstormWarningData.csv')
try:
	df = pd.read_csv(csv_path)
except FileNotFoundError:
	raise FileNotFoundError(f"Could not find the CSV file at {csv_path}. Please make sure 'SZ_rainstormWarningData.csv' is in the main project folder.")


# Rename columns to English
df = df.rename(columns={
	'时间': 'Month',
	'黄色预警': 'Yellow Warning',
	'总计': 'Total',
	'红色预警': 'Red Warning',
	'橙色预警': 'Orange Warning'
})

# Data cleaning: filter out summary rows and handle missing values
df = df[df['Month'].str.contains('月', na=False)]
df = df.replace('-', 0)
df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0)

# Convert Chinese month to English (e.g., '2023年8月' -> '2023-08')
import re
def zh_month_to_en(s):
	m = re.match(r'(\d+)年(\d+)月', s)
	if m:
		year, month = m.groups()
		return f"{year}-{int(month):02d}"
	return s
df['Month_EN'] = df['Month'].apply(zh_month_to_en)

# Sort by month (optional, for better animation)
df['month_num'] = df['Month_EN'].str.extract(r'(\d+)-(\d+)').apply(lambda x: int(x[0])*12+int(x[1]) if pd.notnull(x[0]) and pd.notnull(x[1]) else 0, axis=1)
df = df.sort_values('month_num')




# Create a grouped bar chart for yellow, orange, and red warnings
fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
month_labels = [d.replace('-', '/') for d in df['Month_EN']]
bar_width = 0.2
x = np.arange(len(month_labels))

# Convert warning columns to numeric
df['Yellow Warning'] = pd.to_numeric(df['Yellow Warning'], errors='coerce').fillna(0)
df['Orange Warning'] = pd.to_numeric(df['Orange Warning'], errors='coerce').fillna(0)
df['Red Warning'] = pd.to_numeric(df['Red Warning'], errors='coerce').fillna(0)

# Define blue shades: darkest for red, medium for orange, lightest for yellow
color_yellow = '#7EC8E3'  # light blue
color_orange = '#357ABD'  # medium blue
color_red = '#002FA7'     # dark blue

bars_yellow = ax.bar(x - bar_width, df['Yellow Warning'], width=bar_width, color=color_yellow, label='Yellow Warning')
bars_orange = ax.bar(x, df['Orange Warning'], width=bar_width, color=color_orange, label='Orange Warning')
bars_red = ax.bar(x + bar_width, df['Red Warning'], width=bar_width, color=color_red, label='Red Warning')

ax.set_xlabel('Month/Year', color='#002FA7', fontsize=14, fontname='monospace', labelpad=18)
ax.set_ylabel('Number of Warnings', color='#002FA7', fontsize=14, fontname='monospace', labelpad=10)
ax.set_title('Monthly Rainstorm Warnings by Severity', color='#002FA7', fontsize=16, pad=20, fontname='monospace')
ax.set_xticks(x)
ax.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=12, fontname='monospace', color='#002FA7')
ax.tick_params(axis='y', labelsize=12, colors='#002FA7')
for label in ax.get_yticklabels():
	label.set_fontname('monospace')
for spine in ax.spines.values():
	spine.set_visible(False)
ax.legend(fontsize=12, frameon=False)
plt.tight_layout()
plt.show()
