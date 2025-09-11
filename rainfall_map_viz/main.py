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



# Create a pixel grid: 1 row, N months (horizontal rectangle)
pixel_data = df['Total'].values.astype(float)
pixel_grid = pixel_data.reshape(1, -1)
# Make the chart a horizontal rectangle
scale_y = 10
scale_x = 60
pixel_grid_big = np.kron(pixel_grid, np.ones((scale_y, scale_x)))



# Make the canvas a horizontal rectangle
fig, ax = plt.subplots(figsize=(14, 6), facecolor='#002FA7')
# Create a custom colormap: white (low alpha) to white (full), on #002FA7 background
from matplotlib.colors import LinearSegmentedColormap
white_cmap = LinearSegmentedColormap.from_list('white_alpha', [(1,1,1,0.2), (1,1,1,1)], N=256)

# Draw background
fig.patch.set_facecolor('#002FA7')
ax.set_facecolor('#002FA7')





im = ax.imshow(pixel_grid_big, cmap=white_cmap, vmin=0, vmax=pixel_data.max(), aspect='auto')

mono_font = {'fontname': 'monospace'}
ax.set_xticks(np.linspace(0, pixel_grid_big.shape[1]-scale_x/2, len(df['Month_EN'])))
ax.set_xticklabels(df['Month_EN'], rotation=45, ha='right', color='white', fontsize=12, fontname='monospace')
ax.set_yticks([])

# Remove spines and ticks for a clean look
for spine in ax.spines.values():
	spine.set_visible(False)
ax.tick_params(axis='x', colors='white', labelsize=12)

plt.title('Monthly Rainstorm Warnings (Pixel Art)', color='white', fontsize=16, pad=20, fontname='monospace')
cbar = plt.colorbar(im, ax=ax, label='Total Warnings', orientation='vertical', pad=0.02)
cbar.ax.yaxis.label.set_color('white')
cbar.outline.set_edgecolor('white')
cbar.ax.tick_params(colors='white', labelsize=12)
for label in cbar.ax.get_yticklabels():
	label.set_fontname('monospace')
plt.tight_layout()
plt.show()

# Placeholder for city coordinates and plotting logic
