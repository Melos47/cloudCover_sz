# Animated Monthly Average Warnings Pixel Map
# Uses matplotlib to animate a dark blue pixelated pattern of monthly average warnings on a map.

# Requirements: pandas, matplotlib, numpy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Load data
import os
csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SZ_rainstormWarningData.csv')
try:
	df = pd.read_csv(csv_path)
except FileNotFoundError:
	raise FileNotFoundError(f"Could not find the CSV file at {csv_path}. Please make sure 'SZ_rainstormWarningData.csv' is in the main project folder.")

# Data cleaning: filter out summary rows and handle missing values
df = df[df['时间'].str.contains('月', na=False)]
df = df.replace('-', 0)
df['总计'] = pd.to_numeric(df['总计'], errors='coerce').fillna(0)

# Sort by month (optional, for better animation)
df['month_num'] = df['时间'].str.extract(r'(\d+)年(\d+)月').apply(lambda x: int(x[0])*12+int(x[1]) if pd.notnull(x[0]) and pd.notnull(x[1]) else 0, axis=1)
df = df.sort_values('month_num')

# Create a pixel grid: 1 row, N months (or reshape for more rows if desired)
pixel_data = df['总计'].values.astype(float)
pixel_grid = pixel_data.reshape(1, -1)

fig, ax = plt.subplots(figsize=(10, 2))
im = ax.imshow(pixel_grid, cmap='Blues', vmin=0, vmax=pixel_data.max())
ax.set_xticks(np.arange(len(df['时间'])))
ax.set_xticklabels(df['时间'], rotation=45, ha='right')
ax.set_yticks([])
plt.title('Monthly Rainstorm Warnings (Pixel Map)')
plt.colorbar(im, ax=ax, label='Total Warnings')
plt.tight_layout()
plt.show()
