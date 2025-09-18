"""
Cloud cover data visualizations for CloudSZ22.50N114.00E.csv

Generates:
- Time series (August): Total vs Low/Mid/High cloud cover
- Heatmap (August): Total cloud cover by Day (rows) x Hour (cols)

Run:
  /Users/siqi/Documents/PolyU/Sem1/SD5913/cloudCover_sz/.venv/bin/python cloud_data_viz/main.py
"""

import os
import math
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'CloudSZ22.50N114.00E.csv')


def load_cloud_block(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find where the second header starts (if present)
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if line.strip().startswith('time,') and 'sunset' in line:
            end_idx = i
            break

    cloud_data = ''.join(lines[3:end_idx])
    df = pd.read_csv(StringIO(cloud_data))
    df.columns = df.columns.str.strip()
    df['time'] = pd.to_datetime(df['time'])
    return df


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def plot_time_series_august(df: pd.DataFrame, out_dir: str):
    aug = df[df['time'].dt.month == 8].copy()
    if aug.empty:
        print('No August data to plot.')
        return
    aug = aug.sort_values('time')

    t = aug['time']
    total = aug.get('cloud_cover (%)', pd.Series([np.nan]*len(aug)))
    low = aug.get('cloud_cover_low (%)', pd.Series([np.nan]*len(aug)))
    mid = aug.get('cloud_cover_mid (%)', pd.Series([np.nan]*len(aug)))
    high = aug.get('cloud_cover_high (%)', pd.Series([np.nan]*len(aug)))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, total, color='#2D6CDF', label='Total', linewidth=2.0)
    ax.plot(t, low, color='#9CC7FF', label='Low', alpha=0.9)
    ax.plot(t, mid, color='#4B93FF', label='Mid', alpha=0.95)
    ax.plot(t, high, color='#0B2F7A', label='High', alpha=0.95)
    ax.set_title('Cloud Cover — August (Total, Low, Mid, High)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cloud cover (%)')
    ax.grid(True, color='#e7eef9', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.legend()
    fig.autofmt_xdate()
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, 'cloud_timeseries_august.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'Saved: {out_path}')


def plot_heatmap_august(df: pd.DataFrame, out_dir: str):
    aug = df[df['time'].dt.month == 8].copy()
    if aug.empty:
        print('No August data to plot.')
        return
    aug['date'] = aug['time'].dt.date
    aug['hour'] = aug['time'].dt.hour
    pivot = aug.pivot_table(index='date', columns='hour', values='cloud_cover (%)', aggfunc='mean')
    # Ensure hours 0..23 all present
    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = np.nan
    pivot = pivot[sorted(pivot.columns)]
    # Plot
    fig, ax = plt.subplots(figsize=(12, max(4, 0.3 * len(pivot))))
    im = ax.imshow(pivot.values, aspect='auto', interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    ax.set_title('Total Cloud Cover Heatmap — August (Day × Hour)')
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Date')
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([str(h) for h in range(0, 24, 2)])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([d.strftime('%m-%d') for d in pd.to_datetime(pivot.index)])
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Cloud cover (%)')
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, 'cloud_heatmap_august.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'Saved: {out_path}')


def main():
    df = load_cloud_block(DATA_FILE)
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    plot_time_series_august(df, out_dir)
    plot_heatmap_august(df, out_dir)


if __name__ == '__main__':
    main()
