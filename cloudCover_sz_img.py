"""
Standalone CSV visualization for CloudSZ22.50N114.00E.csv

Outputs:
- outputs/cloud_timeseries_august.png
- outputs/cloud_heatmap_august.png

Run:
    /Users/siqi/Documents/PolyU/Sem1/SD5913/cloudCover_sz/.venv/bin/python cloudCover_sz_img.py --show
"""

import os
import argparse
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib
import sys
import importlib.util

# Parse --show early so we can pick an interactive backend before pyplot is imported
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--show', action='store_true')
early_args, _ = parser.parse_known_args()

if early_args.show:
    # Try GUI backends; prefer ones actually available to avoid runtime ImportErrors
    def has_module(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None

    chosen = None
    # Prefer QtAgg if any Qt binding is installed
    if any(has_module(m) for m in ("PyQt6", "PySide6", "PyQt5", "PySide2")):
        try:
            matplotlib.use("QtAgg", force=True)
            chosen = "QtAgg"
        except Exception:
            chosen = None
    # Else try TkAgg if tkinter present and Tk >= 8.6 (Matplotlib requires >=8.6)
    if chosen is None:
        try:
            import tkinter as _tk
            if getattr(_tk, "TkVersion", 0) >= 8.6:
                matplotlib.use("TkAgg", force=True)
                chosen = "TkAgg"
        except Exception:
            chosen = None
    # Fallback to native macOS backend
    if chosen is None and sys.platform == "darwin":
        try:
            matplotlib.use("MacOSX", force=True)
            chosen = "MacOSX"
        except Exception:
            chosen = None

import matplotlib.pyplot as plt

DATA_FILE = os.path.join(os.path.dirname(__file__), 'CloudSZ22.50N114.00E.csv')


def load_cloud_block(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

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
    series = {
        'Total': aug.get('cloud_cover (%)', pd.Series([np.nan]*len(aug))),
        'Low': aug.get('cloud_cover_low (%)', pd.Series([np.nan]*len(aug))),
        'Mid': aug.get('cloud_cover_mid (%)', pd.Series([np.nan]*len(aug))),
        'High': aug.get('cloud_cover_high (%)', pd.Series([np.nan]*len(aug))),
    }
    colors = {
        'Total': '#2D6CDF',
        'Low': '#9CC7FF',
        'Mid': '#4B93FF',
        'High': '#0B2F7A',
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, s in series.items():
        ax.plot(t, s, label=name, color=colors[name], linewidth=2 if name=='Total' else 1.5, alpha=0.95)
    ax.set_title('Cloud Cover — August (Total, Low, Mid, High)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cloud cover (%)')
    ax.grid(True, color='#e7eef9', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.legend()
    fig.autofmt_xdate()
    ensure_dir(out_dir)
    path = os.path.join(out_dir, 'cloud_timeseries_august.png')
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f'Saved: {path}')
    return fig


def plot_heatmap_august(df: pd.DataFrame, out_dir: str):
    aug = df[df['time'].dt.month == 8].copy()
    if aug.empty:
        print('No August data to plot.')
        return
    aug['date'] = aug['time'].dt.date
    aug['hour'] = aug['time'].dt.hour
    pivot = aug.pivot_table(index='date', columns='hour', values='cloud_cover (%)', aggfunc='mean')
    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = np.nan
    pivot = pivot[sorted(pivot.columns)]
    fig, ax = plt.subplots(figsize=(12, max(4, 0.3 * len(pivot))))
    im = ax.imshow(pivot.values, aspect='auto', interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    ax.set_title('Total Cloud Cover Heatmap — August (Day × Hour)')
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Date')
    ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels([str(h) for h in range(0, 24, 2)])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([pd.to_datetime(str(d)).strftime('%m-%d') for d in pivot.index])
    cbar = fig.colorbar(im, ax=ax, pad=0.02); cbar.set_label('Cloud cover (%)')
    ensure_dir(out_dir)
    path = os.path.join(out_dir, 'cloud_heatmap_august.png')
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f'Saved: {path}')
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize cloud CSV (August).')
    parser.add_argument('--show', action='store_true', help='Show plots in windows after saving')
    args = parser.parse_args()

    df = load_cloud_block(DATA_FILE)
    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    f1 = plot_time_series_august(df, out_dir)
    f2 = plot_heatmap_august(df, out_dir)
    if args.show:
        plt.ioff()
        try:
            # Helpful debug: print backend so user knows what's active
            import matplotlib
            print(f"Matplotlib backend: {matplotlib.get_backend()}")
        except Exception:
            pass
        plt.show(block=True)


if __name__ == '__main__':
    main()
