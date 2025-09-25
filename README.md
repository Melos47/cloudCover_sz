# Shenzhen Cloud Coverage Visualization

An atmospheric, looping visualization of Shenzhen’s cloud coverage with smooth, blue-toned dynamic contours rendered in Pygame, plus Matplotlib utilities to generate static charts. The basemap uses a deep, dark style for contrast and the highest cloud amounts shift toward a blue–magenta hue.

## What’s inside

- Pygame animation (`cloudCover_sz.py`)
  - Smooth, coherent flow using Gaussian scalar fields and soft contour shading (no particles)
  - Deep-colored Shenzhen basemap underlay (Esri Dark Gray + subtle blue tint)
  - Motion and visual intensity scale with cloud coverage
  - Keyboard controls for opacity and basemap blending
- Static charts (`cloudCover_sz_img.py`)
  - Time series and heatmap for August
  - Optional GUI pop-up with `--show`
- Sample data: `CloudSZ22.50N114.00E.csv`
- Example outputs in `outputs/` (PNG)

## Repository layout

- `cloudCover_sz.py` – Pygame animated visualization
- `cloudCover_sz_img.py` – Matplotlib plots (time series + heatmap)
- `CloudSZ22.50N114.00E.csv` – Input CSV with cloud cover data
- `requirements.txt` – Python dependencies
- `assets/`
  - `shenzhen_basemap_dark.png` – Cached deep-color basemap (auto-fetched if missing)
  - `shenzhen_basemap.png` – Legacy light basemap (not used by default)
- `outputs/`
  - `cloud_heatmap_august.png`
  - `cloud_timeseries_august.png`

## Requirements

- Python 3.9+ (tested on macOS)
- pip to install dependencies

Install dependencies:

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you don’t use a virtual environment, omit the first two lines and install system-wide or user-wide.

## Data format

The scripts expect a CSV with at least the following columns:

- `time` – Date/time in a parseable format
- `cloud_cover (%)`
- `cloud_cover_low (%)`
- `cloud_cover_mid (%)`
- `cloud_cover_high (%)`

By default, the visualization filters to August and uses the provided `CloudSZ22.50N114.00E.csv`. To use a different file, change the `DATA_FILE` constant near the top of `cloudCover_sz.py` and `cloudCover_sz_img.py` (if applicable) or replace the CSV file with the same name.

## How to run (animated visualization)

Run the Pygame visual:

```zsh
python3 cloudCover_sz.py
```

First run will try to load `assets/shenzhen_basemap_dark.png`. If not found, the app fetches a deep, dark basemap from Esri and caches it there, then applies a subtle deep-blue tint overlay for contrast.

### Controls

- Esc – Quit
- `[` / `]` – Decrease / increase cloud layer opacity
- `,` / `.` – Decrease / increase basemap opacity
- `r` – Re-fetch the basemap (deep tint is reapplied)
- `s` – Save a screenshot to `assets/preview_YYYYMMDD_HHMMSS.png`

### Visual behavior

- Total, Low, Mid, High coverage are shown as soft, blurred contour fields with darker cores and lighter edges.
- The Total layer’s color smoothly shifts toward a blue–magenta hue as coverage increases, blending with lighter blues.
- Motion gets more obvious when the total coverage is high: the flow speed scales from ~0.9× (clear) up to ~2.5× (overcast).
- Deep, dark Shenzhen basemap under the data for context.

## How to run (static charts)

Generate August plots to `outputs/`:

```zsh
python3 cloudCover_sz_img.py
```

Show the plots in a pop-up window (blocks until closed):

```zsh
python3 cloudCover_sz_img.py --show
```

If a GUI is available, the script will try common backends (Qt/Tk/Mac) to display the window.

## Configuration notes

- Basemap source: Esri Dark Gray Canvas via the public export endpoint. The cached file is `assets/shenzhen_basemap_dark.png`.
- Basemap tint: A subtle deep-blue overlay is applied for a richer, “deep color” look. You can adjust opacity in code or with `,`/`.` at runtime.
- Month filter: The animation currently filters data to August. Modify the filter in `cloudCover_sz.py` if you’d like a different period.
- Colors: Base blues are tuned for readability; the Total layer has a magenta-leaning high-coverage target for contrast.

## Troubleshooting (macOS)

- Pygame window not showing or black:
  - Ensure the right Python is active (`which python3`) and that `pygame` is installed into that environment.
  - Close other apps that may be capturing GPU resources and try again.
- Basemap not loading:
  - Check your internet connection on first run (to cache the basemap).
  - Verify `assets/` is writable. You should see `shenzhen_basemap_dark.png` appear after the fetch.
- Matplotlib `--show` does nothing:
  - Some headless setups can’t open windows. Remove `--show` to just save images, or try a different GUI backend.

## Credits

- Basemap tiles: Esri Canvas (Dark Gray Base). Data and imagery © Esri and its contributors.
- Visualization: Gaussian scalar fields, soft contour shading, and coherent flow implemented in `cloudCover_sz.py`.

## Notes for development

- Code is formatted for clarity with minimal external dependencies: `numpy`, `pandas`, `pygame`, `matplotlib`, `requests`.
- When changing public behavior (e.g., month filter or color mappings), prefer adding small constants near the top of the file for easy tuning.
- Contributions: open an issue or PR with a concise description of the change and a short before/after explanation or screenshot.
