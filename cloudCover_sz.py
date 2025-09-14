# cloudCover_sz.py
# Visualize cloud cover data for Shenzhen from CloudSZ22.50N114.00E.csv
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Read only the first data block (cloud cover), stop at the next header

    # Skip the first two metadata lines and the empty line, then read until the next header
    with open('CloudSZ22.50N114.00E.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Find where the second header starts
    for i, line in enumerate(lines):
        if line.strip() == 'time,sunset (iso8601)':
            end_idx = i
            break
    else:
        end_idx = len(lines)
    # The cloud cover data starts at line 4 (index 3)
    from io import StringIO
    cloud_data = ''.join(lines[3:end_idx])
    df = pd.read_csv(StringIO(cloud_data))
    df.columns = df.columns.str.strip()
    df['time'] = pd.to_datetime(df['time'])

    plt.figure(figsize=(14, 6))
    plt.plot(df['time'], df['cloud_cover (%)'], label='Total Cloud Cover', color='#002FA7', linewidth=2)
    plt.plot(df['time'], df['cloud_cover_low (%)'], label='Low Cloud', color='#7EC8E3', linestyle='--')
    plt.plot(df['time'], df['cloud_cover_mid (%)'], label='Mid Cloud', color='#357ABD', linestyle='-.')
    plt.plot(df['time'], df['cloud_cover_high (%)'], label='High Cloud', color='#6A5ACD', linestyle=':')
    plt.xlabel('Time', fontsize=13)
    plt.ylabel('Cloud Cover (%)', fontsize=13)
    plt.title('Cloud Cover in Shenzhen (Hourly)', fontsize=15, color='#002FA7')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
