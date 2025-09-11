import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# URL for Shenzhen 2023 cloud cover data
data_url = "https://weatherspark.com/h/y/127893/2023/Historical-Weather-during-2023-in-Shenzhen-China"

# Fetch the page content
response = requests.get(data_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table containing cloud cover data (manual inspection required for exact selector)
tables = soup.find_all('table')
cloud_table = None
for table in tables:
    if 'Cloud Cover' in table.text:
        cloud_table = table
        break

# Extract data from the table
rows = cloud_table.find_all('tr') if cloud_table else []
data = []
for row in rows:
    cols = [col.get_text(strip=True) for col in row.find_all(['td', 'th'])]
    if cols:
        data.append(cols)

# Convert to DataFrame
if data:
    df = pd.DataFrame(data[1:], columns=data[0])
    print("Cloud Cover Data Table:")
    print(tabulate(df, headers='keys', tablefmt='psql'))
    # Plotting (example: plot monthly average if available)
    if 'Month' in df.columns and 'Cloud Cover' in df.columns:
        df['Cloud Cover'] = pd.to_numeric(df['Cloud Cover'].str.replace('%',''), errors='coerce')
        df.plot(x='Month', y='Cloud Cover', kind='bar', legend=False)
        plt.ylabel('Cloud Cover Rate (%)')
        plt.title('Monthly Cloud Cover Rate in Shenzhen 2023')
        plt.tight_layout()
        plt.show()
else:
    print("Cloud cover data table not found or could not be parsed.")
