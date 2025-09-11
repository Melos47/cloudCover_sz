# Generative Art Rainstorm Warning Visualization
# Uses tidal data and pygame to create a bar chart visualization of rainstorm warning data.

# Requirements: pygame, pandas

import pygame
import pandas as pd


# Load rainstorm warning data
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
df = df.sort_values('时间')

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('Rainstorm Warning Generative Art')
    running = True

    # Bar chart parameters
    margin = 80
    bar_width = 30
    bar_gap = 20
    max_bar_height = 400
    bar_color = (70, 130, 180)
    axis_color = (220, 220, 220)
    font = pygame.font.SysFont('Arial', 16)

    months = df['时间'].tolist()
    totals = df['总计'].tolist()
    n = len(months)
    max_total = max(totals) if totals else 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((30, 30, 60))

        # Draw axes
        pygame.draw.line(screen, axis_color, (margin, margin), (margin, margin+max_bar_height), 2)
        pygame.draw.line(screen, axis_color, (margin, margin+max_bar_height), (margin+n*(bar_width+bar_gap), margin+max_bar_height), 2)

        # Draw bars
        for i, (month, total) in enumerate(zip(months, totals)):
            bar_height = int((total / max_total) * max_bar_height)
            x = margin + i * (bar_width + bar_gap)
            y = margin + max_bar_height - bar_height
            pygame.draw.rect(screen, bar_color, (x, y, bar_width, bar_height))
            # Month label
            label = font.render(month, True, axis_color)
            label_rect = label.get_rect(center=(x + bar_width//2, margin + max_bar_height + 20))
            screen.blit(label, label_rect)
            # Value label
            value_label = font.render(str(int(total)), True, axis_color)
            value_rect = value_label.get_rect(center=(x + bar_width//2, y - 10))
            screen.blit(value_label, value_rect)

        # Axis labels
        y_label = font.render('Total Warnings', True, axis_color)
        screen.blit(y_label, (10, margin + max_bar_height//2))
        x_label = font.render('Month', True, axis_color)
        screen.blit(x_label, (margin + n*(bar_width+bar_gap)//2, margin + max_bar_height + 50))

        pygame.display.flip()
    pygame.quit()

if __name__ == '__main__':
    main()
