"""
cloudCover_sz.py

Generative art visualization (pygame) of Shenzhen cloud cover data.

This script reads the first data block from `CloudSZ22.50N114.00E.csv`, filters
for August entries, and visualizes each timestamp as a burst of horizontally
moving particles. Particle counts, speeds and blue tones are derived from the
cloud cover percentages (low/mid/high/total).

Run with: python cloudCover_sz.py

Requirements: `pandas`, `pygame`
"""

import sys
import random
import math
from io import StringIO
import pandas as pd

try:
    import pygame
except Exception as e:
    print("This script requires pygame. Install with: pip install pygame")
    raise


DATA_FILE = 'CloudSZ22.50N114.00E.csv'


def load_cloud_block(path=DATA_FILE):
    """Read the first data block from the CSV and return a cleaned DataFrame."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find where the second header starts (if present)
    for i, line in enumerate(lines):
        if line.strip().startswith('time,') and 'sunset' in line:
            end_idx = i
            break
    else:
        end_idx = len(lines)

    # The cloud cover data in the provided file begins at line index 3
    cloud_data = ''.join(lines[3:end_idx])
    df = pd.read_csv(StringIO(cloud_data))
    df.columns = df.columns.str.strip()
    df['time'] = pd.to_datetime(df['time'])
    return df


class Particle:
    def __init__(self, x, y, vx, vy, size, color, alpha, screen_w):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.color = color
        # alpha lifecycle: particles fade in to target_alpha, then fade out
        self.target_alpha = alpha
        self.alpha = 0.0
        # fade_in duration randomized a bit for organic look (seconds)
        self.fade_in = 0.3 + random.random() * 0.9
        self.age = 0.0
        # fade_rate: how many alpha units per second the particle loses after fade-in
        self.fade_rate = 10 + random.random() * 40
        self.screen_w = screen_w

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        # update age and handle fade-in then fade-out
        self.age += dt
        if self.age <= self.fade_in:
            # ramp alpha from 0 to target_alpha
            t = self.age / self.fade_in
            self.alpha = self.target_alpha * t
        else:
            # after fade-in, reduce alpha over time
            self.alpha -= self.fade_rate * dt
            if self.alpha < 0:
                self.alpha = 0
        if self.x - self.size > self.screen_w:
            self.x = -self.size
        # wrap vertically so particles stay in view
        if self.y - self.size > pygame.display.get_surface().get_height():
            self.y = -self.size
        elif self.y + self.size < 0:
            self.y = pygame.display.get_surface().get_height() + self.size

    def draw(self, surf):
        # draw a soft halo: multiple concentric circles with decreasing alpha
        r, g, b = self.color
        base_alpha = max(0, int(self.alpha))
        # draw outer faint halo
        outer_radius = int(self.size * 3)
        s = pygame.Surface((outer_radius * 2, outer_radius * 2), pygame.SRCALPHA)
        s.fill((0, 0, 0, 0))
        # three layers: outer glow, mid glow, core
        pygame.draw.circle(s, (r, g, b, int(base_alpha * 0.12)), (outer_radius, outer_radius), outer_radius)
        pygame.draw.circle(s, (r, g, b, int(base_alpha * 0.28)), (outer_radius, outer_radius), int(self.size * 2))
        pygame.draw.circle(s, (r, g, b, base_alpha), (outer_radius, outer_radius), int(self.size))
        surf.blit(s, (int(self.x - outer_radius), int(self.y - outer_radius)))


def tone_for_component(component_name):
    """Return an RGB tone for low/mid/high cloud components."""
    if component_name == 'low':
        return (173, 216, 230)  # light blue
    if component_name == 'mid':
        return (70, 130, 180)   # steel blue
    return (25, 25, 112)       # midnight blue for high


def spawn_particles_for_row(row, screen_w, screen_h, angle_deg=0):
    """Create a particle list for a single timestamp row.

    - Number of particles scales with `cloud_cover (%)`.
    - Particles are distributed vertically; speeds and sizes vary slightly.
    - Color selection is weighted by low/mid/high contributions.
    """
    total = max(0.0, float(row.get('cloud_cover (%)', 0)))
    low = float(row.get('cloud_cover_low (%)', 0))
    mid = float(row.get('cloud_cover_mid (%)', 0))
    high = float(row.get('cloud_cover_high (%)', 0))

    # Particle count: scale to keep things visible but not too dense
    count = max(8, int(total * 0.6))

    comp_sum = low + mid + high
    if comp_sum <= 0:
        weights = [('mid', 1.0)]
    else:
        weights = []
        if low > 0:
            weights.append(('low', low / comp_sum))
        if mid > 0:
            weights.append(('mid', mid / comp_sum))
        if high > 0:
            weights.append(('high', high / comp_sum))

    particles = []
    for _ in range(count):
        # choose tone by weights
        r = random.random()
        acc = 0
        chosen = 'mid'
        for name, w in weights:
            acc += w
            if r <= acc:
                chosen = name
                break

        color = tone_for_component(chosen)

        # alpha and size are influenced by total cloudiness
        alpha = 50 + (total / 100.0) * 205
        size = 2 + (total / 100.0) * 6 + random.random() * 2

        # speed magnitude: base speed with some variation
        speed = 20 + (100 - total) * 0.03 + random.random() * 40

        # keep particle trajectories parallel and horizontal
        vx = speed
        vy = 0

        x = random.random() * screen_w
        y = random.random() * screen_h
        particles.append(Particle(x, y, vx, vy, size, color, alpha, screen_w))

    return particles


def run_visualization(df):
    pygame.init()
    screen_w, screen_h = 1200, 700
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption('Shenzhen Cloud Cover — August (Generative Art)')
    clock = pygame.time.Clock()

    # Helper: time-of-day classification and palettes
    def period_for_hour(h):
        # dawn: 4-7, morning: 7-11, noon: 11-15, sunset: 15-19, evening: 19-24 & 0-4
        if 4 <= h < 7:
            return 'dawn'
        if 7 <= h < 11:
            return 'morning'
        if 11 <= h < 15:
            return 'noon'
        if 15 <= h < 19:
            return 'sunset'
        return 'evening'

    palettes = {
        'dawn': {
            'bg_layers': [
                ((18, 24, 40), (60, 80, 110), 1.0),  # deep pre-dawn
                ((40, 70, 110), (120, 160, 200), 0.22),  # gentle blue warm-up
            ],
            'tint': (1.0, 0.95, 0.9)
        },
        'morning': {
            'bg_layers': [
                ((60, 90, 130), (140, 185, 220), 1.0),
                ((200, 220, 240), (240, 245, 255), 0.18),
            ],
            'tint': (1.0, 1.02, 1.04)
        },
        'noon': {
            'bg_layers': [
                ((120, 170, 220), (180, 205, 235), 1.0),
                ((200, 230, 255), (245, 250, 255), 0.14),
            ],
            'tint': (0.95, 1.0, 1.05)
        },
        'sunset': {
            'bg_layers': [
                ((18, 28, 60), (60, 90, 130), 1.0),
                ((90, 120, 160), (160, 190, 220), 0.28),
            ],
            'tint': (1.0, 0.95, 0.9)
        },
        'evening': {
            'bg_layers': [
                ((6, 12, 30), (30, 50, 80), 1.0),
                ((40, 60, 90), (80, 100, 130), 0.18),
            ],
            'tint': (0.8, 0.9, 1.0)
        }
    }

    # Filter for August
    august = df[df['time'].dt.month == 8].copy()
    if august.empty:
        print('No August data found in the CSV. Exiting.')
        return

    # Sort timestamps and convert to list of rows
    august = august.sort_values('time')
    rows = august.to_dict('records')

    particles = []
    idx = 0
    time_display_font = pygame.font.SysFont('Arial', 20)

    # (Trailing removed) particles will be drawn directly onto the screen;
    # per-particle halos and alpha fading are kept.
    # background fade alpha (0-255). Lower = more lingering smear.
    bg_fade_alpha = 240

    # spawn initial particles for first timestamp
    angle_deg = 0
    particles.extend(spawn_particles_for_row(rows[idx], screen_w, screen_h, angle_deg))

    spawn_interval_s = 2.0
    spawn_timer = 0.0

    running = True
    while running:
        dt_ms = clock.tick(60)
        dt = dt_ms / 1000.0
        spawn_timer += dt
        # slowly rotate global angle every frame for smoother progression
        angle_deg = (angle_deg + 0.2) % 360

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # advance to next timestamp every spawn_interval_s seconds and add more particles
        if spawn_timer >= spawn_interval_s:
            spawn_timer = 0.0
            idx = (idx + 1) % len(rows)
            # increment global angle by 1 degree each timestep
            angle_deg = (angle_deg + 1) % 360
            particles.extend(spawn_particles_for_row(rows[idx], screen_w, screen_h, angle_deg))

        # update particles
        for p in particles:
            p.update(dt)

        # simple lifetime/trim: keep list manageable
        if len(particles) > 2000:
            particles = particles[-1500:]

        # draw layered background gradient based on current time-of-day palette
        current_row = rows[idx]
        hour = current_row['time'].hour
        period = period_for_hour(hour)
        layers = palettes[period].get('bg_layers')
        if layers:
            # compose layered background onto a temporary surface, then blit it
            bg_surf = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            # compute blue factor from hour to shift tones over time (0-23)
            hf = (hour - 12) / 24.0
            blue_factor = 0.9 + 0.4 * math.cos(hf * 2 * math.pi)
            for top_col, bottom_col, layer_alpha in layers:
                layer_surf = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
                for y in range(screen_h):
                    t = y / screen_h
                    r = int(top_col[0] + (bottom_col[0] - top_col[0]) * t)
                    g = int(top_col[1] + (bottom_col[1] - top_col[1]) * t)
                    b = int(top_col[2] + (bottom_col[2] - top_col[2]) * t)
                    # enforce blue-dominant tone: reduce red/green, boost blue by factor
                    r = int(r * 0.6)
                    g = int(g * 0.8)
                    b = min(255, int(b * blue_factor))
                    a = int(255 * layer_alpha)
                    pygame.draw.line(layer_surf, (r, g, b, a), (0, y), (screen_w, y))
                bg_surf.blit(layer_surf, (0, 0))
            bg_surf.set_alpha(bg_fade_alpha)
            screen.blit(bg_surf, (0, 0))
        else:
            # fallback single background: draw to temp surface then composite
            bg_surf = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            top_col, bottom_col = palettes[period].get('bg', ((10, 10, 30), (80, 100, 140)))
            # compute blue factor from hour
            hf = (hour - 12) / 24.0
            blue_factor = 0.9 + 0.4 * math.cos(hf * 2 * math.pi)
            for y in range(screen_h):
                t = y / screen_h
                r = int(top_col[0] + (bottom_col[0] - top_col[0]) * t)
                g = int(top_col[1] + (bottom_col[1] - top_col[1]) * t)
                b = int(top_col[2] + (bottom_col[2] - top_col[2]) * t)
                r = int(r * 0.6)
                g = int(g * 0.8)
                b = min(255, int(b * blue_factor))
                pygame.draw.line(bg_surf, (r, g, b), (0, y), (screen_w, y))
            bg_surf.set_alpha(bg_fade_alpha)
            screen.blit(bg_surf, (0, 0))

        # draw particles directly on the screen with time-of-day tint applied
        tint = palettes[period]['tint']
        for p in particles:
            r, g, b = p.color
            tr = min(255, int(r * tint[0]))
            tg = min(255, int(g * tint[1]))
            tb = min(255, int(b * tint[2]))
            orig = p.color
            p.color = (tr, tg, tb)
            p.draw(screen)
            p.color = orig

        # overlay: show current timestamp and summary
        current_row = rows[idx]
        t_text = current_row['time'].strftime('%Y-%m-%d %H:%M')
        total_cloud = current_row.get('cloud_cover (%)', 0)
        label = f'{t_text}  Total: {total_cloud:.0f}%'
        text_surf = time_display_font.render(label, True, (230, 230, 230))
        screen.blit(text_surf, (12, 12))

        pygame.display.flip()

    pygame.quit()


def main():
    try:
        df = load_cloud_block(DATA_FILE)
    except FileNotFoundError:
        print(f"Data file '{DATA_FILE}' not found in the current directory.")
        return

    run_visualization(df)


if __name__ == '__main__':
    main()
