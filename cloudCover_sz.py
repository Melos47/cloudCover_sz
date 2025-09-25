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
import os
import random
import math
from io import StringIO
import pandas as pd
import numpy as np
import requests
from datetime import datetime

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

class ContourLayer:
    def __init__(self, width, height, bands=24, speed=20.0, scale=1.0, alpha=120):
        self.w = width
        self.h = height
        self.bands = max(4, int(bands))
        self.speed = speed
        self.scale = max(0.25, float(scale))
        self.offset = 0.0
        self.alpha = alpha
        self._target_alpha = alpha
        self.surf = self._generate_surface()

    def _generate_surface(self):
        tw = max(128, int(self.w * self.scale))
        th = max(128, int(self.h * self.scale))
        tile = pygame.Surface((tw, th))
        # generate quantized interference pattern (contour-like)
        kx1, ky1 = 0.022, 0.018
        kx2, ky2 = 0.011, -0.016
        for y in range(th):
            for x in range(tw):
                v = 0.5 + 0.5 * math.sin(x * kx1 + y * ky1)
                v += 0.5 + 0.5 * math.sin(x * kx2 + y * ky2)
                v *= 0.5
                q = math.floor(v * self.bands) / self.bands
                g = int(255 * q)
                r = int(g * 0.82)
                gg = int(g * 0.9)
                b = min(255, int(g * 1.18))
                tile.set_at((x, y), (r, gg, b))
        # upscale smoothly to screen size for better quality
        surf = pygame.transform.smoothscale(tile, (self.w, self.h))
        surf.set_alpha(self.alpha)
        return surf

    def set_cloud_factor(self, factor):
        factor = max(0.0, min(1.0, factor))
        # map cloudiness to visibility
        self._target_alpha = int(40 + factor * 180)

    def update(self, dt):
        # ease alpha toward target
        if self.alpha != self._target_alpha:
            da = self._target_alpha - self.alpha
            self.alpha += int(math.copysign(min(abs(da), 120 * dt), da))
            self.surf.set_alpha(max(0, min(255, self.alpha)))
        self.offset = (self.offset + self.speed * dt) % self.w

    def draw(self, screen):
        ox = int(self.offset)
        # wrap horizontally by drawing two parts
        right_rect = pygame.Rect(0, 0, self.w - ox, self.h)
        left_rect = pygame.Rect(self.w - ox, 0, ox, self.h)
        if right_rect.width > 0:
            screen.blit(self.surf, (ox, 0), area=right_rect)
        if left_rect.width > 0:
            screen.blit(self.surf, (0, 0), area=left_rect)


class DynamicContourLayer:
    def __init__(self, width, height, scale=0.4, blobs=5, base_sigma=None, alpha=120,
                 base_color=(80, 130, 200), center_darkness=0.55, edge_lightness=1.20,
                 edge_alpha_floor=70, alpha_scale=160,
                 detail_blobs=0, detail_sigma_factor=0.35, detail_speed=1.4,
                 deep_color=None):
        self.w = width
        self.h = height
        self.scale = max(0.2, min(1.0, float(scale)))
        self.gw = max(80, int(self.w * self.scale))
        self.gh = max(60, int(self.h * self.scale))
        self.alpha = int(alpha)
        self._target_alpha = int(alpha)
        # color shading parameters
        self.base_color0 = np.array(base_color, dtype=np.float32)
        self.base_color = self.base_color0.copy()
        self.deep_color = np.array(deep_color, dtype=np.float32) if deep_color is not None else None
        self.center_darkness = float(center_darkness)
        self.edge_lightness = float(edge_lightness)
        self.edge_alpha_floor = int(edge_alpha_floor)
        self.alpha_scale = int(alpha_scale)
        # grid coords
        xs = np.linspace(0, self.gw - 1, self.gw, dtype=np.float32)
        ys = np.linspace(0, self.gh - 1, self.gh, dtype=np.float32)
        self.X, self.Y = np.meshgrid(xs, ys)  # shape (gh, gw)
        # blobs (centers in grid coords)
        self.n = max(2, int(blobs))
        rng = np.random.default_rng()
        self.cx = rng.uniform(0, self.gw, size=self.n).astype(np.float32)
        self.cy = rng.uniform(0, self.gh, size=self.n).astype(np.float32)
        # velocities in grid units/sec
        self.vx = rng.uniform(-6, 6, size=self.n).astype(np.float32)
        self.vy = rng.uniform(-5, 5, size=self.n).astype(np.float32)
        # gaussian sigma in grid units
        self.base_sigma = base_sigma if base_sigma is not None else min(self.gw, self.gh) / 10.0
        self.sigma = float(self.base_sigma)
        self._target_sigma = float(self.sigma)
        # intensity mapping
        self.intensity = 1.0
        self._target_intensity = 1.0
        # gamma for falloff shaping
        self.gamma = 0.75
        # last scalar field and contour cache
        self.last_V = None
        self._contour_lines = None
        self._contour_timer = 0.0
        # shimmer highlight controls
        self.shimmer_amp = 0.12
        self.shimmer_threshold = 0.75
        self.shimmer_speed = 0.6
        self.phase = 0.0
        # detail field (adds small-scale motion to avoid uniform look at 100%)
        self.detail_n = max(0, int(detail_blobs))
        self.detail_sigma_factor = float(detail_sigma_factor)
        self.detail_speed = float(detail_speed)
        if self.detail_n > 0:
            self.dx = rng.uniform(0, self.gw, size=self.detail_n).astype(np.float32)
            self.dy = rng.uniform(0, self.gh, size=self.detail_n).astype(np.float32)
            self.dvx = rng.uniform(-8, 8, size=self.detail_n).astype(np.float32) * self.detail_speed
            self.dvy = rng.uniform(-8, 8, size=self.detail_n).astype(np.float32) * self.detail_speed
        else:
            self.dx = self.dy = self.dvx = self.dvy = None
        self.detail_strength = 0.0
        self._target_detail_strength = 0.0
        self._target_gamma = self.gamma
        # global opacity multiplier (0..1) applied on top of layer alpha
        self.opacity_multiplier = 1.0
        # external speed multiplier (driven by overall cloudiness)
        self.external_speed_scale = 1.0

    def set_cloud_factor(self, f):
        f = max(0.0, min(1.0, float(f)))
        # larger coverage -> larger sigma (area) and stronger intensity (darker center)
        # keep sigma growth but temper at high f to preserve structure
        self._target_sigma = self.base_sigma * (0.7 + 1.2 * f)
        self._target_intensity = 0.5 + 1.5 * f
        self._target_alpha = int(40 + 180 * f)
        # more coverage -> a touch harder falloff and more detail
        self._target_gamma = 0.80 + 0.35 * f  # toward 1.15 at f=1
        self._target_detail_strength = 0.10 + 0.65 * f  # up to 0.75
        # color blend toward deep_color at higher cloud amounts, preserving smooth blend
        if self.deep_color is not None:
            s = f ** 0.85
            self.base_color = (1.0 - s) * self.base_color0 + s * self.deep_color

    def update(self, dt):
        # move centers and wrap (scaled by external speed)
        s = max(0.2, float(self.external_speed_scale))
        self.cx = (self.cx + self.vx * dt * s) % self.gw
        self.cy = (self.cy + self.vy * dt * s) % self.gh
        # move detail centers and wrap
        if self.detail_n > 0:
            self.dx = (self.dx + self.dvx * dt * s) % self.gw
            self.dy = (self.dy + self.dvy * dt * s) % self.gh
        # ease parameters
        self.sigma += (self._target_sigma - self.sigma) * min(1.0, 1.0 * dt)
        self.intensity += (self._target_intensity - self.intensity) * min(1.0, 1.0 * dt)
        self.gamma += (self._target_gamma - self.gamma) * min(1.0, 0.8 * dt)
        self.detail_strength += (self._target_detail_strength - self.detail_strength) * min(1.0, 0.8 * dt)
        # ease alpha
        if self.alpha != self._target_alpha:
            da = self._target_alpha - self.alpha
            self.alpha += int(math.copysign(min(abs(da), 120 * dt), da))
        self._contour_timer += dt
        self.phase = (self.phase + self.shimmer_speed * dt) % 1000.0

    def draw(self, screen):
        # compute scalar field
        F = np.zeros((self.gh, self.gw), dtype=np.float32)
        s2 = 2.0 * (self.sigma ** 2)
        for i in range(self.n):
            dx = self.X - self.cx[i]
            dy = self.Y - self.cy[i]
            F += np.exp(-(dx * dx + dy * dy) / s2)
        # optional detail field
        if self.detail_n and self.detail_strength > 0.001:
            Fd = np.zeros_like(F)
            s2d = 2.0 * ((self.sigma * self.detail_sigma_factor) ** 2)
            for i in range(self.detail_n):
                dx = self.X - self.dx[i]
                dy = self.Y - self.dy[i]
                Fd += np.exp(-(dx * dx + dy * dy) / s2d)
            # mix detail additively then normalize
            F = F + self.detail_strength * Fd
        # normalize to [0,1]
        m = F.max()
        if m > 1e-6:
            V = (F / m) ** self.gamma
        else:
            V = F
        # value field for shading
        val = np.clip(V * self.intensity, 0.0, 1.0)
        w = 1.0 - val  # 0 at center (dark), 1 at edges (light)
        # compute per-channel colors by interpolating between darker center and lighter edge
        c_center = np.clip(self.base_color * self.center_darkness, 0, 255)
        c_edge = np.clip(self.base_color * self.edge_lightness, 0, 255)
        r = (c_center[0] + w * (c_edge[0] - c_center[0])).astype(np.uint8)
        g = (c_center[1] + w * (c_edge[1] - c_center[1])).astype(np.uint8)
        b = (c_center[2] + w * (c_edge[2] - c_center[2])).astype(np.uint8)
        # make edges less transparent by adding a floor to alpha; scale by eased layer alpha
        a_base = np.clip(self.edge_alpha_floor + val * self.alpha_scale, 0, 255)
        a = a_base * max(0.0, min(1.0, self.alpha / 200.0))
        # apply global opacity multiplier
        a = np.clip(a * float(self.opacity_multiplier), 0, 255).astype(np.uint8)
        rgba = np.dstack((r, g, b, a))
        # blit to small surface
        # create surface from buffer for speed
        surf_small = pygame.image.frombuffer(rgba.tobytes(), (self.gw, self.gh), 'RGBA')
        # convert_alpha to match display then scale
        surf_small = surf_small.convert_alpha()
        scaled = pygame.transform.smoothscale(surf_small, (self.w, self.h))
        screen.blit(scaled, (0, 0))
        # store for contour overlay
        self.last_V = V

    def _interp(self, p1, p2, v1, v2, level):
        if abs(v2 - v1) < 1e-6:
            t = 0.5
        else:
            t = (level - v1) / (v2 - v1)
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    def _compute_contours(self, V, levels):
        gh, gw = V.shape
        sx = self.w / float(gw)
        sy = self.h / float(gh)
        lines = []
        for level in levels:
            for y in range(gh - 1):
                for x in range(gw - 1):
                    tl = V[y, x]
                    tr = V[y, x + 1]
                    br = V[y + 1, x + 1]
                    bl = V[y + 1, x]
                    idx = 0
                    if tl >= level: idx |= 8
                    if tr >= level: idx |= 4
                    if br >= level: idx |= 2
                    if bl >= level: idx |= 1
                    if idx == 0 or idx == 15:
                        continue
                    # positions in grid coordinates
                    p_tl = (x, y)
                    p_tr = (x + 1, y)
                    p_br = (x + 1, y + 1)
                    p_bl = (x, y + 1)
                    # edge interpolation
                    # edges: top (tl-tr), right (tr-br), bottom (bl-br), left (tl-bl)
                    e = [None, None, None, None]
                    if idx in (1, 14, 13, 2, 11, 4, 7, 8, 0, 15, 3, 12, 6, 9, 5, 10):
                        pass
                    # compute intersections as needed
                    if idx in (1, 5, 13, 9):
                        e[3] = self._interp(p_tl, p_bl, tl, bl, level)
                    if idx in (8, 10, 11, 9):
                        e[0] = self._interp(p_tl, p_tr, tl, tr, level)
                    if idx in (2, 6, 7, 3):
                        e[1] = self._interp(p_tr, p_br, tr, br, level)
                    if idx in (4, 5, 7, 6):
                        e[2] = self._interp(p_bl, p_br, bl, br, level)
                    if idx in (12, 14, 10, 11):
                        e[3] = self._interp(p_tl, p_bl, tl, bl, level) if e[3] is None else e[3]
                    if idx in (12, 13, 15, 14):
                        e[0] = self._interp(p_tl, p_tr, tl, tr, level) if e[0] is None else e[0]
                    if idx in (8, 12, 10, 11):
                        e[1] = self._interp(p_tr, p_br, tr, br, level) if e[1] is None else e[1]
                    if idx in (1, 3, 5, 7):
                        e[2] = self._interp(p_bl, p_br, bl, br, level) if e[2] is None else e[2]
                    # segment pairs per case (resolve ambiguities 5 and 10 consistently)
                    table = {
                        1:  [(e[3], e[2])],
                        2:  [(e[1], e[2])],
                        3:  [(e[3], e[1])],
                        4:  [(e[0], e[1])],
                        5:  [(e[0], e[3]), (e[1], e[2])],
                        6:  [(e[0], e[2])],
                        7:  [(e[0], e[3])],
                        8:  [(e[0], e[3])],
                        9:  [(e[0], e[2])],
                        10: [(e[0], e[1]), (e[3], e[2])],
                        11: [(e[0], e[1])],
                        12: [(e[3], e[1])],
                        13: [(e[1], e[2])],
                        14: [(e[3], e[2])]
                    }
                    segs = table.get(idx, [])
                    for a, b in segs:
                        if a is None or b is None:
                            continue
                        # scale to screen coords
                        ax, ay = a[0] * sx, a[1] * sy
                        bx, by = b[0] * sx, b[1] * sy
                        lines.append(((ax, ay), (bx, by)))
        return lines

    def draw_contours(self, screen, levels=(0.35, 0.5, 0.65, 0.8), color=(80, 130, 230), width=1, alpha=80):
        if self.last_V is None:
            return
        # recompute at most 5 times per second
        if self._contour_lines is None or self._contour_timer >= 0.3:
            self._contour_lines = self._compute_contours(self.last_V, levels)
            self._contour_timer = 0.0
        if not self._contour_lines:
            return
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        adj_alpha = int(max(10, min(255, alpha)) * float(self.opacity_multiplier))
        col = (color[0], color[1], color[2], max(0, min(255, adj_alpha)))
        for (a, b) in self._contour_lines:
            pygame.draw.aaline(overlay, col, a, b)
            if width > 1:
                pygame.draw.line(overlay, col, a, b, width)
        screen.blit(overlay, (0, 0))


def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def load_or_fetch_basemap(screen_w, screen_h):
    """Load basemap from assets or fetch from Esri export if missing.

    Uses Esri Dark Gray Canvas for a deep-colored backdrop.
    Extent roughly covering Shenzhen: bbox (minLon,minLat,maxLon,maxLat).
    """
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    ensure_dir(assets_dir)
    basemap_path = os.path.join(assets_dir, 'shenzhen_basemap_dark.png')
    if os.path.exists(basemap_path):
        try:
            img = pygame.image.load(basemap_path).convert_alpha()
            return img
        except Exception:
            pass
    # Try fetch from Esri export endpoint
    # Service: Canvas/World_Dark_Gray_Base (Web Mercator)
    bbox = (113.75, 22.4, 114.5, 22.9)
    url = (
        'https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/export'
        f'?bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}'
        '&bboxSR=4326'
        f'&size={screen_w},{screen_h}'
        '&imageSR=3857'
        '&format=png32'
        '&transparent=true'
        '&f=image'
    )
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200 and resp.content:
            with open(basemap_path, 'wb') as f:
                f.write(resp.content)
            img = pygame.image.load(basemap_path).convert_alpha()
            return img
    except Exception:
        pass
    return None

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

    # background fade strength (0-255). Lower = more motion smear
    bg_fade_alpha = 240

    # Optional basemap: try to load or fetch a Shenzhen map
    basemap_alpha = 230  # 0..255 (more prominent by default)
    basemap_raw = load_or_fetch_basemap(screen_w, screen_h)
    basemap_surf = None
    if basemap_raw is not None:
        # scale/crop to fill while preserving aspect ratio
        rw, rh = basemap_raw.get_size()
        scale = max(screen_w / rw, screen_h / rh)
        tw, th = int(rw * scale), int(rh * scale)
        scaled = pygame.transform.smoothscale(basemap_raw, (tw, th))
        x0 = max(0, (tw - screen_w) // 2)
        y0 = max(0, (th - screen_h) // 2)
        basemap_surf = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
        basemap_surf.blit(scaled, (-x0, -y0))
        # deep blue tint overlay for a richer dark backdrop
        tint = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
        tint.fill((10, 20, 60, 90))  # deep blue with ~35% alpha
        basemap_surf.blit(tint, (0, 0))

    # disable particles; show cloud cover via dynamic contour fields only
    particles.clear()

    # Four dynamic contour layers for Total, Low, Mid, High
    layer_total = DynamicContourLayer(
        screen_w, screen_h, scale=0.6, blobs=10, alpha=100,
        base_color=(90, 140, 210), center_darkness=0.55, edge_lightness=1.15,
        edge_alpha_floor=80, alpha_scale=170,
        detail_blobs=6, detail_sigma_factor=0.30, detail_speed=1.2,
        deep_color=(160, 50, 200)
    )
    # Low: light blue
    layer_low   = DynamicContourLayer(screen_w, screen_h, scale=0.65, blobs=8, alpha=0,
                                      base_color=(160, 200, 255), center_darkness=0.60, edge_lightness=1.20,
                                      edge_alpha_floor=85, alpha_scale=175,
                                      detail_blobs=4, detail_sigma_factor=0.30, detail_speed=1.3)
    # Mid: medium blue
    layer_mid   = DynamicContourLayer(screen_w, screen_h, scale=0.6, blobs=8, alpha=0,
                                      base_color=(90, 150, 235), center_darkness=0.55, edge_lightness=1.15,
                                      edge_alpha_floor=90, alpha_scale=180,
                                      detail_blobs=4, detail_sigma_factor=0.30, detail_speed=1.25)
    # High: dark blue
    layer_high  = DynamicContourLayer(screen_w, screen_h, scale=0.55, blobs=7, alpha=0,
                                      base_color=(20, 60, 150), center_darkness=0.35, edge_lightness=1.08,
                                      edge_alpha_floor=90, alpha_scale=210,
                                      detail_blobs=6, detail_sigma_factor=0.28, detail_speed=1.35)

    def set_layer_alphas_from_row(row):
        t = float(row.get('cloud_cover (%)', 0)) / 100.0
        lo = float(row.get('cloud_cover_low (%)', 0)) / 100.0
        mi = float(row.get('cloud_cover_mid (%)', 0)) / 100.0
        hi = float(row.get('cloud_cover_high (%)', 0)) / 100.0
        layer_total.set_cloud_factor(t)
        layer_low.set_cloud_factor(lo)
        layer_mid.set_cloud_factor(mi)
        layer_high.set_cloud_factor(hi)
        # Increase motion speed when total coverage is high
        speed_scale = 0.9 + 1.6 * t  # 0.9x at 0%, up to ~2.5x at 100%
        layer_total.external_speed_scale = speed_scale
        layer_low.external_speed_scale = speed_scale
        layer_mid.external_speed_scale = speed_scale
        layer_high.external_speed_scale = speed_scale

    set_layer_alphas_from_row(rows[idx])

    # not used now but kept for minimal changes elsewhere
    angle_deg = 0

    spawn_interval_s = 1
    spawn_timer = 0.0

    # Global data opacity (0..1) applied to all data layers
    data_opacity = 0.7
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
                elif event.key == pygame.K_LEFTBRACKET:  # decrease data opacity
                    data_opacity = max(0.0, data_opacity - 0.05)
                elif event.key == pygame.K_RIGHTBRACKET:  # increase data opacity
                    data_opacity = min(1.0, data_opacity + 0.05)
                elif event.key == pygame.K_COMMA:  # decrease basemap opacity
                    basemap_alpha = max(0, basemap_alpha - 10)
                elif event.key == pygame.K_PERIOD:  # increase basemap opacity
                    basemap_alpha = min(255, basemap_alpha + 10)
                elif event.key == pygame.K_r:  # refetch basemap
                    basemap_raw = load_or_fetch_basemap(screen_w, screen_h)
                    if basemap_raw is not None:
                        rw, rh = basemap_raw.get_size()
                        scale = max(screen_w / rw, screen_h / rh)
                        tw, th = int(rw * scale), int(rh * scale)
                        scaled = pygame.transform.smoothscale(basemap_raw, (tw, th))
                        x0 = max(0, (tw - screen_w) // 2)
                        y0 = max(0, (th - screen_h) // 2)
                        basemap_surf = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
                        basemap_surf.blit(scaled, (-x0, -y0))
                        # no tint on refetch
                        # re-apply deep blue tint on refetch
                        tint = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
                        tint.fill((10, 20, 60, 90))  # deep blue with ~35% alpha
                        basemap_surf.blit(tint, (0, 0))
                elif event.key == pygame.K_s:  # save a screenshot
                    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
                    ensure_dir(assets_dir)
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    out_path = os.path.join(assets_dir, f'preview_{ts}.png')
                    pygame.image.save(pygame.display.get_surface(), out_path)
                    print(f'Screenshot saved to {out_path}')

        # advance to next timestamp
        if spawn_timer >= spawn_interval_s:
            spawn_timer = 0.0
            idx = (idx + 1) % len(rows)
            angle_deg = (angle_deg + 1) % 360
            set_layer_alphas_from_row(rows[idx])

        # particles disabled

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

        # Draw basemap under the cloud layers if available
        if basemap_surf is not None and basemap_alpha > 0:
            basemap_surf.set_alpha(basemap_alpha)
            screen.blit(basemap_surf, (0, 0))

        # apply global data opacity to layers
        layer_total.opacity_multiplier = data_opacity
        layer_low.opacity_multiplier = data_opacity
        layer_high.opacity_multiplier = data_opacity

        # draw dynamic contour layers (Total under components)
        layer_total.update(dt)
        layer_low.update(dt)
        layer_mid.update(dt)
        layer_high.update(dt)
        layer_total.draw(screen)
        layer_low.draw(screen)
        layer_mid.draw(screen)
        layer_high.draw(screen)

        # translucent contour overlay removed per request

        # overlay: show current timestamp and Total/Low/Mid/High
        current_row = rows[idx]
        t_text = current_row['time'].strftime('%Y-%m-%d %H:%M')
        total_cloud = current_row.get('cloud_cover (%)', 0)
        low_cloud = current_row.get('cloud_cover_low (%)', 0)
        mid_cloud = current_row.get('cloud_cover_mid (%)', 0)
        high_cloud = current_row.get('cloud_cover_high (%)', 0)
        label = f'{t_text}  Total:{total_cloud:.0f}%  Low:{low_cloud:.0f}%  Mid:{mid_cloud:.0f}%  High:{high_cloud:.0f}%'
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
