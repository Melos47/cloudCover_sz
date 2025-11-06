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
        w = 1.0 - val
        
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
        surf_small = pygame.image.frombuffer(rgba.tobytes(), (self.gw, self.gh), 'RGBA')
        surf_small = surf_small.convert_alpha()
        
        # smooth scale to final size
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
    # Retro-futuristic layout: left sidebar (280px) + main viz area
    sidebar_w = 280
    screen_w, screen_h = 1200, 700
    viz_w = screen_w - sidebar_w
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption('SHENZHEN CLOUD MONITOR v3.1')
    clock = pygame.time.Clock()
    
    # Geographic bounds for Shenzhen area (approximate)
    # Map covers roughly: 22.4N-22.8N, 113.8E-114.3E
    lat_min, lat_max = 22.4, 22.8
    lon_min, lon_max = 113.8, 114.3
    
    # Shenzhen districts and landmarks
    shenzhen_areas = [
        {'name': 'Futian District', 'lat': 22.52, 'lon': 114.06},
        {'name': 'Luohu District', 'lat': 22.55, 'lon': 114.13},
        {'name': 'Nanshan District', 'lat': 22.53, 'lon': 113.93},
        {'name': 'Yantian District', 'lat': 22.58, 'lon': 114.24},
        {'name': 'Baoan District', 'lat': 22.56, 'lon': 113.88},
        {'name': 'Longgang District', 'lat': 22.72, 'lon': 114.25},
        {'name': 'Longhua District', 'lat': 22.65, 'lon': 114.03},
        {'name': 'Pingshan District', 'lat': 22.70, 'lon': 114.35},
        {'name': 'Guangming District', 'lat': 22.75, 'lon': 113.95},
        {'name': 'Dapeng Peninsula', 'lat': 22.60, 'lon': 114.48},
    ]
    
    def get_location_info(mouse_x, mouse_y):
        """Convert mouse position to lat/lon and find nearest area."""
        # Adjust for sidebar offset
        if mouse_x < sidebar_w:
            return None
        
        viz_x = mouse_x - sidebar_w
        # Convert screen position to lat/lon
        lon = lon_min + (viz_x / viz_w) * (lon_max - lon_min)
        lat = lat_max - (mouse_y / screen_h) * (lat_max - lat_min)
        
        # Find nearest area
        min_dist = float('inf')
        nearest_area = 'Shenzhen'
        for area in shenzhen_areas:
            dist = ((lat - area['lat'])**2 + (lon - area['lon'])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                nearest_area = area['name']
        
        return {'name': nearest_area, 'lat': lat, 'lon': lon}
    
    # Custom color palette
    RETRO_BG = (14, 47, 100)          # Primary theme color - deep blue background
    RETRO_PANEL = (14, 47, 100)       # Sidebar panel - same as background
    RETRO_BORDER = (101, 216, 223)    # Accent color - cyan borders
    RETRO_TEXT = (89, 170, 245)       # Text color - bright blue
    RETRO_ACCENT = (64, 109, 242)     # Accent color - vibrant blue
    RETRO_DIM = (99, 100, 138)        # Dimmed text - muted purple-gray
    RETRO_SECONDARY = (64, 109, 242)  # Secondary theme color - teal-gray

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
    # Tech-style monospace fonts - using Menlo (macOS code editor font)
    font_title = pygame.font.SysFont('menlo', 18, bold=True)
    font_label = pygame.font.SysFont('menlo', 14)
    font_value = pygame.font.SysFont('menlo', 16, bold=True)
    font_small = pygame.font.SysFont('menlo', 12)

    # background fade strength (0-255). Lower = more motion smear
    bg_fade_alpha = 240

    # Optional basemap: try to load or fetch a Shenzhen map (sized for viz area)
    basemap_alpha = 180  # 0..255 (subtle for retro look)
    basemap_raw = load_or_fetch_basemap(viz_w, screen_h)
    basemap_surf = None
    if basemap_raw is not None:
        # scale/crop to fill while preserving aspect ratio
        rw, rh = basemap_raw.get_size()
        scale = max(viz_w / rw, screen_h / rh)
        tw, th = int(rw * scale), int(rh * scale)
        scaled = pygame.transform.smoothscale(basemap_raw, (tw, th))
        x0 = max(0, (tw - viz_w) // 2)
        y0 = max(0, (th - screen_h) // 2)
        basemap_surf = pygame.Surface((viz_w, screen_h), pygame.SRCALPHA)
        basemap_surf.blit(scaled, (-x0, -y0))
        # Theme-consistent tint overlay
        tint = pygame.Surface((viz_w, screen_h), pygame.SRCALPHA)
        tint.fill((14, 47, 100, 100))  # Primary theme color with ~40% alpha
        basemap_surf.blit(tint, (0, 0))

    # disable particles; show cloud cover via dynamic contour fields only
    particles.clear()

    # Four dynamic contour layers for Total, Low, Mid, High
    # Using softer cyan-purple tones to distinguish from UI blue
    layer_total = DynamicContourLayer(
        viz_w, screen_h, scale=0.6, blobs=10, alpha=100,
        base_color=(120, 200, 220), center_darkness=0.55, edge_lightness=1.15,
        edge_alpha_floor=80, alpha_scale=170,
        detail_blobs=6, detail_sigma_factor=0.30, detail_speed=1.2,
        deep_color=(80, 150, 200)  # Shift to softer blue at high coverage
    )
    # Low: light cyan
    layer_low   = DynamicContourLayer(viz_w, screen_h, scale=0.65, blobs=8, alpha=0,
                                      base_color=(140, 210, 230), center_darkness=0.60, edge_lightness=1.20,
                                      edge_alpha_floor=85, alpha_scale=175,
                                      detail_blobs=4, detail_sigma_factor=0.30, detail_speed=1.3)
    # Mid: cyan-blue blend
    layer_mid   = DynamicContourLayer(viz_w, screen_h, scale=0.6, blobs=8, alpha=0,
                                      base_color=(110, 180, 210), center_darkness=0.55, edge_lightness=1.15,
                                      edge_alpha_floor=90, alpha_scale=180,
                                      detail_blobs=4, detail_sigma_factor=0.30, detail_speed=1.25)
    # High: deeper purple-blue
    layer_high  = DynamicContourLayer(viz_w, screen_h, scale=0.55, blobs=7, alpha=0,
                                      base_color=(100, 110, 200), center_darkness=0.35, edge_lightness=1.08,
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
                    basemap_raw = load_or_fetch_basemap(viz_w, screen_h)
                    if basemap_raw is not None:
                        rw, rh = basemap_raw.get_size()
                        scale = max(viz_w / rw, screen_h / rh)
                        tw, th = int(rw * scale), int(rh * scale)
                        scaled = pygame.transform.smoothscale(basemap_raw, (tw, th))
                        x0 = max(0, (tw - viz_w) // 2)
                        y0 = max(0, (th - screen_h) // 2)
                        basemap_surf = pygame.Surface((viz_w, screen_h), pygame.SRCALPHA)
                        basemap_surf.blit(scaled, (-x0, -y0))
                        # re-apply theme tint on refetch
                        tint = pygame.Surface((viz_w, screen_h), pygame.SRCALPHA)
                        tint.fill((14, 47, 100, 100))  # Primary theme color with ~40% alpha
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

        # === Custom themed background with subtle gradient ===
        current_row = rows[idx]
        screen.fill(RETRO_BG)
        
        # Main viz area: subtle gradient based on theme colors
        viz_surf = pygame.Surface((viz_w, screen_h))
        for y in range(screen_h):
            t = y / screen_h
            # Gradient from primary to slightly lighter
            r = int(14 + 20 * t)
            g = int(47 + 30 * t)
            b = int(100 + 40 * t)
            pygame.draw.line(viz_surf, (r, g, b), (0, y), (viz_w, y))
        
        # Draw basemap in viz area if available (with retro tint)
        if basemap_surf is not None and basemap_alpha > 0:
            basemap_surf.set_alpha(basemap_alpha)
            viz_surf.blit(basemap_surf, (0, 0))

        # apply global data opacity to layers
        layer_total.opacity_multiplier = data_opacity
        layer_low.opacity_multiplier = data_opacity
        layer_mid.opacity_multiplier = data_opacity
        layer_high.opacity_multiplier = data_opacity

        # draw dynamic contour layers (Total under components) onto viz surface
        layer_total.update(dt)
        layer_low.update(dt)
        layer_mid.update(dt)
        layer_high.update(dt)
        layer_total.draw(viz_surf)
        layer_low.draw(viz_surf)
        layer_mid.draw(viz_surf)
        layer_high.draw(viz_surf)
        
        # Blit viz surface to main screen at sidebar offset
        screen.blit(viz_surf, (sidebar_w, 0))

        # translucent contour overlay removed per request

        # === Retro UI: Left sidebar panel ===
        current_row = rows[idx]
        t_text = current_row['time'].strftime('%Y-%m-%d %H:%M')
        total_cloud = current_row.get('cloud_cover (%)', 0)
        low_cloud = current_row.get('cloud_cover_low (%)', 0)
        mid_cloud = current_row.get('cloud_cover_mid (%)', 0)
        high_cloud = current_row.get('cloud_cover_high (%)', 0)
        
        # Draw sidebar background
        sidebar_surf = pygame.Surface((sidebar_w, screen_h))
        sidebar_surf.fill(RETRO_PANEL)
        
        # Simple border
        pygame.draw.rect(sidebar_surf, RETRO_BORDER, (0, 0, sidebar_w, screen_h), 2)
        
        # Title bar
        pygame.draw.line(sidebar_surf, RETRO_BORDER, (0, 60), (sidebar_w, 60), 1)
        title_surf = font_title.render('CLOUD MONITOR', True, RETRO_ACCENT)
        sidebar_surf.blit(title_surf, (20, 18))
        
        # System time
        time_surf = font_small.render(t_text, True, RETRO_DIM)
        sidebar_surf.blit(time_surf, (20, 70))
        
        # Data section
        y_offset = 110
        
        # Total cloud cover (prominent)
        label_total = font_label.render('TOTAL COVERAGE', True, RETRO_TEXT)
        sidebar_surf.blit(label_total, (20, y_offset))
        val_total = font_value.render(f'{total_cloud:.1f}%', True, RETRO_ACCENT)
        sidebar_surf.blit(val_total, (20, y_offset + 24))
        
        # Progress bar for total
        bar_w = sidebar_w - 40
        bar_h = 16
        pygame.draw.rect(sidebar_surf, RETRO_DIM, (20, y_offset + 54, bar_w, bar_h), 1)
        fill_w = int(bar_w * (total_cloud / 100.0))
        if fill_w > 0:
            pygame.draw.rect(sidebar_surf, RETRO_ACCENT, (21, y_offset + 55, fill_w, bar_h - 2))
        
        y_offset += 90
        
        # Separator
        pygame.draw.line(sidebar_surf, RETRO_BORDER, (20, y_offset), (sidebar_w-20, y_offset), 1)
        
        y_offset += 25
        
        # Low cloud
        label_low = font_label.render('LOW', True, RETRO_TEXT)
        sidebar_surf.blit(label_low, (20, y_offset))
        val_low = font_value.render(f'{low_cloud:.1f}%', True, (140, 210, 230))
        sidebar_surf.blit(val_low, (sidebar_w - val_low.get_width() - 20, y_offset))
        
        # Mini bar
        mini_bar_w = sidebar_w - 40
        pygame.draw.rect(sidebar_surf, RETRO_DIM, (20, y_offset + 28, mini_bar_w, 10), 1)
        fill_low = int(mini_bar_w * (low_cloud / 100.0))
        if fill_low > 0:
            pygame.draw.rect(sidebar_surf, (140, 210, 230), (21, y_offset + 29, fill_low, 8))
        
        y_offset += 54
        
        # Mid cloud
        label_mid = font_label.render('MID', True, RETRO_TEXT)
        sidebar_surf.blit(label_mid, (20, y_offset))
        val_mid = font_value.render(f'{mid_cloud:.1f}%', True, (110, 180, 210))
        sidebar_surf.blit(val_mid, (sidebar_w - val_mid.get_width() - 20, y_offset))
        
        pygame.draw.rect(sidebar_surf, RETRO_DIM, (20, y_offset + 28, mini_bar_w, 10), 1)
        fill_mid = int(mini_bar_w * (mid_cloud / 100.0))
        if fill_mid > 0:
            pygame.draw.rect(sidebar_surf, (110, 180, 210), (21, y_offset + 29, fill_mid, 8))
        
        y_offset += 54
        
        # High cloud
        label_high = font_label.render('HIGH', True, RETRO_TEXT)
        sidebar_surf.blit(label_high, (20, y_offset))
        val_high = font_value.render(f'{high_cloud:.1f}%', True, (100, 110, 200))
        sidebar_surf.blit(val_high, (sidebar_w - val_high.get_width() - 20, y_offset))
        
        pygame.draw.rect(sidebar_surf, RETRO_DIM, (20, y_offset + 28, mini_bar_w, 10), 1)
        fill_high = int(mini_bar_w * (high_cloud / 100.0))
        if fill_high > 0:
            pygame.draw.rect(sidebar_surf, (100, 110, 200), (21, y_offset + 29, fill_high, 8))
        
        # Footer info
        y_footer = screen_h - 70
        pygame.draw.line(sidebar_surf, RETRO_BORDER, (0, y_footer - 10), (sidebar_w, y_footer - 10), 1)
        
        footer1 = font_small.render('SHENZHEN 22.5N 114.0E', True, RETRO_DIM)
        sidebar_surf.blit(footer1, (20, y_footer + 5))
        
        footer2 = font_small.render(f'Opacity: {int(data_opacity*100)}%', True, RETRO_DIM)
        sidebar_surf.blit(footer2, (20, y_footer + 25))
        
        footer3 = font_small.render('[/] opacity ,. map ESC quit', True, RETRO_DIM)
        sidebar_surf.blit(footer3, (12, y_footer + 36))
        
        screen.blit(sidebar_surf, (0, 0))
        
        # Mouse hover info display (bottom-right corner of viz area)
        mouse_pos = pygame.mouse.get_pos()
        location_info = get_location_info(mouse_pos[0], mouse_pos[1])
        
        if location_info and mouse_pos[0] >= sidebar_w:
            # Create semi-transparent info panel
            info_w, info_h = 300, 70
            info_x = screen_w - info_w - 15
            info_y = screen_h - info_h - 15
            
            # Background panel
            info_surf = pygame.Surface((info_w, info_h), pygame.SRCALPHA)
            info_surf.fill((14, 47, 100, 220))  # Semi-transparent background
            pygame.draw.rect(info_surf, RETRO_BORDER, (0, 0, info_w, info_h), 1)
            
            # Location name
            name_surf = font_label.render(location_info['name'], True, RETRO_ACCENT)
            info_surf.blit(name_surf, (10, 10))
            
            # Coordinates
            lat_str = f"LAT: {location_info['lat']:.4f}N"
            lon_str = f"LON: {location_info['lon']:.4f}E"
            lat_surf = font_small.render(lat_str, True, RETRO_TEXT)
            lon_surf = font_small.render(lon_str, True, RETRO_TEXT)
            info_surf.blit(lat_surf, (10, 35))
            info_surf.blit(lon_surf, (10, 52))
            
            # Blit info panel to screen
            screen.blit(info_surf, (info_x, info_y))

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
