"""Generate .ico and .png icon files for SNP Viewer.

Pure-Python using Pillow only (no native cairo dependency).
"""
import os
import math
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SIZES = [16, 24, 32, 48, 64, 128, 256, 512]

BG_COLOR = (30, 58, 95)
GRID_COLOR = (126, 184, 224)
RED = (230, 75, 53)
CYAN = (77, 187, 213)
WHITE = (255, 255, 255)
LABEL_BLUE = (126, 184, 224)


def draw_icon(size):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size
    r = max(1, int(s * 0.18))

    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=r, fill=BG_COLOR)

    grid_alpha = 30
    for i in range(1, 6):
        y = int(s * i / 6)
        x = int(s * i / 6)
        d.line([(0, y), (s, y)], fill=(*GRID_COLOR, grid_alpha), width=1)
        d.line([(x, 0), (x, s)], fill=(*GRID_COLOR, grid_alpha), width=1)

    lw = max(1, int(s * 0.012))

    def trace_s21(frac_x):
        base_y = 0.60
        if 0.33 < frac_x < 0.45:
            t = (frac_x - 0.33) / 0.12
            peak = math.sin(t * math.pi)
            return base_y - peak * 0.42
        if 0.65 < frac_x < 0.77:
            t = (frac_x - 0.65) / 0.12
            peak = math.sin(t * math.pi)
            return base_y - peak * 0.45
        return base_y

    def trace_s11(frac_x):
        base_y = 0.68
        if 0.33 < frac_x < 0.45:
            t = (frac_x - 0.33) / 0.12
            dip = math.sin(t * math.pi)
            return base_y - dip * 0.18
        if 0.65 < frac_x < 0.77:
            t = (frac_x - 0.65) / 0.12
            dip = math.sin(t * math.pi)
            return base_y - dip * 0.20
        return base_y

    n = max(60, s)
    s21_pts = []
    s11_pts = []
    for i in range(n):
        fx = i / (n - 1)
        s21_pts.append((int(fx * s), int(trace_s21(fx) * s)))
        s11_pts.append((int(fx * s), int(trace_s11(fx) * s)))

    if len(s11_pts) > 1:
        dash_len = max(3, s // 30)
        for i in range(0, len(s11_pts) - 1, 2):
            seg = s11_pts[i:i + 2]
            if len(seg) == 2:
                d.line(seg, fill=CYAN, width=max(1, lw - 1))

    if len(s21_pts) > 1:
        d.line(s21_pts, fill=RED, width=lw + 1)

    dot_r = max(1, s // 60)
    p1x, p1y = int(0.39 * s), int(trace_s21(0.39) * s)
    p2x, p2y = int(0.71 * s), int(trace_s21(0.71) * s)
    d.ellipse([p1x - dot_r, p1y - dot_r, p1x + dot_r, p1y + dot_r], fill=RED)
    d.ellipse([p2x - dot_r, p2y - dot_r, p2x + dot_r, p2y + dot_r], fill=RED)

    if s >= 48:
        try:
            font_sz = max(8, int(s * 0.16))
            font = ImageFont.truetype("arial.ttf", font_sz)
        except OSError:
            font = ImageFont.load_default()
        bbox = d.textbbox((0, 0), "SNP", font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (s - tw) // 2
        ty = int(s * 0.78) - th // 2
        d.text((tx, ty), "SNP", fill=WHITE, font=font)

        if s >= 128:
            try:
                sub_sz = max(6, int(s * 0.045))
                sub_font = ImageFont.truetype("arial.ttf", sub_sz)
            except OSError:
                sub_font = ImageFont.load_default()
            bbox2 = d.textbbox((0, 0), "VIEWER", font=sub_font)
            tw2 = bbox2[2] - bbox2[0]
            d.text(((s - tw2) // 2, ty + th + int(s * 0.01)),
                   "VIEWER", fill=LABEL_BLUE, font=sub_font)

    mask = Image.new('L', (s, s), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, s - 1, s - 1], radius=r, fill=255)
    img.putalpha(mask)

    return img


icons = {}
for sz in SIZES:
    img = draw_icon(sz)
    icons[sz] = img
    img.save(os.path.join(OUT_DIR, f'icon_{sz}.png'))

ico_sizes = [16, 24, 32, 48, 64, 128, 256]
ico_images = [icons[s] for s in ico_sizes]
ico_images[0].save(
    os.path.join(OUT_DIR, 'snpviewer.ico'),
    format='ICO',
    sizes=[(s, s) for s in ico_sizes],
    append_images=ico_images[1:],
)

print(f"Created snpviewer.ico and {len(SIZES)} PNG files in {OUT_DIR}")
