"""Generate PWA icons for Tahoe Snow app."""
from PIL import Image, ImageDraw


def generate_icon(size, output_path):
    img = Image.new('RGB', (size, size), '#0a1420')
    draw = ImageDraw.Draw(img)

    # Simple mountain + snowflake design
    # Mountain triangle
    cx, cy = size // 2, size // 2
    s = size * 0.35
    draw.polygon([
        (cx, cy - s),
        (cx - s, cy + s * 0.7),
        (cx + s, cy + s * 0.7)
    ], fill='#1e3a5f', outline='#4a9eff')

    # Snow cap
    cap_s = s * 0.4
    draw.polygon([
        (cx, cy - s),
        (cx - cap_s, cy - s + cap_s * 1.2),
        (cx + cap_s, cy - s + cap_s * 1.2)
    ], fill='#ffffff')

    # Snowflake dots
    for dx, dy in [(-0.5, -0.8), (0.5, -0.6), (0, 0.3), (-0.6, 0.1), (0.7, -0.1)]:
        r = size * 0.015
        x, y = cx + dx * s, cy + dy * s
        draw.ellipse([x - r, y - r, x + r, y + r], fill='#7ec8e3')

    img.save(output_path)
    print(f"Generated {output_path} ({size}x{size})")


if __name__ == "__main__":
    import os
    os.makedirs("static", exist_ok=True)
    generate_icon(192, "static/icon-192.png")
    generate_icon(512, "static/icon-512.png")
    print("Done! Icons saved to static/")
