"""Build a 10-image evaluation dataset with ground-truth layer info.

Each test case:
  - images/<name>.png — the test image
  - groundtruth/<name>.json — expected layers:
    {
      "image_type": "ui-banner" | "poster" | "photo" | "text-heavy",
      "canvas": {"width": W, "height": H},
      "layers": [
        {"role": "background", "bbox": {...}, "color": "..."},
        {"role": "headline", "bbox": {...}, "text": "..."},
        ...
      ]
    }
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
IMG_DIR = ROOT / "images"
GT_DIR = ROOT / "groundtruth"


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Arial.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def save(name: str, img: Image.Image, gt: dict) -> None:
    img.save(IMG_DIR / f"{name}.png")
    (GT_DIR / f"{name}.json").write_text(
        json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  {name}: {len(gt['layers'])} gt layers")


# ---------------------------------------------------------------------------
# Case 1: Simple light-bg banner with headline + subhead + CTA
# ---------------------------------------------------------------------------
def case1_simple_banner():
    W, H = 1200, 600
    img = Image.new("RGB", (W, H), "#F5F5DC")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([50, 200, 500, 500], radius=20, fill="#FF6B6B")
    d.text((80, 260), "SALE", fill="#FFFFFF", font=font(120))
    d.text((600, 180), "Spring Collection", fill="#333333", font=font(60))
    d.text((600, 280), "Up to 50% OFF", fill="#FF6B6B", font=font(50))
    d.rounded_rectangle([600, 420, 900, 490], radius=30, fill="#FF6B6B")
    d.text((650, 430), "SHOP NOW", fill="#FFFFFF", font=font(32))

    gt = {
        "image_type": "ui-banner",
        "canvas": {"width": W, "height": H},
        "layers": [
            {"role": "background", "bbox": {"x": 0, "y": 0, "width": W, "height": H}, "color": "#F5F5DC"},
            {"role": "card", "bbox": {"x": 50, "y": 200, "width": 450, "height": 300}, "color": "#FF6B6B"},
            {"role": "headline", "bbox": {"x": 80, "y": 260, "width": 350, "height": 130}, "text": "SALE"},
            {"role": "headline", "bbox": {"x": 600, "y": 180, "width": 580, "height": 70}, "text": "Spring Collection"},
            {"role": "subheadline", "bbox": {"x": 600, "y": 280, "width": 500, "height": 60}, "text": "Up to 50% OFF"},
            {"role": "button", "bbox": {"x": 600, "y": 420, "width": 300, "height": 70}, "color": "#FF6B6B"},
            {"role": "button", "bbox": {"x": 650, "y": 430, "width": 240, "height": 40}, "text": "SHOP NOW"},
        ],
    }
    save("01_simple_banner", img, gt)


# ---------------------------------------------------------------------------
# Case 2: Dark gradient tech banner (hard)
# ---------------------------------------------------------------------------
def case2_tech_banner():
    W, H = 1000, 500
    img = Image.new("RGB", (W, H), "#0A0A2E")
    d = ImageDraw.Draw(img)
    for y in range(H):
        r = int(10 + y * 0.02)
        g = int(10 + y * 0.01)
        b = min(255, int(46 + y * 0.15))
        d.line([(0, y), (W, y)], fill=(r, g, b))
    d.rounded_rectangle([50, 50, 450, 450], radius=15, fill="#1A1A4E", outline="#6C63FF", width=2)
    d.text((80, 80), "AI", fill="#6C63FF", font=font(120))
    d.text((80, 220), "Powered", fill="#FFFFFF", font=font(50))
    d.text((80, 290), "Analytics", fill="#A8A8D8", font=font(50))
    d.text((500, 100), "Transform your", fill="#FFFFFF", font=font(48))
    d.text((500, 170), "data into insights", fill="#A8A8D8", font=font(48))
    d.rounded_rectangle([500, 280, 780, 340], radius=25, fill="#6C63FF")
    d.text((540, 288), "Get Started", fill="#FFFFFF", font=font(32))
    d.rounded_rectangle([500, 370, 780, 430], radius=25, outline="#6C63FF", width=2)
    d.text((550, 378), "Learn More", fill="#6C63FF", font=font(32))

    gt = {
        "image_type": "ui-banner",
        "canvas": {"width": W, "height": H},
        "layers": [
            {"role": "background", "bbox": {"x": 0, "y": 0, "width": W, "height": H}, "color": "#0F1145"},
            {"role": "card", "bbox": {"x": 50, "y": 50, "width": 400, "height": 400}, "color": "#1A1A4E"},
            {"role": "headline", "bbox": {"x": 80, "y": 80, "width": 160, "height": 140}, "text": "AI"},
            {"role": "subheadline", "bbox": {"x": 80, "y": 220, "width": 260, "height": 60}, "text": "Powered"},
            {"role": "subheadline", "bbox": {"x": 80, "y": 290, "width": 260, "height": 60}, "text": "Analytics"},
            {"role": "headline", "bbox": {"x": 500, "y": 100, "width": 420, "height": 60}, "text": "Transform your"},
            {"role": "body_text", "bbox": {"x": 500, "y": 170, "width": 450, "height": 60}, "text": "data into insights"},
            {"role": "button", "bbox": {"x": 500, "y": 280, "width": 280, "height": 60}, "color": "#6C63FF"},
            {"role": "button", "bbox": {"x": 540, "y": 288, "width": 200, "height": 44}, "text": "Get Started"},
            {"role": "button", "bbox": {"x": 500, "y": 370, "width": 280, "height": 60}},
            {"role": "button", "bbox": {"x": 550, "y": 378, "width": 200, "height": 44}, "text": "Learn More"},
        ],
    }
    save("02_tech_banner", img, gt)


# ---------------------------------------------------------------------------
# Case 3: Product card with price
# ---------------------------------------------------------------------------
def case3_product_card():
    W, H = 600, 800
    img = Image.new("RGB", (W, H), "#FFFFFF")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([40, 40, 560, 560], radius=20, fill="#F0F0F0")
    d.ellipse([150, 150, 450, 450], fill="#FF6B35")
    d.rounded_rectangle([40, 600, 200, 650], radius=25, fill="#FF6B35")
    d.text((70, 605), "NEW", fill="#FFFFFF", font=font(28))
    d.text((40, 680), "Premium Coffee", fill="#2D2D2D", font=font(44))
    d.text((40, 730), "$29.99", fill="#FF6B35", font=font(36))

    gt = {
        "image_type": "ui-banner",
        "canvas": {"width": W, "height": H},
        "layers": [
            {"role": "background", "bbox": {"x": 0, "y": 0, "width": W, "height": H}, "color": "#FFFFFF"},
            {"role": "card", "bbox": {"x": 40, "y": 40, "width": 520, "height": 520}, "color": "#F0F0F0"},
            {"role": "illustration", "bbox": {"x": 150, "y": 150, "width": 300, "height": 300}, "color": "#FF6B35"},
            {"role": "button", "bbox": {"x": 40, "y": 600, "width": 160, "height": 50}, "color": "#FF6B35"},
            {"role": "button", "bbox": {"x": 70, "y": 605, "width": 90, "height": 40}, "text": "NEW"},
            {"role": "headline", "bbox": {"x": 40, "y": 680, "width": 400, "height": 50}, "text": "Premium Coffee"},
            {"role": "headline", "bbox": {"x": 40, "y": 730, "width": 200, "height": 44}, "text": "$29.99"},
        ],
    }
    save("03_product_card", img, gt)


# ---------------------------------------------------------------------------
# Case 4: Single headline poster
# ---------------------------------------------------------------------------
def case4_simple_poster():
    W, H = 800, 1000
    img = Image.new("RGB", (W, H), "#FFD93D")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 300], fill="#FF6B6B")
    d.text((50, 80), "SUMMER", fill="#FFFFFF", font=font(100))
    d.text((50, 190), "VIBES", fill="#FFFFFF", font=font(90))
    d.text((50, 400), "Discover the beach", fill="#2D2D2D", font=font(50))
    d.text((50, 470), "this weekend", fill="#2D2D2D", font=font(50))
    d.rounded_rectangle([50, 620, 350, 700], radius=35, fill="#2D2D2D")
    d.text((90, 635), "GET TICKETS", fill="#FFFFFF", font=font(32))

    gt = {
        "image_type": "poster",
        "canvas": {"width": W, "height": H},
        "layers": [
            {"role": "background", "bbox": {"x": 0, "y": 0, "width": W, "height": H}, "color": "#FFD93D"},
            {"role": "card", "bbox": {"x": 0, "y": 0, "width": W, "height": 300}, "color": "#FF6B6B"},
            {"role": "headline", "bbox": {"x": 50, "y": 80, "width": 500, "height": 110}, "text": "SUMMER"},
            {"role": "headline", "bbox": {"x": 50, "y": 190, "width": 400, "height": 100}, "text": "VIBES"},
            {"role": "subheadline", "bbox": {"x": 50, "y": 400, "width": 500, "height": 60}, "text": "Discover the beach"},
            {"role": "subheadline", "bbox": {"x": 50, "y": 470, "width": 400, "height": 60}, "text": "this weekend"},
            {"role": "button", "bbox": {"x": 50, "y": 620, "width": 300, "height": 80}, "color": "#2D2D2D"},
            {"role": "button", "bbox": {"x": 90, "y": 635, "width": 230, "height": 50}, "text": "GET TICKETS"},
        ],
    }
    save("04_simple_poster", img, gt)


# ---------------------------------------------------------------------------
# Case 5: Sale badge + product
# ---------------------------------------------------------------------------
def case5_sale_badge():
    W, H = 900, 600
    img = Image.new("RGB", (W, H), "#EEEEEE")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([50, 50, 850, 550], radius=15, fill="#FFFFFF")
    d.ellipse([100, 100, 350, 350], fill="#FF4757")
    d.text((150, 160), "70%", fill="#FFFFFF", font=font(70))
    d.text((180, 230), "OFF", fill="#FFFFFF", font=font(50))
    d.text((420, 140), "Flash Sale", fill="#2F3542", font=font(60))
    d.text((420, 220), "Limited time only", fill="#747D8C", font=font(32))
    d.text((420, 270), "Until midnight", fill="#747D8C", font=font(32))
    d.rounded_rectangle([420, 380, 700, 450], radius=30, fill="#FF4757")
    d.text((450, 395), "SHOP NOW", fill="#FFFFFF", font=font(32))

    gt = {
        "image_type": "ui-banner",
        "canvas": {"width": W, "height": H},
        "layers": [
            {"role": "background", "bbox": {"x": 0, "y": 0, "width": W, "height": H}, "color": "#EEEEEE"},
            {"role": "card", "bbox": {"x": 50, "y": 50, "width": 800, "height": 500}, "color": "#FFFFFF"},
            {"role": "badge", "bbox": {"x": 100, "y": 100, "width": 250, "height": 250}, "color": "#FF4757"},
            {"role": "headline", "bbox": {"x": 150, "y": 160, "width": 180, "height": 80}, "text": "70%"},
            {"role": "subheadline", "bbox": {"x": 180, "y": 230, "width": 120, "height": 60}, "text": "OFF"},
            {"role": "headline", "bbox": {"x": 420, "y": 140, "width": 400, "height": 70}, "text": "Flash Sale"},
            {"role": "body_text", "bbox": {"x": 420, "y": 220, "width": 350, "height": 40}, "text": "Limited time only"},
            {"role": "body_text", "bbox": {"x": 420, "y": 270, "width": 280, "height": 40}, "text": "Until midnight"},
            {"role": "button", "bbox": {"x": 420, "y": 380, "width": 280, "height": 70}, "color": "#FF4757"},
            {"role": "button", "bbox": {"x": 450, "y": 395, "width": 220, "height": 40}, "text": "SHOP NOW"},
        ],
    }
    save("05_sale_badge", img, gt)


# ---------------------------------------------------------------------------
# Case 6: Text-heavy article header
# ---------------------------------------------------------------------------
def case6_article_header():
    W, H = 1000, 400
    img = Image.new("RGB", (W, H), "#FAFAFA")
    d = ImageDraw.Draw(img)
    d.text((50, 40), "TECHNOLOGY", fill="#E63946", font=font(20))
    d.text((50, 80), "The Future of AI", fill="#1D3557", font=font(60))
    d.text((50, 150), "in Everyday Life", fill="#1D3557", font=font(60))
    d.text((50, 240), "How artificial intelligence is reshaping", fill="#457B9D", font=font(28))
    d.text((50, 280), "the way we work, learn, and connect", fill="#457B9D", font=font(28))
    d.text((50, 340), "By Jane Smith", fill="#6C757D", font=font(20))
    d.text((220, 340), "March 15, 2024", fill="#6C757D", font=font(20))

    gt = {
        "image_type": "text-heavy",
        "canvas": {"width": W, "height": H},
        "layers": [
            {"role": "background", "bbox": {"x": 0, "y": 0, "width": W, "height": H}, "color": "#FAFAFA"},
            {"role": "body_text", "bbox": {"x": 50, "y": 40, "width": 200, "height": 30}, "text": "TECHNOLOGY"},
            {"role": "headline", "bbox": {"x": 50, "y": 80, "width": 600, "height": 70}, "text": "The Future of AI"},
            {"role": "headline", "bbox": {"x": 50, "y": 150, "width": 500, "height": 70}, "text": "in Everyday Life"},
            {"role": "body_text", "bbox": {"x": 50, "y": 240, "width": 700, "height": 40}, "text": "How artificial intelligence is reshaping"},
            {"role": "body_text", "bbox": {"x": 50, "y": 280, "width": 700, "height": 40}, "text": "the way we work, learn, and connect"},
            {"role": "body_text", "bbox": {"x": 50, "y": 340, "width": 160, "height": 30}, "text": "By Jane Smith"},
            {"role": "body_text", "bbox": {"x": 220, "y": 340, "width": 200, "height": 30}, "text": "March 15, 2024"},
        ],
    }
    save("06_article_header", img, gt)


# ---------------------------------------------------------------------------
# Case 7: App screenshot style
# ---------------------------------------------------------------------------
def case7_app_screen():
    W, H = 400, 800
    img = Image.new("RGB", (W, H), "#F8F9FA")
    d = ImageDraw.Draw(img)
    # Top nav
    d.rectangle([0, 0, W, 80], fill="#FFFFFF")
    d.text((20, 30), "Explore", fill="#2D2D2D", font=font(28))
    # Card 1
    d.rounded_rectangle([20, 100, 380, 350], radius=12, fill="#FFFFFF")
    d.rectangle([30, 110, 370, 240], fill="#FFB6C1")
    d.text((30, 260), "Beach Paradise", fill="#2D2D2D", font=font(22))
    d.text((30, 290), "From $299", fill="#E63946", font=font(18))
    # Card 2
    d.rounded_rectangle([20, 380, 380, 630], radius=12, fill="#FFFFFF")
    d.rectangle([30, 390, 370, 520], fill="#87CEEB")
    d.text((30, 540), "Mountain Retreat", fill="#2D2D2D", font=font(22))
    d.text((30, 570), "From $399", fill="#E63946", font=font(18))
    # Bottom nav
    d.rectangle([0, H-80, W, H], fill="#FFFFFF")
    d.text((40, H-55), "Home", fill="#6C757D", font=font(14))
    d.text((160, H-55), "Search", fill="#2D2D2D", font=font(14))
    d.text((290, H-55), "Profile", fill="#6C757D", font=font(14))

    gt = {
        "image_type": "ui-screenshot",
        "canvas": {"width": W, "height": H},
        "layers": [
            {"role": "background", "bbox": {"x": 0, "y": 0, "width": W, "height": H}, "color": "#F8F9FA"},
            {"role": "card", "bbox": {"x": 0, "y": 0, "width": W, "height": 80}, "color": "#FFFFFF"},
            {"role": "headline", "bbox": {"x": 20, "y": 30, "width": 150, "height": 40}, "text": "Explore"},
            {"role": "card", "bbox": {"x": 20, "y": 100, "width": 360, "height": 250}, "color": "#FFFFFF"},
            {"role": "illustration", "bbox": {"x": 30, "y": 110, "width": 340, "height": 130}, "color": "#FFB6C1"},
            {"role": "headline", "bbox": {"x": 30, "y": 260, "width": 250, "height": 30}, "text": "Beach Paradise"},
            {"role": "body_text", "bbox": {"x": 30, "y": 290, "width": 150, "height": 25}, "text": "From $299"},
            {"role": "card", "bbox": {"x": 20, "y": 380, "width": 360, "height": 250}, "color": "#FFFFFF"},
            {"role": "illustration", "bbox": {"x": 30, "y": 390, "width": 340, "height": 130}, "color": "#87CEEB"},
            {"role": "headline", "bbox": {"x": 30, "y": 540, "width": 250, "height": 30}, "text": "Mountain Retreat"},
            {"role": "body_text", "bbox": {"x": 30, "y": 570, "width": 150, "height": 25}, "text": "From $399"},
            {"role": "card", "bbox": {"x": 0, "y": H-80, "width": W, "height": 80}, "color": "#FFFFFF"},
            {"role": "body_text", "bbox": {"x": 40, "y": H-55, "width": 60, "height": 20}, "text": "Home"},
            {"role": "body_text", "bbox": {"x": 160, "y": H-55, "width": 70, "height": 20}, "text": "Search"},
            {"role": "body_text", "bbox": {"x": 290, "y": H-55, "width": 70, "height": 20}, "text": "Profile"},
        ],
    }
    save("07_app_screen", img, gt)


# ---------------------------------------------------------------------------
# Case 8: Minimal logo + tagline
# ---------------------------------------------------------------------------
def case8_minimal_logo():
    W, H = 800, 600
    img = Image.new("RGB", (W, H), "#000000")
    d = ImageDraw.Draw(img)
    # Big centered logo (circle with letter)
    d.ellipse([300, 150, 500, 350], fill="#FF6B35")
    d.text((360, 200), "N", fill="#FFFFFF", font=font(120))
    # Brand
    d.text((280, 400), "NOVA", fill="#FFFFFF", font=font(70))
    # Tagline
    d.text((220, 490), "Redefining tomorrow", fill="#BBBBBB", font=font(26))

    gt = {
        "image_type": "poster",
        "canvas": {"width": W, "height": H},
        "layers": [
            {"role": "background", "bbox": {"x": 0, "y": 0, "width": W, "height": H}, "color": "#000000"},
            {"role": "logo", "bbox": {"x": 300, "y": 150, "width": 200, "height": 200}, "color": "#FF6B35"},
            {"role": "logo", "bbox": {"x": 360, "y": 200, "width": 80, "height": 120}, "text": "N"},
            {"role": "headline", "bbox": {"x": 280, "y": 400, "width": 250, "height": 80}, "text": "NOVA"},
            {"role": "body_text", "bbox": {"x": 220, "y": 490, "width": 380, "height": 40}, "text": "Redefining tomorrow"},
        ],
    }
    save("08_minimal_logo", img, gt)


# ---------------------------------------------------------------------------
# Case 9: Multi-badge promotion
# ---------------------------------------------------------------------------
def case9_multi_badge():
    W, H = 1000, 500
    img = Image.new("RGB", (W, H), "#0F4C81")
    d = ImageDraw.Draw(img)
    # 3 feature cards
    colors = ["#F4A261", "#2A9D8F", "#E76F51"]
    labels = [("Fast", "Shipping"), ("Free", "Returns"), ("24/7", "Support")]
    for i, (c, (t1, t2)) in enumerate(zip(colors, labels)):
        x = 80 + i * 300
        d.rounded_rectangle([x, 100, x + 240, 400], radius=15, fill=c)
        d.text((x + 30, 150), t1, fill="#FFFFFF", font=font(48))
        d.text((x + 30, 220), t2, fill="#FFFFFF", font=font(36))

    gt_layers = [
        {"role": "background", "bbox": {"x": 0, "y": 0, "width": W, "height": H}, "color": "#0F4C81"},
    ]
    for i, (c, (t1, t2)) in enumerate(zip(colors, labels)):
        x = 80 + i * 300
        gt_layers.append({"role": "card", "bbox": {"x": x, "y": 100, "width": 240, "height": 300}, "color": c})
        gt_layers.append({"role": "headline", "bbox": {"x": x + 30, "y": 150, "width": 180, "height": 60}, "text": t1})
        gt_layers.append({"role": "subheadline", "bbox": {"x": x + 30, "y": 220, "width": 180, "height": 44}, "text": t2})

    gt = {
        "image_type": "ui-banner",
        "canvas": {"width": W, "height": H},
        "layers": gt_layers,
    }
    save("09_multi_badge", img, gt)


# ---------------------------------------------------------------------------
# Case 10: High-contrast hero
# ---------------------------------------------------------------------------
def case10_high_contrast():
    W, H = 1600, 900
    img = Image.new("RGB", (W, H), "#000000")
    d = ImageDraw.Draw(img)
    # Red accent bar
    d.rectangle([0, 0, 20, H], fill="#FF0055")
    # Massive headline
    d.text((80, 200), "BOLD", fill="#FFFFFF", font=font(180))
    d.text((80, 400), "MOVES", fill="#FF0055", font=font(180))
    # Small CTA
    d.rectangle([80, 700, 400, 780], fill="#FFFFFF")
    d.text((130, 720), "START NOW", fill="#000000", font=font(40))
    d.text((80, 820), "Est. 2024 / New York", fill="#888888", font=font(20))

    gt = {
        "image_type": "poster",
        "canvas": {"width": W, "height": H},
        "layers": [
            {"role": "background", "bbox": {"x": 0, "y": 0, "width": W, "height": H}, "color": "#000000"},
            {"role": "decoration", "bbox": {"x": 0, "y": 0, "width": 20, "height": H}, "color": "#FF0055"},
            {"role": "headline", "bbox": {"x": 80, "y": 200, "width": 550, "height": 200}, "text": "BOLD"},
            {"role": "headline", "bbox": {"x": 80, "y": 400, "width": 700, "height": 200}, "text": "MOVES"},
            {"role": "button", "bbox": {"x": 80, "y": 700, "width": 320, "height": 80}, "color": "#FFFFFF"},
            {"role": "button", "bbox": {"x": 130, "y": 720, "width": 240, "height": 50}, "text": "START NOW"},
            {"role": "body_text", "bbox": {"x": 80, "y": 820, "width": 500, "height": 30}, "text": "Est. 2024 / New York"},
        ],
    }
    save("10_high_contrast", img, gt)


def case11_illustration_scene():
    W, H = 1000, 700
    img = Image.new("RGB", (W, H), "#87CEEB")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 500, W, H], fill="#6BAE74")
    d.ellipse([150, 100, 350, 280], fill="#FFD93D")
    for cx, cy, r, c in [(700, 120, 40, "#FFFFFF"), (800, 160, 55, "#FFFFFF"), (880, 130, 35, "#FFFFFF")]:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
    d.polygon([(300, 500), (420, 300), (540, 500)], fill="#4A7C7E")
    d.polygon([(500, 500), (640, 250), (780, 500)], fill="#3A5F61")
    d.rectangle([80, 540, 160, 620], fill="#8B4513")
    d.ellipse([40, 470, 200, 570], fill="#228B22")
    d.text((100, 640), "Nature", fill="#FFFFFF", font=font(32))
    gt = {
        "image_type": "illustration",
        "canvas": {"width": W, "height": H},
        "layers": [
            {"role": "background", "bbox": {"x": 0, "y": 0, "width": W, "height": H}, "color": "#87CEEB"},
            {"role": "decoration", "bbox": {"x": 0, "y": 500, "width": W, "height": 200}, "color": "#6BAE74"},
            {"role": "illustration", "bbox": {"x": 150, "y": 100, "width": 200, "height": 180}, "color": "#FFD93D"},
            {"role": "decoration", "bbox": {"x": 660, "y": 80, "width": 255, "height": 115}, "color": "#FFFFFF"},
            {"role": "illustration", "bbox": {"x": 300, "y": 300, "width": 240, "height": 200}, "color": "#4A7C7E"},
            {"role": "illustration", "bbox": {"x": 500, "y": 250, "width": 280, "height": 250}, "color": "#3A5F61"},
            {"role": "illustration", "bbox": {"x": 40, "y": 470, "width": 160, "height": 150}, "color": "#228B22"},
            {"role": "headline", "bbox": {"x": 100, "y": 640, "width": 200, "height": 44}, "text": "Nature"},
        ],
    }
    save("11_illustration_scene", img, gt)


def case12_photo_mixed_overlay():
    W, H = 1200, 800
    img = Image.new("RGB", (W, H), "#000000")
    d = ImageDraw.Draw(img)
    for y in range(H):
        for x in range(0, W, 4):
            r = int(100 + 80 * math.sin((x + y) * 0.01))
            g = int(50 + 40 * math.cos(y * 0.02))
            b = int(80 + 100 * math.sin(x * 0.015))
            d.rectangle([x, y, x + 4, y + 1], fill=(max(0, r), max(0, g), max(0, b)))
    d.rectangle([80, 200, 700, 600], fill="#000000")
    d.text((120, 250), "Wanderlust", fill="#FFFFFF", font=font(72))
    d.text((120, 360), "Capture every moment", fill="#DDDDDD", font=font(40))
    d.rounded_rectangle([120, 470, 400, 540], radius=6, fill="#FF6B35")
    d.text((160, 485), "Explore Now", fill="#FFFFFF", font=font(28))
    gt = {
        "image_type": "photo_mixed",
        "canvas": {"width": W, "height": H},
        "layers": [
            {"role": "background", "bbox": {"x": 0, "y": 0, "width": W, "height": H}, "color": "#1F2E4A"},
            {"role": "card", "bbox": {"x": 80, "y": 200, "width": 620, "height": 400}, "color": "#000000"},
            {"role": "headline", "bbox": {"x": 120, "y": 250, "width": 500, "height": 80}, "text": "Wanderlust"},
            {"role": "subheadline", "bbox": {"x": 120, "y": 360, "width": 550, "height": 50}, "text": "Capture every moment"},
            {"role": "button", "bbox": {"x": 120, "y": 470, "width": 280, "height": 70}, "color": "#FF6B35"},
            {"role": "button", "bbox": {"x": 160, "y": 485, "width": 200, "height": 40}, "text": "Explore Now"},
        ],
    }
    save("12_photo_mixed_overlay", img, gt)


def case13_matome_layout():
    W, H = 900, 1400
    img = Image.new("RGB", (W, H), "#FFF7EB")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 120], fill="#FF7043")
    d.text((40, 40), "Best Picks 2026", fill="#FFFFFF", font=font(44))
    def card(y, title, price, tag_color):
        d.rounded_rectangle([40, y, W - 40, y + 260], radius=12, fill="#FFFFFF")
        d.rectangle([60, y + 20, 260, y + 240], fill="#DDDDDD")
        d.rounded_rectangle([280, y + 30, 400, y + 70], radius=20, fill=tag_color)
        d.text((295, y + 38), "NEW", fill="#FFFFFF", font=font(22))
        d.text((280, y + 90), title, fill="#2D2D2D", font=font(30))
        d.text((280, y + 140), price, fill="#E63946", font=font(36))
        d.rounded_rectangle([280, y + 190, 500, y + 240], radius=8, fill="#FF7043")
        d.text((315, y + 200), "Detail", fill="#FFFFFF", font=font(26))
    titles = [("Item A", "\u00a52,980"), ("Item B", "\u00a54,200"), ("Item C", "\u00a51,680"), ("Item D", "\u00a56,500")]
    colors = ["#43A047", "#1E88E5", "#FB8C00", "#8E24AA"]
    for i, ((t, p), c) in enumerate(zip(titles, colors)):
        card(160 + i * 290, t, p, c)
    layers = [
        {"role": "background", "bbox": {"x": 0, "y": 0, "width": W, "height": H}, "color": "#FFF7EB"},
        {"role": "card", "bbox": {"x": 0, "y": 0, "width": W, "height": 120}, "color": "#FF7043"},
        {"role": "headline", "bbox": {"x": 40, "y": 40, "width": 500, "height": 50}, "text": "Best Picks 2026"},
    ]
    for i, ((t, p), c) in enumerate(zip(titles, colors)):
        y = 160 + i * 290
        layers.extend([
            {"role": "card", "bbox": {"x": 40, "y": y, "width": W - 80, "height": 260}, "color": "#FFFFFF"},
            {"role": "illustration", "bbox": {"x": 60, "y": y + 20, "width": 200, "height": 220}, "color": "#DDDDDD"},
            {"role": "button", "bbox": {"x": 280, "y": y + 30, "width": 120, "height": 40}, "color": c},
            {"role": "button", "bbox": {"x": 295, "y": y + 38, "width": 80, "height": 30}, "text": "NEW"},
            {"role": "headline", "bbox": {"x": 280, "y": y + 90, "width": 300, "height": 40}, "text": t},
            {"role": "headline", "bbox": {"x": 280, "y": y + 140, "width": 240, "height": 50}, "text": p},
            {"role": "button", "bbox": {"x": 280, "y": y + 190, "width": 220, "height": 50}, "color": "#FF7043"},
            {"role": "button", "bbox": {"x": 315, "y": y + 200, "width": 150, "height": 40}, "text": "Detail"},
        ])
    gt = {"image_type": "ui-matome", "canvas": {"width": W, "height": H}, "layers": layers}
    save("13_matome_layout", img, gt)


def main():
    print("Building eval dataset...")
    case1_simple_banner()
    case2_tech_banner()
    case3_product_card()
    case4_simple_poster()
    case5_sale_badge()
    case6_article_header()
    case7_app_screen()
    case8_minimal_logo()
    case9_multi_badge()
    case10_high_contrast()
    case11_illustration_scene()
    case12_photo_mixed_overlay()
    case13_matome_layout()
    print(f"\nDone. Written to {IMG_DIR} and {GT_DIR}")


if __name__ == "__main__":
    main()
