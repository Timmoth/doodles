from PIL import Image, ImageOps
import os
from pathlib import Path
import numpy as np
import json
from datetime import datetime
import calendar

RAW_DIR = Path("raw_doodles")
OUT_DIR = Path("doodles")
OUT_DIR.mkdir(exist_ok=True)

MAX_WIDTH, MAX_HEIGHT = 1920, 1080
BORDER_RATIO = 0.2  # 20%
THUMB_SIZE = (256, 256)

def add_border_box(bbox, img_size, border_ratio=BORDER_RATIO):
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    
    border_x = int(width * border_ratio)
    border_y = int(height * border_ratio)
    
    new_left = max(0, left - border_x)
    new_top = max(0, top - border_y)
    new_right = min(img_size[0], right + border_x)
    new_bottom = min(img_size[1], bottom + border_y)
    
    return (new_left, new_top, new_right, new_bottom)

def process_image(file_path, out_path):
    img = Image.open(file_path).convert("RGB")

    # Convert to grayscale and threshold
    gray = ImageOps.grayscale(img)
    np_img = np.array(gray)
    mask = np_img < 240
    coords = np.argwhere(mask)

    if coords.size == 0:
        print(f"Skipping {file_path}, nothing found (all white).")
        return

    # Bounding box
    top_left = coords.min(axis=0)[::-1]
    bottom_right = coords.max(axis=0)[::-1]
    bbox = (*top_left, *bottom_right)

    # Crop with border
    crop_box = add_border_box(bbox, img.size, BORDER_RATIO)
    cropped = img.crop(crop_box)

    # Resize main image
    width, height = cropped.size
    if width > MAX_WIDTH or height > MAX_HEIGHT:
        cropped.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.LANCZOS)

    # Save main doodle
    cropped.save(out_path, "JPEG", quality=95)

    # Create thumbnail
    thumb = ImageOps.fit(cropped, THUMB_SIZE, Image.LANCZOS)
    thumb_path = out_path.with_name(out_path.stem + "_thumbnail.jpg")
    thumb.save(thumb_path, "JPEG", quality=90)

def parse_date_from_filename(stem, mtime):
    parts = stem.split("_")
    if len(parts) >= 4:
        try:
            year = int(parts[0]) + 2000
            month = int(parts[1])
            day = int(parts[2])
            return datetime(year, month, day)
        except Exception:
            pass

    # fallback: file modification date
    return datetime.fromtimestamp(mtime)

def main():
    metadata = []
    for file_path in RAW_DIR.glob("*.jp*g"):
        out_path = OUT_DIR / file_path.name
        if out_path.exists():
            print(f"Skipping {file_path.name}, already exists.")
            continue
        process_image(file_path, out_path)
        print(f"Processed {file_path.name}")

    # Build metadata list
    for img_path in OUT_DIR.glob("*.jp*g"):
        if img_path.stem.endswith("_thumbnail"):
            continue
        thumb_path = img_path.with_name(img_path.stem + "_thumbnail.jpg")
        mtime = img_path.stat().st_mtime

        parts = img_path.stem.split("_")
        if len(parts) >= 4:
            display_name = " ".join(parts[3:])
        else:
            display_name = img_path.stem.replace("_", " ")

        dt = parse_date_from_filename(img_path.stem, mtime)

        metadata.append({
            "name": display_name,
            "thumbnail": thumb_path.name,
            "fullsize": img_path.name,
            "timestamp": dt.isoformat()
        })

    # Sort newest first
    metadata.sort(key=lambda x: x["timestamp"], reverse=True)

    with open(OUT_DIR / "gallery.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()
