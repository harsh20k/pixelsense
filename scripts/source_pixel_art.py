"""
MAS-5: Source pixel art dataset via PixelLab API v2.

Generates pixel art images across varied categories using the
PixelLab /create-image-pixflux endpoint (synchronous, returns base64 PNG).

Free tier limits:
  - 40 fast generations on signup
  - 5 daily slower generations after that (stores up to 20)
  - Max image size: 200x200 px

Rate limiting strategy:
  - Respect Retry-After header on 429 responses
  - Exponential backoff (2s → 4s → 8s, max 3 retries)
  - Configurable --delay between calls (default 2s to be conservative)

Usage:
    pip install -r requirements.txt
    cp .env.example .env  # add PIXELLAB_API_KEY
    python scripts/source_pixel_art.py [--count 10] [--out data/pixel_art]
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv()

API_BASE = "https://api.pixellab.ai/v2"
DEFAULT_OUT = Path("data/pixel_art")
DEFAULT_COUNT = 10          # conservative default — free tier has limited fast gens
IMAGE_SIZE = {"width": 64, "height": 64}

# Retry config for rate-limit / transient errors
MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds; doubles each retry

# Varied prompts across subjects to ensure dataset diversity
PROMPTS = [
    # Characters
    "brave knight in shining armor, pixel art",
    "hooded wizard casting a spell, pixel art",
    "pirate captain with a parrot, pixel art",
    "ninja warrior in black outfit, pixel art",
    "medieval archer with bow, pixel art",
    "dwarf blacksmith at forge, pixel art",
    "elf ranger in forest, pixel art",
    "skeleton warrior with sword, pixel art",
    "young hero adventurer, pixel art",
    "female mage with staff, pixel art",
    # Animals
    "orange tabby cat sitting, pixel art",
    "wolf howling at moon, pixel art",
    "dragon breathing fire, pixel art",
    "horse galloping in field, pixel art",
    "bear standing in forest, pixel art",
    "eagle soaring in sky, pixel art",
    "fox with bushy tail, pixel art",
    "frog on lily pad, pixel art",
    "owl perched on branch, pixel art",
    "deer in meadow, pixel art",
    # Landscapes / environments
    "medieval castle on hill, pixel art",
    "cozy cottage in forest, pixel art",
    "desert pyramid at sunset, pixel art",
    "snowy mountain peak, pixel art",
    "tropical beach with palm trees, pixel art",
    "dark dungeon corridor, pixel art",
    "enchanted forest path, pixel art",
    "volcano erupting at night, pixel art",
    "underwater coral reef, pixel art",
    "floating sky island, pixel art",
    # Objects / items
    "treasure chest overflowing with gold, pixel art",
    "magic potion bottle glowing, pixel art",
    "ancient sword in stone, pixel art",
    "wooden sailing ship, pixel art",
    "old lantern with flame, pixel art",
    "crystal ball with swirling mist, pixel art",
    "iron shield with crest, pixel art",
    "spell book with runes, pixel art",
    "golden crown with gems, pixel art",
    "wooden bow and quiver, pixel art",
    # Vehicles / structures
    "steam-powered airship, pixel art",
    "small wooden cart, pixel art",
    "stone bridge over river, pixel art",
    "windmill in countryside, pixel art",
    "lighthouse on rocky coast, pixel art",
    "market stall with goods, pixel art",
    "campfire with logs, pixel art",
    "well in village square, pixel art",
    "blacksmith shop exterior, pixel art",
    "tavern sign hanging, pixel art",
]


def generate_image(prompt: str, api_key: str, size: dict, seed: int | None = None) -> bytes:
    """Call PixelLab /create-image-pixflux with exponential backoff on 429."""
    payload = {
        "description": prompt,
        "image_size": size,
        "outline": "single color black outline",
        "shading": "basic shading",
        "detail": "medium detail",
        "text_guidance_scale": 8,
    }
    if seed is not None:
        payload["seed"] = seed

    for attempt in range(MAX_RETRIES + 1):
        resp = requests.post(
            f"{API_BASE}/create-image-pixflux",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )

        if resp.status_code == 429:
            # Honour Retry-After if present, else exponential backoff
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else BACKOFF_BASE ** (attempt + 1)
            if attempt < MAX_RETRIES:
                print(f"\n  Rate limited. Waiting {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})...",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            else:
                resp.raise_for_status()

        if resp.status_code == 529:
            # PixelLab-specific overload code
            wait = BACKOFF_BASE ** (attempt + 1)
            if attempt < MAX_RETRIES:
                print(f"\n  Server overloaded (529). Waiting {wait:.0f}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            else:
                resp.raise_for_status()

        resp.raise_for_status()
        break

    data = resp.json()

    # Response contains base64-encoded image
    img_b64 = data.get("image", {}).get("base64") or data.get("base64")
    if not img_b64:
        raise ValueError(f"Unexpected response shape: {list(data.keys())}")
    return base64.b64decode(img_b64)


def main():
    parser = argparse.ArgumentParser(description="Source pixel art via PixelLab API")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help=f"Number of images to generate (default {DEFAULT_COUNT})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output directory (default {DEFAULT_OUT})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed for reproducibility (incremented per image)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds to wait between API calls (default 2.0 — free tier is limited)")
    args = parser.parse_args()

    api_key = os.getenv("PIXELLAB_API_KEY")
    if not api_key:
        print("ERROR: PIXELLAB_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"

    # Load existing manifest to support idempotent re-runs
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = []

    existing = {entry["filename"] for entry in manifest}
    prompts = PROMPTS[: args.count]
    skipped = 0

    with tqdm(total=len(prompts), desc="Generating pixel art") as pbar:
        for i, prompt in enumerate(prompts):
            filename = f"pixellab_{i:04d}.png"
            out_path = args.out / filename

            if filename in existing and out_path.exists():
                pbar.set_postfix(status="skip")
                pbar.update(1)
                skipped += 1
                continue

            seed = (args.seed + i) if args.seed is not None else None
            try:
                png_bytes = generate_image(prompt, api_key, IMAGE_SIZE, seed)
            except requests.HTTPError as e:
                print(f"\nHTTP error on '{prompt}': {e}", file=sys.stderr)
                pbar.update(1)
                continue
            except Exception as e:
                print(f"\nError on '{prompt}': {e}", file=sys.stderr)
                pbar.update(1)
                continue

            out_path.write_bytes(png_bytes)

            # Read dimensions via Pillow
            with Image.open(out_path) as img:
                width, height = img.size

            manifest.append({
                "filename": filename,
                "source": "pixellab",
                "prompt": prompt,
                "width": width,
                "height": height,
                "seed": seed,
            })

            # Persist manifest after each image so partial runs are recoverable
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            pbar.set_postfix(status="ok", file=filename)
            pbar.update(1)

            if args.delay > 0:
                time.sleep(args.delay)

    total = len(manifest)
    print(f"\nDone. {total} images in {args.out}/ ({skipped} skipped, already existed).")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
