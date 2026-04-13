"""
MAS-6: Source photorealistic dataset via Unsplash API.

Downloads photos across categories that match the pixel art dataset subjects
(nature, architecture, people, animals, objects).

Unsplash free tier limits:
  - 50 API requests / hour (search + download trigger both count)
  - Rate limit headers: X-Ratelimit-Limit, X-Ratelimit-Remaining

Rate limiting strategy:
  - Track remaining requests from response headers
  - Pause when remaining < 5 (safety buffer)
  - Respect Retry-After on 429; exponential backoff up to 3 retries
  - Configurable --delay between downloads (default 1s)

Usage:
    pip install -r requirements.txt
    cp .env.example .env  # add UNSPLASH_ACCESS_KEY
    python scripts/source_photorealistic.py [--per-topic 2] [--out data/photorealistic]

Get a free Unsplash API key at: https://unsplash.com/developers
"""

import argparse
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

UNSPLASH_API = "https://api.unsplash.com"
DEFAULT_OUT = Path("data/photorealistic")
DEFAULT_PER_TOPIC = 2       # 2 × 5 topics = 10 total (safe for free tier check)

# Unsplash free tier: 50 req/hour
RATE_LIMIT_REMAINING_FLOOR = 5   # pause when this many requests remain
RATE_LIMIT_PAUSE = 60            # seconds to pause when floor is hit

MAX_RETRIES = 3
BACKOFF_BASE = 2

# Topics mirror the pixel art dataset categories
TOPICS = [
    "nature",
    "architecture",
    "people",
    "animals",
    "objects",
]

# Shared rate-limit state updated from response headers
_remaining_requests: int | None = None


def _check_rate_limit(resp: requests.Response) -> None:
    """Update remaining-request counter; pause if floor reached."""
    global _remaining_requests
    remaining = resp.headers.get("X-Ratelimit-Remaining")
    if remaining is not None:
        _remaining_requests = int(remaining)
        if _remaining_requests <= RATE_LIMIT_REMAINING_FLOOR:
            print(
                f"\n  Unsplash rate limit floor reached ({_remaining_requests} left). "
                f"Pausing {RATE_LIMIT_PAUSE}s...",
                file=sys.stderr,
            )
            time.sleep(RATE_LIMIT_PAUSE)


def fetch_photos(topic: str, count: int, api_key: str) -> list[dict]:
    """Fetch photo metadata from Unsplash with retry on 429."""
    params = {
        "query": topic,
        "per_page": min(count, 30),  # Unsplash max per_page = 30
        "orientation": "squarish",
        "content_filter": "high",
    }
    for attempt in range(MAX_RETRIES + 1):
        resp = requests.get(
            f"{UNSPLASH_API}/search/photos",
            params=params,
            headers={"Authorization": f"Client-ID {api_key}"},
            timeout=30,
        )
        _check_rate_limit(resp)

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else BACKOFF_BASE ** (attempt + 1)
            if attempt < MAX_RETRIES:
                print(f"\n  Rate limited (429). Waiting {wait:.0f}s...", file=sys.stderr)
                time.sleep(wait)
                continue
        resp.raise_for_status()
        return resp.json().get("results", [])

    return []


def download_image(url: str, dest: Path, api_key: str) -> tuple[int, int]:
    """Stream-download an image to dest, return (width, height).

    Image CDN downloads don't count against the API rate limit, but we still
    pass api_key for the trigger endpoint if needed in future.
    """
    for attempt in range(MAX_RETRIES + 1):
        resp = requests.get(url, stream=True, timeout=60)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else BACKOFF_BASE ** (attempt + 1)
            if attempt < MAX_RETRIES:
                print(f"\n  Download rate limited. Waiting {wait:.0f}s...", file=sys.stderr)
                time.sleep(wait)
                continue
        resp.raise_for_status()
        break

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    with Image.open(dest) as img:
        return img.size


def main():
    parser = argparse.ArgumentParser(description="Source photorealistic images via Unsplash")
    parser.add_argument("--per-topic", type=int, default=DEFAULT_PER_TOPIC,
                        help=f"Images per topic (default {DEFAULT_PER_TOPIC} → {DEFAULT_PER_TOPIC * len(TOPICS)} total)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output directory (default {DEFAULT_OUT})")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between downloads (default 1.0)")
    args = parser.parse_args()

    api_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not api_key:
        print("ERROR: UNSPLASH_ACCESS_KEY not set. Add it to .env or export it.", file=sys.stderr)
        print("Get a free key at: https://unsplash.com/developers", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = []

    existing_ids = {entry["unsplash_id"] for entry in manifest}
    global_idx = len(manifest)
    skipped = 0

    total_planned = args.per_topic * len(TOPICS)
    print(f"Planned: {args.per_topic} images × {len(TOPICS)} topics = {total_planned} total")
    print(f"Rate limit floor: pause when ≤{RATE_LIMIT_REMAINING_FLOOR} requests remain\n")

    for topic in TOPICS:
        print(f"\nFetching '{topic}' ({args.per_topic} photos)...")
        try:
            photos = fetch_photos(topic, args.per_topic, api_key)
        except requests.HTTPError as e:
            print(f"  HTTP error fetching '{topic}': {e}", file=sys.stderr)
            continue

        photos = photos[: args.per_topic]

        with tqdm(total=len(photos), desc=f"  {topic}", leave=False) as pbar:
            for photo in photos:
                photo_id = photo["id"]

                if photo_id in existing_ids:
                    pbar.set_postfix(status="skip")
                    pbar.update(1)
                    skipped += 1
                    continue

                filename = f"unsplash_{global_idx:04d}.jpg"
                out_path = args.out / filename

                # Use "regular" size (~1080px) — full is too large for training
                download_url = photo["urls"]["regular"]
                photographer = photo["user"]["name"]
                unsplash_url = photo["links"]["html"]

                try:
                    width, height = download_image(download_url, out_path, api_key)
                except Exception as e:
                    print(f"\n  Error downloading {photo_id}: {e}", file=sys.stderr)
                    pbar.update(1)
                    continue

                manifest.append({
                    "filename": filename,
                    "source": "unsplash",
                    "unsplash_id": photo_id,
                    "topic": topic,
                    "photographer": photographer,
                    "unsplash_url": unsplash_url,
                    "download_url": download_url,
                    "width": width,
                    "height": height,
                })
                existing_ids.add(photo_id)
                global_idx += 1

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
