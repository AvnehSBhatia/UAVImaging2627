"""Optional: download free nadir-ish aerial stills into backgrounds/web/.

Does NOT replace gen-assets runway/grass/forest — it adds a 'web' bucket
the hyper generator can sample. Sources: Unsplash Source (no key) + a few
fixed Pexels/Unsplash CDN URLs.

  cd datasetgen-2026
  ../.venv/bin/python -m gen_hyper.fetch_web_backgrounds --n 40
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

# curated aerial / top-down-ish free URLs (Unsplash + Pexels CDN)
SEED_URLS = [
    "https://images.pexels.com/photos/13633587/pexels-photo-13633587.jpeg?auto=compress&cs=tinysrgb&w=1920",
    "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=1920&q=80",
    "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=1920&q=80",
    "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=1920&q=80",
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1920&q=80",
    "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=1920&q=80",
    "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=1920&q=80",
    "https://images.unsplash.com/photo-1511497584788-876760111969?w=1920&q=80",
    "https://images.pexels.com/photos/1598073/pexels-photo-1598073.jpeg?auto=compress&cs=tinysrgb&w=1920",
    "https://images.pexels.com/photos/440731/pexels-photo-440731.jpeg?auto=compress&cs=tinysrgb&w=1920",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="../datasets/gen-assets/backgrounds/web")
    ap.add_argument("--n", type=int, default=40,
                    help="target count (cycles SEED_URLS with cache-bust)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    got = 0
    for i in range(args.n):
        url = SEED_URLS[i % len(SEED_URLS)]
        # cache-bust so repeats still download distinct CDN variants when possible
        sep = "&" if "?" in url else "?"
        full = f"{url}{sep}cb={i}"
        name = f"web_{i:04d}_{hashlib.md5(full.encode()).hexdigest()[:8]}.jpg"
        dest = out / name
        if dest.exists() and dest.stat().st_size > 10_000:
            got += 1
            continue
        try:
            req = urllib.request.Request(full, headers={"User-Agent": "UAVImaging2627/gen_hyper"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            if len(data) < 5_000:
                print(f"skip tiny {name}")
                continue
            dest.write_bytes(data)
            got += 1
            print(f"  {name} ({len(data)//1024} KB)")
        except Exception as e:
            print(f"  fail {i}: {e}")
    # meta.jsonl with nominal GSD so AssetLibrary can also see them if copied
    # (gen_hyper attach_web_backgrounds does not require meta)
    print(f"web backgrounds in {out}: {got} files")


if __name__ == "__main__":
    main()
