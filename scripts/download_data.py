"""Download the raw UCI dataset into data/raw (idempotent)."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

RAW = Path("data/raw")
URL = ("https://archive.ics.uci.edu/static/public/492/"
       "metro+interstate+traffic+volume.zip")
TARGET = RAW / "Metro_Interstate_Traffic_volume.csv.gz"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"already present: {TARGET} ({TARGET.stat().st_size} bytes)")
        return
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = [n for n in z.namelist() if n.endswith("Traffic_volume.csv.gz")]
        if not names:
            raise SystemExit(f"unexpected archive content: {z.namelist()}")
        TARGET.write_bytes(z.read(names[0]))
    print(f"wrote {TARGET} ({TARGET.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
