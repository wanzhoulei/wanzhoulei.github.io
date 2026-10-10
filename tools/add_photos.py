#!/usr/bin/env python3
"""Add a photo to the gallery page.

Resizes the photo (2000 px long edge, plus an 800 px thumbnail), converts it to sRGB, strips all
metadata including GPS, saves both copies under assets/gallery/, and adds an entry at the top of
its region in assets/gallery/photos.js.

Usage:
  python3 tools/add_photos.py PHOTO.jpg --region neotropics --species "Morpho helenor" \
      --place "Napo, Ecuador" --date "Nov 2025"
  python3 tools/add_photos.py PHOTO.jpg --region china --species "Aldania imitans" --cn "仿斑伞蛱蝶" \
      --note underside --place "Mingyong, Yunnan" --date "Jun 2026" --featured

--region decides the section (china, neotropics, or other). --featured shows the photo as a large tile.
"""
import argparse
import io
import json
import re
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
GALLERY = ROOT / "assets" / "gallery"
DATA = GALLERY / "photos.js"
FULL_EDGE, THUMB_EDGE = 2000, 800


def to_srgb(im):
    """Convert to RGB in the sRGB space, so colors survive dropping the embedded profile."""
    icc = im.info.get("icc_profile")
    if icc:
        try:
            from PIL import ImageCms
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            return ImageCms.profileToProfile(im.convert("RGB"), src, ImageCms.createProfile("sRGB"), outputMode="RGB")
        except Exception:
            pass
    return im.convert("RGB")


def save_jpeg(im, edge, path):
    im = im.copy()
    im.thumbnail((edge, edge), Image.LANCZOS)
    # No exif/icc/xmp arguments are passed, so the saved file carries no metadata.
    im.save(path, "JPEG", quality=88, optimize=True, progressive=True)
    return im.size


def process(photo, name):
    """Write assets/gallery/NAME.jpg and assets/gallery/thumbs/NAME.jpg; return the full size (w, h)."""
    im = Image.open(photo)
    im.draft("RGB", (FULL_EDGE, FULL_EDGE))      # fast JPEG decode at reduced scale, still >= 2000 px
    im = to_srgb(ImageOps.exif_transpose(im))
    (GALLERY / "thumbs").mkdir(parents=True, exist_ok=True)
    size = save_jpeg(im, FULL_EDGE, GALLERY / f"{name}.jpg")
    save_jpeg(im, THUMB_EDGE, GALLERY / "thumbs" / f"{name}.jpg")
    return size


def read_entries():
    text = DATA.read_text(encoding="utf-8")
    m = re.search(r"window\.GALLERY = (\[.*\]);", text, re.S)
    if not m:
        raise SystemExit(f"could not find 'window.GALLERY = [...];' in {DATA}")
    return text, m, json.loads(m.group(1))


def write_entries(text, m, entries):
    body = "[\n" + ",\n".join("  " + json.dumps(e, ensure_ascii=False) for e in entries) + "\n]"
    DATA.write_text(text[:m.start(1)] + body + text[m.end(1):], encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", type=Path)
    ap.add_argument("--region", required=True, choices=["china", "neotropics", "other"])
    ap.add_argument("--species", default="", help="scientific name, shown in italics")
    ap.add_argument("--cn", default="", help="Chinese name")
    ap.add_argument("--note", default="", help='e.g. "upperside", "underside", "male"')
    ap.add_argument("--place", default="", help='e.g. "Napo, Ecuador"')
    ap.add_argument("--date", default="", help='e.g. "Nov 2025"')
    ap.add_argument("--featured", action="store_true", help="show as a large tile")
    args = ap.parse_args()

    text, m, entries = read_entries()
    base = re.sub(r"[^a-z0-9]+", "-", f"{args.species or 'photo'} {args.note} {args.date}".lower()).strip("-")
    name, n = base, 2
    taken = {e["file"] for e in entries}
    while f"{name}.jpg" in taken or (GALLERY / f"{name}.jpg").exists():
        name, n = f"{base}-{n}", n + 1
    w, h = process(args.photo, name)

    entry = {"file": f"{name}.jpg", "w": w, "h": h, "region": args.region}
    for key in ("species", "cn", "note", "place", "date"):
        if getattr(args, key):
            entry[key] = getattr(args, key)
    if args.featured:
        entry["featured"] = True
    first = next((i for i, e in enumerate(entries) if e.get("region") == args.region), len(entries))
    entries.insert(first, entry)
    write_entries(text, m, entries)
    print(f"added assets/gallery/{name}.jpg ({w}x{h})")


if __name__ == "__main__":
    main()
