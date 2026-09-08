#!/usr/bin/env python3
"""Normaliserar flaskans storlek och position i färdiga noter-bilder.

Alla bilder har helt vit bakgrund och kapsylen överst i mitten, så kapsylens
överkant och bredd mäts och bilden skalas/förskjuts så att kapsyltoppen hamnar
på samma höjd och kapsylen får samma bredd i varje bild. Ingredienserna skalas
med, så skalan mot flaskan bevaras. Bilder där något skulle hamna utanför
kanten rapporteras.

Kör:  python3 scripts/normalize.py output/*_noter_gpt.png --out output/final
      python3 scripts/normalize.py output/*_noter.jpg --out output/final_gemini --cap-top 0.15 --cap-width 0.165
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

WHITE = 238  # gråvärde under detta räknas som "inte bakgrund"


def measure_cap(im: Image.Image) -> tuple[int, int, int]:
    """Returnerar (kapsyltopp y, kapsylbredd, kapsylens mittpunkt x) i pixlar."""
    g = np.asarray(im.convert("L"))
    h, w = g.shape
    band = slice(int(w * 0.35), int(w * 0.65))
    dark = g[:, band] < WHITE
    rows = np.where(dark.any(axis=1))[0]
    if len(rows) == 0:
        raise ValueError("hittar ingen kapsyl i mittbandet")
    top = int(rows[0])
    y = top + int(h * 0.04)  # en bit ned på kapsylen
    cols = np.where(g[y, band] < WHITE)[0]
    left, right = int(cols[0]) + band.start, int(cols[-1]) + band.start
    return top, right - left, (left + right) // 2


def content_bbox(im: Image.Image) -> tuple[int, int, int, int]:
    g = np.asarray(im.convert("L")) < WHITE
    ys, xs = np.where(g)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def white_balance_background(im: Image.Image) -> tuple[Image.Image, tuple[int, int, int]]:
    """Skalar varje färgkanal så att bakgrunden (medianen av kantpixlarna) blir exakt vit.

    Modellernas 'vita' är ofta svagt varm (t.ex. 252,251,251); utan detta syns en tonad
    fyrkant där bilden möter den vita duken efter skalning."""
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    h, w, _ = a.shape
    b = max(4, int(min(h, w) * 0.03))
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    out = np.clip(a * (255.0 / np.maximum(bg, 1.0)), 0, 255)
    return Image.fromarray(out.astype(np.uint8)), tuple(int(x) for x in bg)


def normalize(path: Path, out_dir: Path, cap_top_frac: float, cap_width_frac: float, fmt: str, quality: int) -> str:
    im, bg = white_balance_background(Image.open(path).convert("RGB"))
    w, h = im.size
    top, cap_w, cx = measure_cap(im)
    s = (cap_width_frac * w) / cap_w
    new = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    # placera så att kapsyltopp -> cap_top_frac*h och kapsylmitt -> w/2
    dx = round(w / 2 - cx * s)
    dy = round(cap_top_frac * h - top * s)
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(new, (dx, dy))
    x0, y0, x1, y1 = content_bbox(new)
    clipped = x0 + dx < 0 or y0 + dy < 0 or x1 + dx >= w or y1 + dy >= h
    stem = path.stem.replace("_gpt", "")
    ext = "jpg" if fmt == "jpeg" else "png"
    dest = out_dir / f"{stem}.{ext}"
    canvas.save(dest, "JPEG", quality=quality, subsampling=0) if fmt == "jpeg" else canvas.save(dest, "PNG")
    note = "  OBS: innehåll klipps vid kanten" if clipped else ""
    return f"{path.name}: bakgrund {bg} -> vit, kapsyltopp {top/h:.3f}, kapsylbredd {cap_w/w:.3f} -> skala {s:.3f}{note}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap-top", type=float, default=0.15, help="kapsyltoppens läge som andel av bildhöjden")
    ap.add_argument("--cap-width", type=float, default=None, help="kapsylbredd som andel av bildbredden (default: medianen av bilderna)")
    ap.add_argument("--format", choices=["jpeg", "png"], default="jpeg")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--measure", action="store_true", help="mät bara, skriv inget")
    ap.add_argument("--fit", action="store_true", help="välj största kapsylbredd som ryms i alla bilder utan att något klipps (med --margin)")
    ap.add_argument("--margin", type=float, default=0.02, help="minsta tomma marginal mot kanten som andel av bilden (för --fit)")
    args = ap.parse_args()

    measured = []
    for p in args.images:
        im = Image.open(p)
        top, cap_w, cx = measure_cap(im)
        measured.append((p, top / im.height, cap_w / im.width))
    if args.measure:
        for p, t, cw in measured:
            print(f"{p.name}: kapsyltopp {t:.3f}, kapsylbredd {cw:.3f}")
        return
    if args.fit:
        limits = []
        for p, _, cw in measured:
            im = Image.open(p); w, h = im.size
            top, cap_w, cx = measure_cap(im)
            x0, y0, x1, y1 = content_bbox(im)
            m = args.margin
            s_max = min(
                (w * (0.5 - m)) / max(cx - x0, 1),
                (w * (0.5 - m)) / max(x1 - cx, 1),
                (h * (1 - m - args.cap_top)) / max(y1 - top, 1),
            )
            limits.append((s_max * cap_w / w, p.name))
        cap_width, limiting = min(limits)
        print(f"--fit: största gemensamma kapsylbredd {cap_width:.4f} (begränsas av {limiting})")
    else:
        cap_width = args.cap_width or float(np.median([cw for _, _, cw in measured]))
    print(f"mål: kapsyltopp {args.cap_top:.3f}, kapsylbredd {cap_width:.3f}")
    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.images:
        print(normalize(p, args.out, args.cap_top, cap_width, args.format, args.quality))


if __name__ == "__main__":
    main()
