#!/usr/bin/env python3
"""Jämnar ut ojämn belysning ("grumlighet") på flaskans etikett.

Hittar etikettens rektangel i mittbandet, uppskattar den ojämna belysningen från
själva pappret (ljusa, omättade, släta pixlar) med en normaliserad oskärpa och
delar bort den (flat field). Text och illustration påverkas bara av samma lokala
korrektion som pappret runt dem, så deras toner behålls.

Kör:  python3 scripts/flatten_label.py output/final_editorial/*.jpg --inplace
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def gaussian1d(a: np.ndarray, sigma: float) -> np.ndarray:
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    pad = np.pad(a, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(a.shape[1])], axis=1)


def fblur(x: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian1d(gaussian1d(x.astype(np.float32), sigma).T, sigma).T


def local_std(g: np.ndarray, r: int) -> np.ndarray:
    def box_mean(x):
        pad = np.pad(x, r, mode="edge")
        c = np.cumsum(np.cumsum(pad, axis=0), axis=1); c = np.pad(c, ((1, 0), (1, 0))); w = 2 * r + 1
        return (c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w]) / (w * w)
    m1 = box_mean(g.astype(np.float64)); m2 = box_mean(g.astype(np.float64) ** 2)
    return np.sqrt(np.clip(m2 - m1 ** 2, 0, None)).astype(np.float32)


GEO = None


def find_label(a: np.ndarray):
    """Etikettens box från flaskans geometri: kapsyltopp och kapsylbredd mäts i bilden,
    etikettens läge relativt dem är fast (data/label_geometry.json, mätt i referensfotot)."""
    import json
    global GEO
    if GEO is None:
        GEO = json.loads((Path(__file__).resolve().parents[1] / "data" / "label_geometry.json").read_text())
    H, W, _ = a.shape
    e = int(W * 0.05)
    # bakgrund per rad från kanterna; kapsylen = första raden i mittbandet som avviker tydligt
    bg = np.median(np.concatenate([a[:, :e], a[:, W - e:]], axis=1), axis=1)
    xb0, xb1 = int(W * 0.45), int(W * 0.55)
    d = np.linalg.norm(a[:, xb0:xb1] - bg[:, None, :], axis=2)
    rows = np.where((d > 40).mean(axis=1) > 0.5)[0]
    rows = rows[(rows > H * 0.15) & (rows < H * 0.60)]
    if len(rows) == 0:
        return None
    cap_top = int(rows[0])
    y = cap_top + int(H * 0.03)
    wb0, wb1 = int(W * 0.30), int(W * 0.70)
    dr = np.linalg.norm(a[y, wb0:wb1] - bg[y][None, :], axis=1) > 40
    xs = np.where(dr)[0]
    if len(xs) == 0:
        return None
    capw = int(xs[-1] - xs[0]); cx = wb0 + (xs[-1] + xs[0]) / 2
    span = capw * GEO["span_per_capw"]
    x0 = int(cx + GEO["x0_capw"] * capw); x1 = int(cx + GEO["x1_capw"] * capw)
    y0 = int(cap_top + GEO["y0_frac"] * span); y1 = int(cap_top + GEO["y1_frac"] * span)
    return x0, y0, x1, y1


def flatten(path: Path, out_path: Path, quality: int, sigma_frac: float, strength: float) -> str:
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(np.float32); H, W, _ = a.shape
    box = find_label(a)
    if box is None:
        im.save(out_path, "JPEG", quality=quality, subsampling=0)
        return f"{path.name}: ingen etikett hittad, oförändrad"
    x0, y0, x1, y1 = box
    ix, iy = int((x1 - x0) * 0.03), int((y1 - y0) * 0.02)
    x0, y0, x1, y1 = x0 + ix, y0 + iy, x1 - ix, y1 - iy
    lab = a[y0:y1, x0:x1]
    gray = lab.mean(axis=2)
    paper = (lab.min(axis=2) > 150) & ((lab.max(axis=2) - lab.min(axis=2)) < 45) & (local_std(gray, 2) < 6)
    if paper.mean() < 0.2:
        im.save(out_path, "JPEG", quality=quality, subsampling=0)
        return f"{path.name}: för lite papper i etiketten, oförändrad"
    sigma = max(6.0, (y1 - y0) * sigma_frac)
    pm = paper.astype(np.float32)
    wsum = fblur(pm, sigma) + 1e-3
    L = np.stack([fblur(lab[:, :, c] * pm, sigma) / wsum for c in range(3)], axis=2)      # lokal pappersljushet
    target = np.array([min(max(np.percentile(lab[:, :, c][paper], 97), 236.0), 248.0) for c in range(3)], dtype=np.float32)
    gain = np.clip(target[None, None, :] / np.maximum(L, 1.0), 0.8, 1.4)
    gain = 1.0 + strength * (gain - 1.0)
    corrected = np.clip(lab * gain, 0, 255)
    # mjuk kant mot resten av flaskan
    mask = np.ones((y1 - y0, x1 - x0), dtype=np.float32)
    f = 4
    mask[:f, :] *= np.linspace(0, 1, f)[:, None]; mask[-f:, :] *= np.linspace(1, 0, f)[:, None]
    mask[:, :f] *= np.linspace(0, 1, f)[None, :]; mask[:, -f:] *= np.linspace(1, 0, f)[None, :]
    out = a.copy()
    out[y0:y1, x0:x1] = lab * (1 - mask[:, :, None]) + corrected * mask[:, :, None]
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(out_path, "JPEG", quality=quality, subsampling=0)
    spread = float(np.percentile(gray[paper], 95) - np.percentile(gray[paper], 5))
    return f"{path.name}: etikett {x0}-{x1} x {y0}-{y1}, papper {paper.mean()*100:.0f}%, ljusspridning före {spread:.0f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="utmapp (default: skriv över med --inplace)")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--sigma", type=float, default=0.07, help="oskärpa som andel av etiketthöjden")
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args()
    if not args.inplace and args.out is None:
        raise SystemExit("ange --out eller --inplace")
    if args.out: args.out.mkdir(parents=True, exist_ok=True)
    for p in args.images:
        dest = p if args.inplace else args.out / p.name
        print(flatten(p, dest, args.quality, args.sigma, args.strength))


if __name__ == "__main__":
    main()
