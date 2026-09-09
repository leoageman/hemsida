#!/usr/bin/env python3
"""Jämnar ut fläckiga gradientbakgrunder i editorial-bilder.

Bakgrundens färg mäts rad för rad i bildens ytterkanter, jämnas vertikalt och ersätter
bakgrundspixlarna med en ren vertikal gradient plus en mjuk syntetisk glöd bakom flaskan.
Flaskan (mittboxen), ingredienserna (nedre bandet, strikt tröskel) och hyllan rörs inte.

Kör:  python3 scripts/smooth_backdrop.py output/*_editorial_gpt.png --out output/final_editorial
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


def local_std(g: np.ndarray, r: int) -> np.ndarray:
    """Lokal standardavvikelse i ett (2r+1)^2-fönster via integralbilder (flyttal)."""
    def box_mean(x):
        pad = np.pad(x, r, mode="edge")
        c = np.cumsum(np.cumsum(pad, axis=0), axis=1)
        c = np.pad(c, ((1, 0), (1, 0)))
        w = 2 * r + 1
        return (c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w]) / (w * w)
    m1 = box_mean(g.astype(np.float64)); m2 = box_mean(g.astype(np.float64) ** 2)
    return np.sqrt(np.clip(m2 - m1 ** 2, 0, None)).astype(np.float32)


def detect_shelf(a: np.ndarray, fallback: float) -> float:
    """Hittar hyllans framkant: starkaste horisontella kanten i bildens ytterkolumner mellan 50 och 85 % av höjden."""
    H, W, _ = a.shape
    e = int(W * 0.08)
    lum = np.concatenate([a[:, :e].mean(axis=2), a[:, W - e:].mean(axis=2)], axis=1).mean(axis=1)
    lo, hi = int(H * 0.50), int(H * 0.85)
    d = np.abs(np.diff(lum[lo:hi]))
    if d.max() < 6:
        return fallback
    first = int(np.argmax(d > max(6.0, 0.4 * d.max())))   # första tydliga kanten uppifrån = hyllans överkant
    return (lo + first - int(H * 0.004)) / H


def smooth(path: Path, out_dir: Path, shelf: float, glow: float, quality: int) -> str:
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(np.float32); H, W, _ = a.shape
    edge = int(W * 0.05)
    y_max = int(H * detect_shelf(a, shelf))
    # 1) bakgrundsfärg per rad från kanterna, vertikalt utjämnad
    rows = np.concatenate([a[:y_max, :edge], a[:y_max, W - edge:]], axis=1)
    med = np.median(rows, axis=1)                       # (y_max, 3)
    bg = gaussian1d(med, sigma=H * 0.03)                # jämn vertikal gradient
    # 2) syntetisk, mjuk glöd bakom flaskan
    yy, xx = np.mgrid[0:y_max, 0:W]
    g = np.exp(-(((xx - W / 2) / (0.32 * W)) ** 2 + ((yy - 0.50 * H) / (0.30 * H)) ** 2))
    G = np.clip(bg[:, None, :] + glow * g[:, :, None], 0, 255)
    # 3) bakgrundsmask via flood fill från kanterna genom kantfria, släta pixlar.
    #    Motivets konturer (starka kanter) stoppar fyllningen, så färglikhet spelar ingen roll.
    def fblur(x, sigma):
        return gaussian1d(gaussian1d(x.astype(np.float32), sigma).T, sigma).T
    gray = a[:y_max].mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1])); gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    edge = np.maximum(gx, gy)
    tex = local_std(gray, 2)
    passable = ((edge < 6.0) & (tex < 5.0)).astype(np.uint8) * 255
    pas = Image.fromarray(passable).filter(ImageFilter.MinFilter(3))       # bredda barriärerna
    from PIL import ImageDraw
    seeds = [(x, 2) for x in range(2, W - 2, 64)] + [(2, y) for y in range(2, y_max - 2, 64)] + [(W - 3, y) for y in range(2, y_max - 2, 64)]
    for sx, sy in seeds:
        if pas.getpixel((sx, sy)) == 255:
            ImageDraw.floodfill(pas, (sx, sy), 128)
    fill = (np.asarray(pas) == 128).astype(np.float32)
    fill = np.asarray(Image.fromarray((fill * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))).astype(np.float32) / 255
    # lokal gammal bakgrund (inkl. glöd) från den fyllda ytan, för kantskiftning
    wsum = fblur(fill, 30) + 1e-3
    local_bg = np.stack([fblur(a[:y_max, :, c] * fill, 30) / wsum for c in range(3)], axis=2)
    m_soft = np.asarray(Image.fromarray((fill * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32) / 255
    shift_w = np.maximum(m_soft, fblur(fill, 3))
    shifted = a[:y_max] + (G - local_bg) * shift_w[:, :, None]
    out = a.copy()
    out[:y_max] = fill[:, :, None] * G + (1 - fill[:, :, None]) * shifted
    m = fill
    out += np.random.uniform(-0.5, 0.5, out.shape)      # dither mot banding
    res = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (path.stem.replace("_gpt", "") + ".jpg")
    res.save(dest, "JPEG", quality=quality, subsampling=0)
    return f"{path.name}: hylla vid {y_max / H:.2f}, ersatt {m.mean() * 100:.0f}% av ytan ovanför -> {dest.name}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shelf", type=float, default=0.62, help="andel av höjden ovanför hyllan som behandlas")
    ap.add_argument("--glow", type=float, default=12.0, help="styrka på den syntetiska glöden (0 = ingen)")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()
    for p in args.images:
        print(smooth(p, args.out, args.shelf, args.glow, args.quality))


if __name__ == "__main__":
    main()
