#!/usr/bin/env python3
"""Skapar OBC "noter"-produktbilder med Nano Banana Pro (Gemini image model).

För varje parfym i data/selection.json:
  1. hämtar flaskbilden från Shopify (cache: reference/bottles/<nr>_<namn>.png)
  2. bygger en prompt av de fyra valda noterna (sparas i prompts/<nr>_<namn>.txt)
  3. skickar flaskbilden + prompten till Gemini och sparar resultatet i
     output/<nr>_<namn>_noter.<png|jpg>

Kräver:  pip install google-genai requests
         export GEMINI_API_KEY=...   (från https://aistudio.google.com/apikey)

Exempel:
  python3 scripts/generate.py --dry-run                # skriv bara prompts
  python3 scripts/generate.py --skus 1-10              # generera 1.0–10.0
  python3 scripts/generate.py --skus 2,5 --force       # gör om 2.0 och 5.0
  python3 scripts/generate.py --list-models            # visa tillgängliga image-modeller
  python3 scripts/generate.py --fetch-bottles --skus 1-200   # ladda bara ner alla flaskbilder från Shopify
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ROOT / "data" / "products.json"
SELECTION = ROOT / "data" / "selection.json"
BOTTLES = ROOT / "reference" / "bottles"
PROMPTS = ROOT / "prompts"
OUTPUT = ROOT / "output"

DEFAULT_MODEL = "gemini-3-pro-image-preview"  # "Nano Banana Pro"

PROMPT_TEMPLATE = """Edit this image. It is a product photo of a 50 ml perfume bottle from One Bold Chemist. Keep the bottle EXACTLY as it is: the same glass bottle shape, the same brushed silver cap, the same liquid colour, and the same label with the exact text "{label}", "extrait de parfum" and "one bold chemist" plus the same small halftone illustration inside the label's frame. Do not redraw, retouch or alter a single letter or the artwork on the label. Re-frame the scene so the unchanged bottle stands centred on a pure white surface, about 60% of the image height with its base a little below the middle of the frame, seen from a slightly elevated camera angle, leaving open white space around it.

On the white surface around the bottle, place the fragrance's raw ingredients as four separate small groups, spread out in a loose ring mostly in front of and beside the bottle, with clear white space between the groups and at most one or two pieces peeking out behind the bottle. Nothing is stacked, heaped or piled up, and nothing covers the label:
{ingredients}

Every ingredient is a real, physical, tactile object at true scale relative to a 50 ml bottle, lying flat on the surface with realistic textures and soft natural shadows. Airy, minimal editorial composition, like a clean ingredient flat lay shot from a slightly elevated front angle that matches the bottle's perspective. Pure seamless white background with soft, even studio lighting. No text, no captions, no extra props, no hands. Square 1:1 image, high-end product photography."""


def slugify(name: str) -> str:
    name = name.replace("ø", "o").replace("Ø", "O").replace("æ", "ae").replace("Æ", "AE").replace("ß", "ss")
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def parse_skus(spec: str | None, available: list[int]) -> list[int]:
    if not spec:
        return available
    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            wanted.update(range(int(lo), int(hi) + 1))
        elif part:
            wanted.add(int(part))
    return [n for n in available if n in wanted]


def build_prompt(product: dict, ingredients: list[dict]) -> str:
    label = f"{product['name'].upper()} {product['number']}.0"
    lines = "\n".join(f"{i}. {ing['visual']} ({ing['note']})" for i, ing in enumerate(ingredients, 1))
    return PROMPT_TEMPLATE.format(label=label, ingredients=lines)


def ensure_bottle(product: dict) -> Path:
    stem = f"{product['number']:02d}_{slugify(product['name'])}"
    for ext in ("png", "jpg", "jpeg"):
        p = BOTTLES / f"{stem}.{ext}"
        if p.exists():
            return p
    url = product["bottle_image"]
    if not url:
        raise SystemExit(f"{product['title']}: ingen flaskbild i Shopify-manifestet")
    import requests  # noqa: PLC0415

    ext = "jpg" if ".jpg" in url.lower() or ".jpeg" in url.lower() else "png"
    dest = BOTTLES / f"{stem}.{ext}"
    BOTTLES.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"  hämtade flaskbild -> {dest.relative_to(ROOT)}")
    return dest


def get_client():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Sätt GEMINI_API_KEY (eller GOOGLE_API_KEY) i miljön. Nyckel: https://aistudio.google.com/apikey")
    from google import genai  # noqa: PLC0415

    return genai.Client(api_key=key)


def generate_image(client, model: str, size: str, bottle: Path, prompt: str, retries: int = 3) -> tuple[bytes, str]:
    from google.genai import types  # noqa: PLC0415

    mime = "image/jpeg" if bottle.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(aspect_ratio="1:1", image_size=size),
    )
    contents = [types.Part.from_bytes(data=bottle.read_bytes(), mime_type=mime), prompt]
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            texts = []
            for cand in resp.candidates or []:
                for part in cand.content.parts or []:
                    if getattr(part, "inline_data", None) and part.inline_data.data:
                        return part.inline_data.data, part.inline_data.mime_type or "image/png"
                    if getattr(part, "text", None):
                        texts.append(part.text)
            raise RuntimeError("Modellen returnerade ingen bild. Svar: " + (" ".join(texts)[:500] or str(getattr(resp, "prompt_feedback", ""))))
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                wait = 5 * attempt
                print(f"  försök {attempt} misslyckades ({e}); väntar {wait}s ...")
                time.sleep(wait)
    raise RuntimeError(f"gav upp efter {retries} försök: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skus", help="t.ex. '1-10' eller '1,4,9' (default: alla i selection.json)")
    ap.add_argument("--model", default=os.environ.get("NANO_BANANA_MODEL", DEFAULT_MODEL))
    ap.add_argument("--size", default="2K", choices=["1K", "2K", "4K"], help="utbildens storlek (default 2K = 2048 px, som befintliga Noter-bilder)")
    ap.add_argument("--dry-run", action="store_true", help="skriv bara prompts, anropa inte API:et")
    ap.add_argument("--force", action="store_true", help="generera om även om output-filen redan finns")
    ap.add_argument("--sleep", type=float, default=2.0, help="sekunder mellan API-anrop")
    ap.add_argument("--list-models", action="store_true", help="lista modeller som kan generera bilder och avsluta")
    ap.add_argument("--fetch-bottles", action="store_true", help="ladda bara ner flaskbilder från Shopify till reference/bottles/ (ingen generering)")
    args = ap.parse_args()

    if args.list_models:
        client = get_client()
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if "image" in m.name.lower() or any("image" in a.lower() for a in actions):
                print(m.name)
        return

    products = {p["number"]: p for p in json.loads(PRODUCTS.read_text(encoding="utf-8"))}
    selection = {int(k): v for k, v in json.loads(SELECTION.read_text(encoding="utf-8")).items() if k.isdigit()}
    if args.fetch_bottles:
        for n in parse_skus(args.skus, sorted(products)):
            print(products[n]["title"])
            ensure_bottle(products[n])
        return
    numbers = parse_skus(args.skus, sorted(selection))
    if not numbers:
        raise SystemExit("Inga parfymer matchade --skus (finns urval i data/selection.json?)")

    PROMPTS.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    client = None if args.dry_run else get_client()
    failures: list[str] = []

    for n in numbers:
        product = products.get(n)
        if not product:
            print(f"{n}: finns inte i data/products.json, hoppar över")
            continue
        ingredients = selection[n]["ingredients"]
        if len(ingredients) != 4:
            print(f"  OBS: {product['title']} har {len(ingredients)} ingredienser i urvalet (förväntat 4)")
        stem = f"{product['number']:02d}_{slugify(product['name'])}"
        prompt = build_prompt(product, ingredients)
        (PROMPTS / f"{stem}.txt").write_text(prompt + "\n", encoding="utf-8")
        print(f"{product['title']}: {', '.join(i['note'] for i in ingredients)}")

        if args.dry_run:
            continue
        existing = [p for p in OUTPUT.glob(f"{stem}_noter.*")]
        if existing and not args.force:
            print(f"  finns redan: {existing[0].name} (använd --force för att göra om)")
            continue
        try:
            bottle = ensure_bottle(product)
            data, mime = generate_image(client, args.model, args.size, bottle, prompt)
            ext = "jpg" if "jpeg" in mime else "png"
            dest = OUTPUT / f"{stem}_noter.{ext}"
            dest.write_bytes(data)
            print(f"  sparade {dest.relative_to(ROOT)} ({len(data) // 1024} kB)")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{product['title']}: {e}")
            print(f"  FEL: {e}")
        time.sleep(args.sleep)

    if args.dry_run:
        print(f"\nPrompts skrivna till {PROMPTS.relative_to(ROOT)}/ (dry-run, inget genererat)")
    elif failures:
        print("\nMisslyckades:\n  " + "\n  ".join(failures))
        sys.exit(1)
    else:
        print(f"\nKlart. Bilder i {OUTPUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
