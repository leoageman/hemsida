#!/usr/bin/env python3
"""Skapar OBC "noter"-produktbilder med Nano Banana Pro (Gemini image model).

För varje parfym i data/selection.json:
  1. hämtar flaskbilden från Shopify (cache: reference/bottles/<nr>_<namn>.png)
  2. bygger en prompt av de fyra valda noterna (sparas i prompts/<nr>_<namn>.txt)
  3. skickar flaskbilden + prompten till Gemini och sparar resultatet i
     output/<nr>_<namn>_noter.<png|jpg>

Kräver:  pip install google-genai requests
         export GEMINI_API_KEY=...   (från https://aistudio.google.com/apikey)
   eller  export OPENAI_API_KEY=...  och --provider openai (GPT Image, https://platform.openai.com/api-keys)

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
DEFAULT_OPENAI_MODEL = "gpt-image-1"
OPENAI_API = "https://api.openai.com/v1"

PROMPT_TEMPLATE = """Edit this image. Image 1 is a product photo of a 50 ml perfume bottle from One Bold Chemist.{style_ref}

BOTTLE: keep the bottle exactly as in image 1: the same clear glass bottle and liquid colour, and the same cap: a smooth, light, matte silver aluminium cap, pale and evenly lit with only a soft gentle gradient and a slightly rounded top edge, no dark bands, no heavy brushed streaks, no mirror reflections, exactly the colour and finish seen in image 1, and the same label with the exact text "{label}", "extrait de parfum" and "one bold chemist" plus the same small halftone illustration. Do not redraw, rotate, tilt or alter the label in any way. The label is opaque printed paper wrapped around the glass: it is never transparent, and nothing beside or behind the bottle ever shows through it.

CAMERA: a front view with the camera raised clearly above the bottle, looking down at about 20 degrees, so the full top of the cap is visible as an ellipse and the white surface stretches out in front of and around the bottle. The label faces the camera squarely, the bottle is not rotated, 100 mm macro-capable lens with no wide-angle distortion, tack sharp on the bottle and on every ingredient. This exact camera height, angle and distance is identical for every image in the series: the bottle is always the same size and in the same position, horizontally centred, the top of the cap about 15% below the top edge and the base of the bottle about 72% below the top edge, so the bottle spans about 57% of the image height. Never move the camera closer or further away.

SCENE: one single real studio photograph, shot on a medium-format camera at f/11. The bottle and all ingredients stand together on the same pure white seamless surface that continues into a pure white background. The background is evenly lit, clean white everywhere, with no gradient, vignette, grey falloff or darkening at the top; the only tones on the white are the objects' soft shadows and reflections. Lit by one large, soft light from the upper left and gently filled from the right, so every object shares the same soft directional light, has a soft contact shadow under it falling to the lower right, and a faint reflection on the paper. The ingredients closest to the bottle are faintly reflected and refracted in its glass, and the bottle casts its own soft shadow across the ingredients on its right. Same colour temperature, same sharpness and fine grain across the whole frame, slight natural depth of field towards the back. Nothing may look cut out, pasted in or floating; every object sits firmly on the surface. Nothing but the white sweep and the objects standing on it is visible anywhere in the frame; the picture has no visible edges, panels, walls or equipment of any kind.

INGREDIENTS: arrange the fragrance's raw ingredients around the base of the bottle in four separate small groups, in front of and beside the bottle only. The space directly behind the bottle stays completely empty, so nothing is seen through the glass and the label is read against plain background. Clear space between the groups, nothing piled up and nothing covering the label:
{ingredients}
SCALE: the bottle is 10 cm tall; every ingredient keeps its true real-world size relative to it and to each other. Large produce stays large: a grapefruit half is about as wide as the bottle is tall, a pear or a bergamot about two thirds of the bottle's height, a chestnut about a quarter, and cardamom pods, cloves, peppercorns and berries are only about one centimetre. Objects closer to the camera appear slightly larger, never smaller. Keep groups low and loose, a natural handful per group, and place large items beside the bottle rather than in front of the label.

REALISM: every ingredient is photographed with macro-level detail and looks like a real, slightly imperfect raw material picked up at a spice market or in a garden: visible surface texture, fibres, pores, ridges, tiny cracks, dust, moisture and natural colour variation, with small specular highlights where surfaces are glossy and soft translucency where light passes through. Nothing is idealised, symmetrical, waxy, plastic or illustrated, and nothing looks like a clean stock cut-out.

FRAMING: the whole arrangement, bottle and ingredients together, sits inside the central 75% of the frame; the outer 12% on every side is empty white background. No object touches or crosses the picture edge: make an ingredient smaller or move it inward rather than letting it reach the edge. No text, captions, extra props or hands. Square 1:1, high-end product photography."""

STYLE_REF_SENTENCE = " Image 2 is a finished example from the same series showing a DIFFERENT fragrance: match only its camera height, angle and distance, so the bottle has the same size and position in the frame, with the same lighting and shadow direction, as if shot in the same session without touching the camera. Never copy the label, label text, illustration or liquid colour from image 2; those come from image 1 only. Background must be pure white and every ingredient must stay well inside the frame, as described below."


EDITORIAL_TEMPLATE = """Edit this image. Image 1 is a product photo of a 50 ml perfume bottle from One Bold Chemist.{style_ref}

BOTTLE: keep the bottle exactly as in image 1: the same clear glass bottle and liquid colour, and the same cap: a smooth, light silver aluminium cap with a fine, subtle brushed finish, evenly lit with a soft gradient and a slightly rounded top edge, no dark bands. The same label with the exact text "{label}", "extrait de parfum" and "one bold chemist" and the same small halftone illustration. Do not redraw, rotate, tilt or alter the label; it is opaque printed paper and nothing shows through it.

SCENE: a premium fragrance campaign still life. The bottle stands upright and centred on a thick glass shelf. The shelf's top surface is polished and shows a faint, soft reflection of the bottle and ingredients; its front edge is visible as a horizontal band of pale green-tinted glass running across the whole width at about 80% of the image height, and below the shelf the backdrop simply continues. Behind everything is a seamless studio backdrop with a smooth vertical two-tone gradient: {backdrop}. The gradient runs top to bottom, is perfectly smooth with a soft glow behind the bottle, no texture, no horizon other than the shelf, no visible equipment.

CAMERA: straight-on front view with the camera just above shelf level, so the shelf's top surface is barely visible as a thin sliver and the label faces the camera squarely; normal 85 mm lens, no tilt, bottle not rotated. The bottle is large in the frame: the top of its cap about 22% below the top edge and its base on the shelf at about 78%, so the bottle spans roughly 56% of the image height, horizontally centred. This camera height, angle and distance is identical for every image in the series.

ARRANGEMENT: the fragrance's raw ingredients are gathered closely around the base of the bottle on both sides, touching or leaning against it, in two rich, tight clusters that stay lower than the label's top edge and never cover the label text or illustration. Nothing is placed behind the bottle where it would show through the glass, nothing floats, and every piece rests on the shelf and is faintly reflected in it. Both clusters stay well inside the frame with clear margin to the left and right edges:
{ingredients}
Everything is at true real-world scale relative to the 10 cm bottle: large produce stays large, seeds and berries stay about one centimetre.

LIGHT: one large soft key light from above and slightly left, warm and flattering, gentle fill from the right, soft shadows on the shelf, a subtle rim of backdrop colour in the glass and liquid. Photorealistic and hyper-detailed: visible fibres, cracks, pores, moisture and translucency on every ingredient, fine grain, tack sharp. No text, no captions, no extra props, no hands. Square 1:1, high-end campaign photography."""

EDITORIAL_STYLE_REF_SENTENCE = " Image 2 is a finished example from the same series showing a DIFFERENT fragrance: match only its camera height and distance, bottle size and position, shelf position, lighting and reflection, so this image looks shot in the same session. Never copy the label, the label text, the illustration or the liquid colour from image 2; the bottle, label text and illustration must come from image 1 only, and the backdrop colours and ingredients are as described below."

FLOW_TEMPLATE = """Product still life of the exact perfume bottle from the reference image: the same 50 ml glass bottle, brushed silver cap, liquid colour and label with the text "{label}", "extrait de parfum" and "one bold chemist" and the same small halftone illustration, reproduced exactly with no changes to the label. The bottle stands centred on a pure white surface, about 60% of the frame height, seen from a slightly elevated angle.

Around it, spread out as four separate small groups with clear white space between them, mostly in front of and beside the bottle, nothing piled up and nothing hiding the label:
{ingredients}

Everything at true scale relative to a 50 ml bottle, realistic textures, soft natural shadows on the white surface. Airy, minimal editorial flat lay, pure seamless white background, soft even studio light. No text or captions, no extra props, no hands. Square 1:1, high-end product photography."""

FLOW_PROMPTS = PROMPTS / "flow"


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


def build_prompt(product: dict, ingredients: list[dict], style_ref: bool = False) -> str:
    label = f"{product['name'].upper()} {product['number']}.0"
    lines = "\n".join(f"{i}. {ing['visual']} ({ing['note']})" for i, ing in enumerate(ingredients, 1))
    return PROMPT_TEMPLATE.format(label=label, ingredients=lines, style_ref=STYLE_REF_SENTENCE if style_ref else "")


def build_editorial_prompt(product: dict, sel: dict, style_ref: bool = False) -> str:
    label = f"{product['name'].upper()} {product['number']}.0"
    lines = "\n".join(f"{i}. {ing['visual']} ({ing['note']})" for i, ing in enumerate(sel["ingredients"], 1))
    return EDITORIAL_TEMPLATE.format(
        label=label, ingredients=lines, backdrop=sel["backdrop"],
        style_ref=EDITORIAL_STYLE_REF_SENTENCE if style_ref else "",
    )


def build_flow_prompt(product: dict, ingredients: list[dict]) -> str:
    label = f"{product['name'].upper()} {product['number']}.0"
    lines = "\n".join(f"- {ing['visual']} ({ing['note']})" for ing in ingredients)
    return FLOW_TEMPLATE.format(label=label, ingredients=lines)


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


def openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Sätt OPENAI_API_KEY i miljön. Nyckel: https://platform.openai.com/api-keys")
    return key


def openai_list_models() -> None:
    import requests  # noqa: PLC0415

    r = requests.get(f"{OPENAI_API}/models", headers={"Authorization": f"Bearer {openai_key()}"}, timeout=60)
    r.raise_for_status()
    for m in sorted(x["id"] for x in r.json()["data"]):
        if "image" in m or "dall" in m:
            print(m)


def generate_image_openai(model: str, size: str, bottle: Path, prompt: str, retries: int = 3, style_ref: Path | None = None) -> tuple[bytes, str]:
    """GPT Image via /images/edits: flaskbilden (och ev. stilreferens) skickas som bild 1 (och 2)."""
    import base64  # noqa: PLC0415

    import requests  # noqa: PLC0415

    def mime_of(p: Path) -> str:
        return "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"

    px = {"auto": "1024x1024", "1K": "1024x1024", "2K": "2048x2048", "4K": "2048x2048"}[size]
    files = [("image[]", (bottle.name, bottle.read_bytes(), mime_of(bottle)))]
    if style_ref:
        files.append(("image[]", (style_ref.name, style_ref.read_bytes(), mime_of(style_ref))))
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                f"{OPENAI_API}/images/edits",
                headers={"Authorization": f"Bearer {openai_key()}"},
                files=files,
                data={"model": model, "prompt": prompt, "n": "1", "size": px, "quality": "high", "output_format": "png",
                      **({"input_fidelity": "high"} if model.startswith("gpt-image-1") else {})},
                timeout=600,
            )
            if r.status_code == 400 and "size" in r.text and px != "1024x1024":
                px = "1536x1536" if px == "2048x2048" and "1536x1536" in r.text else "1024x1024"
                print(f"  storleken stöds inte, provar {px}")
                continue
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
            item = r.json()["data"][0]
            if item.get("b64_json"):
                return base64.b64decode(item["b64_json"]), "image/png"
            if item.get("url"):
                return requests.get(item["url"], timeout=120).content, "image/png"
            raise RuntimeError("Inget bildinnehåll i svaret")
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                wait = 5 * attempt
                print(f"  försök {attempt} misslyckades ({e}); väntar {wait}s ...")
                time.sleep(wait)
    raise RuntimeError(f"gav upp efter {retries} försök: {last_err}")


def _image_part(path: Path):
    from google.genai import types  # noqa: PLC0415

    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)


def generate_image(client, model: str, size: str, bottle: Path, prompt: str, retries: int = 3, style_ref: Path | None = None) -> tuple[bytes, str]:
    from google.genai import types  # noqa: PLC0415

    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(aspect_ratio="1:1", **({} if size == "auto" else {"image_size": size})),
    )
    contents = [_image_part(bottle)] + ([_image_part(style_ref)] if style_ref else []) + [prompt]
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
    ap.add_argument("--provider", default="gemini", choices=["gemini", "openai"], help="gemini = Nano Banana Pro (default), openai = GPT Image")
    ap.add_argument("--model", default=None, help=f"modell-id (default {DEFAULT_MODEL} / {DEFAULT_OPENAI_MODEL})")
    ap.add_argument("--size", default="2K", choices=["auto", "1K", "2K", "4K"], help="utbildens storlek (default 2K = 2048 px, som befintliga Noter-bilder; 'auto' = skicka ingen storlek, krävs för flash-image-modellerna)")
    ap.add_argument("--dry-run", action="store_true", help="skriv bara prompts, anropa inte API:et")
    ap.add_argument("--force", action="store_true", help="generera om även om output-filen redan finns")
    ap.add_argument("--sleep", type=float, default=2.0, help="sekunder mellan API-anrop")
    ap.add_argument("--list-models", action="store_true", help="lista modeller som kan generera bilder och avsluta")
    ap.add_argument("--style", default="studio", choices=["studio", "editorial"], help="studio = vit studiobild (default), editorial = gradientbakgrund + spegelhylla")
    ap.add_argument("--style-ref", type=Path, default=None, help="färdig bild som stilreferens (skickas som bild 2 till Gemini), t.ex. output/04_bachelder_noter.jpg")
    ap.add_argument("--fetch-bottles", action="store_true", help="ladda bara ner flaskbilder från Shopify till reference/bottles/ (ingen generering)")
    args = ap.parse_args()

    if args.model is None:
        args.model = DEFAULT_OPENAI_MODEL if args.provider == "openai" else os.environ.get("NANO_BANANA_MODEL", DEFAULT_MODEL)
    if args.list_models and args.provider == "openai":
        openai_list_models()
        return
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
    client = None if (args.dry_run or args.provider == "openai") else get_client()
    if args.provider == "openai" and not args.dry_run:
        openai_key()
    suffix = ("_editorial" if args.style == "editorial" else "_noter") + ("_gpt" if args.provider == "openai" else "")
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
        if args.style == "editorial":
            prompt = build_editorial_prompt(product, selection[n], style_ref=bool(args.style_ref))
            (PROMPTS / "editorial").mkdir(exist_ok=True)
            (PROMPTS / "editorial" / f"{stem}.txt").write_text(prompt + "\n", encoding="utf-8")
        else:
            prompt = build_prompt(product, ingredients, style_ref=bool(args.style_ref))
            (PROMPTS / f"{stem}.txt").write_text(prompt + "\n", encoding="utf-8")
        FLOW_PROMPTS.mkdir(exist_ok=True)
        (FLOW_PROMPTS / f"{stem}.txt").write_text(build_flow_prompt(product, ingredients) + "\n", encoding="utf-8")
        print(f"{product['title']}: {', '.join(i['note'] for i in ingredients)}")

        if args.dry_run:
            continue
        existing = [p for p in OUTPUT.glob(f"{stem}{suffix}.*")]
        if existing and not args.force:
            print(f"  finns redan: {existing[0].name} (använd --force för att göra om)")
            continue
        try:
            bottle = ensure_bottle(product)
            if args.provider == "openai":
                data, mime = generate_image_openai(args.model, args.size, bottle, prompt, style_ref=args.style_ref)
            else:
                data, mime = generate_image(client, args.model, args.size, bottle, prompt, style_ref=args.style_ref)
            ext = "jpg" if "jpeg" in mime else "png"
            dest = OUTPUT / f"{stem}{suffix}.{ext}"
            dest.write_bytes(data)
            print(f"  sparade {dest.relative_to(ROOT)} ({len(data) // 1024} kB)")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{product['title']}: {e}")
            print(f"  FEL: {e}")
        time.sleep(args.sleep)

    FLOW_PROMPTS.mkdir(exist_ok=True)
    (FLOW_PROMPTS / "TEMPLATE.txt").write_text(
        FLOW_TEMPLATE.format(label="NAMN N.0", ingredients="- <ingrediens 1, hur den ser ut> (<not>)\n- <ingrediens 2> (<not>)\n- <ingrediens 3> (<not>)\n- <ingrediens 4> (<not>)") + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print(f"\nPrompts skrivna till {PROMPTS.relative_to(ROOT)}/ (API) och {FLOW_PROMPTS.relative_to(ROOT)}/ (Google Flow) (dry-run, inget genererat)")
    elif failures:
        print("\nMisslyckades:\n  " + "\n  ".join(failures))
        sys.exit(1)
    else:
        print(f"\nKlart. Bilder i {OUTPUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
