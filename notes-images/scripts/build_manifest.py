#!/usr/bin/env python3
"""Bygger data/products.json från sparade Shopify Admin API-svar.

Indata:  data/shopify-raw/*.json  (svar från GraphQL `products` med descriptionHtml,
         variants{sku} och media{image{url}})
Utdata:  data/products.json — en post per parfym (titel på formen "Namn N.0"),
         sorterad på nummer, med topp-/mellan-/basnoter parsade ur beskrivningen.

Kör:  python3 scripts/build_manifest.py
"""
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "shopify-raw"
OUT = ROOT / "data" / "products.json"

PERFUME_RE = re.compile(r"^(?P<name>.+?)\s+(?P<num>\d+)\.0$")
NOTES_RE = re.compile(
    r"Toppnoter:\s*(?P<top>.*?)\s*Mellannoter:\s*(?P<middle>.*?)\s*Basnoter:\s*(?P<base>.*?)\s*(?:Innehåller|Koncentration|$)",
    re.S,
)
# Taggar som inte är "inspirerad av"-taggar.
TAG_SKIP = {
    "male", "female", "unisex", "bestseller", "us-warehouse", "buy-now", "coming-soon",
    "hidden", "free-sample", "gift-with-purchase", "pdm", "ska ej finnas", "ultramale",
    "symphony", "louis vuitton",
}


def strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"</p>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).replace("\xa0", " ")


def split_notes(chunk: str) -> list[str]:
    seen, out = set(), []
    for n in chunk.split(","):
        n = " ".join(n.split()).strip(" ,.")
        if n and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def parse_notes(desc_html: str):
    text = " ".join(strip_html(desc_html).split())
    m = NOTES_RE.search(text)
    if not m:
        return None
    return {k: split_notes(m.group(k)) for k in ("top", "middle", "base")}


def intro_text(desc_html: str) -> str:
    first = strip_html(desc_html).strip().split("\n")[0]
    return " ".join(first.split())


def main() -> None:
    products = []
    for path in sorted(RAW_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for edge in data["data"]["products"]["edges"]:
            node = edge["node"]
            m = PERFUME_RE.match(node["title"].strip())
            if not m:
                continue
            images = [
                {"url": e["node"]["image"]["url"], "width": e["node"]["image"]["width"], "height": e["node"]["image"]["height"]}
                for e in node.get("media", {}).get("edges", [])
                if e.get("node", {}).get("image")
            ]
            inspired = [t for t in node.get("tags", []) if t.lower() not in TAG_SKIP]
            variants = node.get("variants", {}).get("edges", [])
            products.append(
                {
                    "number": int(m.group("num")),
                    "name": m.group("name"),
                    "title": node["title"].strip(),
                    "sku": (variants[0]["node"].get("sku") if variants else None),
                    "product_id": node["id"],
                    "handle": node["handle"],
                    "status": node.get("status"),
                    "inspired_by_tags": inspired,
                    "description": intro_text(node.get("descriptionHtml", "")),
                    "notes": parse_notes(node.get("descriptionHtml", "")),
                    "bottle_image": next((i["url"] for i in images if "noter" not in i["url"].lower()), None),
                    "existing_notes_image": next((i["url"] for i in images if "noter" in i["url"].lower()), None),
                    "images": images,
                }
            )
    # Dedupe on product_id (a product could appear on two pages if the catalog changes between calls).
    uniq = {p["product_id"]: p for p in products}
    products = sorted(uniq.values(), key=lambda p: p["number"])
    OUT.write_text(json.dumps(products, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    missing = [p["title"] for p in products if not p["notes"]]
    print(f"{len(products)} parfymer -> {OUT.relative_to(ROOT)}")
    if missing:
        print("Saknar parsade noter:", ", ".join(missing))


if __name__ == "__main__":
    main()
