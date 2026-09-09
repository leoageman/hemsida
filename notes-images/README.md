# OBC "noter"-produktbilder (Nano Banana Pro)

Pipeline för att skapa produktbilder där parfymflaskan står i mitten och fyra av doftens
råvaror ligger runt foten, i samma stil som de befintliga bilderna
`reference/existing-notes-images/` (Béchamp 1.0, Berzelius 10.0, Ostwald 18.0,
Sørensen 26.0, Thompson 41.0 som redan ligger i Shopify).

## Vad som finns här

| Sökväg | Innehåll |
|---|---|
| `data/products.json` | Alla 77 parfymer från Shopify (nummer, namn, SKU, handle, product-id, topp-/mellan-/basnoter, bildlänkar). Byggs av `scripts/build_manifest.py`. |
| `data/selection.json` | De fyra noter som valts för 1.0–10.0 + hur varje ingrediens ska se ut i bild. Lägg till fler nummer här för att köra fler parfymer. |
| `data/shopify-raw/` | Råa Shopify Admin API-svar (GraphQL `products`) som manifestet byggs från. |
| `reference/bottles/` | Nedladdade flaskbilder (1024 px) för 1.0–10.0. Skickas som referensbild till modellen. |
| `reference/existing-notes-images/` | De fem noter-bilder som redan finns i Shopify, som stilreferens. |
| `prompts/` | Färdiga prompts per parfym (genereras av `scripts/generate.py`). Kan klistras in manuellt i Gemini/AI Studio tillsammans med flaskbilden. |
| `output/final/` | **Färdiga bilder 1.0–10.0**: gpt-image-2 i 2048 px, normaliserade så att flaskan har exakt samma storlek och position i alla bilder. |
| `output/*_noter_gpt.png` | Råa gpt-image-2-bilder före normalisering. |
| `output/*_noter.jpg` | Nano Banana Pro-versionerna (samma prompt) som alternativ. |

## Köra genereringen

```bash
cd notes-images
pip install -r requirements.txt
export GEMINI_API_KEY=...        # https://aistudio.google.com/apikey

python3 scripts/generate.py --dry-run          # skriver bara prompts/
python3 scripts/generate.py --skus 1-10 --style-ref reference/style-ref.jpg   # parfym 1.0–10.0, 2K, med stilreferens
python3 scripts/generate.py --skus 3 --force   # gör om en enskild
```

Modell: `gemini-3-pro-image-preview` (Nano Banana Pro; testbilderna 1.0–10.0 kördes med `--model gemini-3-pro-image`). Byt med `--model` eller
`NANO_BANANA_MODEL=...`; `--list-models` visar vad nyckeln har tillgång till.
Storlek: `--size 1K|2K|4K` (default 2K = 2048 px, samma som befintliga noter-bilder).

GPT Image som alternativ: `python3 scripts/generate.py --provider openai --model gpt-image-2 --skus 2` (kräver `OPENAI_API_KEY`, max 1024 px).

`--style-ref` skickar en färdig, godkänd bild som bild 2 till modellen så att vinkel, ljus och inramning hålls konsekvent. `reference/style-ref.jpg` (Avogadro 6.0) är den bild som användes för 1.0–10.0: kamera ca 20° ovanifrån, helt vit bakgrund, allt inom de centrala 75 % av bilden.

## Slutligt flöde (det som användes för 1.0–10.0)

```bash
python3 scripts/generate.py --provider openai --model gpt-image-2 --size 2K --style-ref reference/style-ref.jpg --skus 1-10
python3 scripts/normalize.py output/*_noter_gpt.png --out output/final --cap-top 0.15 --fit
```

`normalize.py` mäter kapsylens överkant och bredd i varje bild (bakgrunden är helt vit) och skalar/förskjuter så att
alla flaskor får samma storlek och plats. `--fit` väljer den största storlek som ryms i alla bilder utan att någon
ingrediens klipps; `--cap-width 0.14` låser samma storlek när fler parfymer körs senare, så nya bilder matchar de gamla.

## Slutbilder och normalisering

`output/final/` är leveransen: gpt-image-2-bilderna (`output/*_noter_gpt.png`) normaliserade så att
flaskan har exakt samma storlek och position i alla bilder och bakgrunden är exakt vit.

```bash
python3 scripts/generate.py --provider openai --model gpt-image-2 --size 2K --style-ref reference/style-ref.jpg --skus 1-10
python3 scripts/normalize.py output/*_noter_gpt.png --out output/final --cap-top 0.15 --fit
```

`normalize.py` vitbalanserar bakgrunden, mäter kapsylens överkant och bredd, och skalar/förskjuter så att
kapsyltoppen hamnar på 15 % av höjden. `--fit` väljer största gemensamma flaskstorlek som ryms i alla
bilder utan att något klipps; för att nya bilder ska matcha de här tio, använd i stället `--cap-width 0.134`.

## Editorial-stil (gradientbakgrund + glashylla)

`--style editorial` ger kampanjbilder: flaskan stor på en glashylla, ingredienserna tätt kring foten och
en vertikal gradientbakgrund i originalparfymens eller etikettens färger (`backdrop` i `data/selection.json`,
motivering i `backdrop_source`). Leveransen ligger i `output/final_editorial/`.

```bash
python3 scripts/generate.py --provider openai --model gpt-image-2 --size 2K --style editorial --style-ref reference/style-ref-editorial.jpg --skus 1-10
```

Stilreferensen är Bachelder 4.0. OBS: referensen visar en annan doft, och prompten säger uttryckligen att
etikettext, illustration och vätskefärg aldrig får kopieras från den. Kontrollera ändå etiketten på varje bild.

## Manuellt i Gemini-appen / AI Studio

1. Öppna `prompts/<nr>_<namn>.txt` och kopiera texten.
2. Ladda upp motsvarande flaskbild från `reference/bottles/`.
3. Välj bildmodellen (Nano Banana Pro) och kör.

## Lägga till fler parfymer

1. Hitta noterna i `data/products.json` (fältet `notes`).
2. Lägg till numret i `data/selection.json` med fyra `{"note": ..., "visual": ...}`.
   `note` ska vara exakt en not från Shopify; `visual` beskriver ingrediensen fysiskt på engelska.
3. Kör `python3 scripts/generate.py --skus <nr>`.

Uppdatera manifestet när Shopify ändras: spara nya API-svar i `data/shopify-raw/` och kör
`python3 scripts/build_manifest.py`.

## Noteringar

- Béchamp 1.0 har bara tre noter i Shopify (kardemumma, kola, amberträ). Amberträ visas därför som
  både harts och trä, som i den befintliga bilden.
- Becquerel 108.0 saknar "Toppnoter/Mellannoter/Basnoter"-block i beskrivningen och får därför
  `notes: null` i manifestet.
