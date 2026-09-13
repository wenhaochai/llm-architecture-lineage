#!/usr/bin/env python3
"""Step 1: read the model list from Sebastian Raschka's LLM Architecture Gallery.

Every gallery card is an <article class="llm-architecture-overview__card"> whose
data-* attributes carry the card's title, organisation, release date and the
gallery's own one-line description of the attention / decoder type. The card body
links to the model's config.json on Hugging Face. We keep all of it in cards.json;
only the config link is used downstream to build edges, the descriptive fields are
kept as an independent cross-check for extract.py.

    python3 fetch_gallery.py            # downloads the page
    python3 fetch_gallery.py page.html  # or parses a saved copy
"""
import html, json, os, re, sys, urllib.request

URL = "https://sebastianraschka.com/llm-architecture-gallery/"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "cards.json")


def main():
    if len(sys.argv) > 1:
        page = open(sys.argv[1], encoding="utf-8").read()
    else:
        page = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})).read().decode("utf-8")
    cards = []
    for raw in re.split(r'<article class="llm-architecture-overview__card"', page)[1:]:
        head = raw.split(">", 1)[0]
        attrs = {k: html.unescape(v) for k, v in re.findall(r'data-([a-z0-9-]+)="([^"]*)"', head)}
        body = raw.split("</article>")[0]

        def link(label):
            m = re.search(r'href="([^"]+)"[^>]*>\s*' + label, body)
            return html.unescape(m.group(1)) if m else None

        cards.append({
            "key": attrs.get("compare-key"), "title": attrs.get("compare-title"), "base": attrs.get("compare-base-title"),
            "company": attrs.get("company"), "date": attrs.get("sort-date"), "decoder": attrs.get("compare-decoder"),
            "attention": attrs.get("compare-attention"), "layer_mix": attrs.get("compare-layer-mix"), "scale": attrs.get("compare-scale"),
            "context": attrs.get("compare-context"), "kv": attrs.get("compare-kv"), "aai": attrs.get("sort-aai"),
            "concepts": re.findall(r'concept-link" href="/llm-architecture-gallery/([^/]+)/">', body),
            "config_url": link(r"config\.json"), "report_url": link("Tech report"), "article_url": link("View in article"),
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(cards, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"{len(cards)} cards -> {OUT}; {sum(1 for c in cards if not c['config_url'])} without a config link")


if __name__ == "__main__":
    main()
