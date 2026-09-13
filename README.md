# LLM Architecture Lineage

An interactive lineage graph of the open-weight models in Sebastian Raschka's
[LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/),
built from each model's `config.json` alone: 103 gallery models, 79 nodes once models that only
change scale fold into the design they copy.

**Live page:** https://wenhaochai.com/blogs/llm-architecture-lineage.html
**Standalone viewer:** open `web/index.html` in a browser (no server needed).

Each model is linked to its closest architectural ancestors, and the figure is laid
out by generation: GPT-2 XL is the single origin, and every other model sits one column past
its latest parent. No edge comes from commentary about who borrowed what;
the graph is a deterministic function of the configs.

## Pipeline

```
fetch_gallery.py   gallery page  -> data/cards.json          (103 cards, config.json links)
fetch_configs.py   Hugging Face  -> data/configs/<key>.json  (103 configs, provenance recorded)
key_taxonomy.py    hand-made lists of the 143 non-architecture keys (dropped)
schema.py          alias table: 416 architecture keys -> 170 canonical fields (+5 derived) in 4 sections; design/scale/tuning kinds
extract.py         configs       -> data/metadata.json       (canonical config + derived traits per model)
build_graph.py     metadata      -> data/graph.json          (online insertion: parents, trait edges, generations)
export_web.py      graph.json    -> web/data.js              (trimmed data for the viewer)
KEYS.md            every key's fate, generated
```

Run them in that order with Python 3.9+ and no third-party packages:

```bash
python3 fetch_gallery.py && python3 fetch_configs.py && python3 extract.py && python3 build_graph.py && python3 export_web.py
```

### 1. Configs

The gallery links the canonical Hugging Face repository of every card. Eleven repositories
are gated. Six have ungated mirrors carrying the identical file (Llama 3, Llama 3.2 ×2,
Gemma 3 ×2, Llama 4 Maverick) and two more surface the file through a quantized mirror
(Tiny Aya, Antares). Mistral Large 3 ships a native `params.json`, converted field by
field. BTL-3 is a LoRA adapter whose `adapter_config.json` names Qwen3.6-27B as its base.
Soofi-S states in its model card and technical report that it adopts the Nemotron 3 Nano
architecture without modification, so its config is reconstructed from that one and
checked against the paper's hyperparameter table. Every non-canonical source is written
into the saved file under `_provenance`.

### 2. Key cleaning and renaming

The 103 configs use 571 distinct keys. `key_taxonomy.py` lists by hand the 155 that carry no
architecture: 68 decoding defaults, tokenizer ids and Hugging Face bookkeeping; 40 dtype,
kernel, parallelism and implementation switches; 26 training-only settings; 8 multimodal
leftovers, 11 multi-token-prediction heads, and the 2 keys naming the modeling code.
`schema.py` renames the remaining 416 architecture keys into 170 canonical fields (98 of them
merge several spellings: seven names for the number of active experts, six for
the norm epsilon), each assigned to one of four sections that follow the forward pass of a decoder
block (token mixing and channel mixing carry subsections). Nested sub-configs are flattened into
the same fields and per-layer index lists become layer schedules by kind. `KEYS.md` lists every key's fate.
Each canonical field is tagged design (67), scale (66) or tuning (42) in `schema.py`. The change
line under a model compares canonical configs with the first parent: design fields count when
their value differs (layer schedules by the kinds of layer they contain), scale fields never
count by value and 16 of them count on appearance because they mark a mechanism (`PRESENCE`),
tuning fields never count; `RESTATED` fields say what the layer schedule already says; and a field
in `CONDITIONAL` is skipped when the mechanism it belongs to is itself the change. A key written
with its default is settled to the same value as an absent key (`DEFAULTS`, `NO_EFFECT`). A wider or deeper copy of the same
design therefore reads as no change. `extract.py` writes the canonical config per model (`config_canonical`, with the original
spelling in `config_canonical_raw_key`) next to the derived traits used by the score. Where a
modeling class fixes a trait the config does not spell out (QK-Norm in Gemma 3), the trait is
attached by `model_type` in the `IMPLIED` table.

### 3. The graph: online insertion in release order

`build_graph.py` (mirrored in `web/lineage.js`) takes the
103 models in release order; on one day the larger model comes first. GPT-2 XL opens the graph.
Every later model is compared with each placed model that is not a scale copy, using the change
list above. Candidates are ranked by the number of mechanism-level changes (`MECHANISM` in
`schema.py`: attention kind, sequence mixer, MoE, sparse attention, positions, hyper-connections,
looped depth, per-layer embeddings, KV sharing, encoder-decoder, n-gram memories, gated attention,
QK-Norm, attention residuals, gated residuals), then by the total number of changes, then same modeling class, same organisation, earlier
release (the earlier of two equally close designs is where that design came from). Zero changes make it a scale copy: it is folded into
its parent's node, which keeps the earliest name and lists the copies as aliases, and it can
never be a parent. If even the closest placed model needs more than `ORIGIN_THRESHOLD` changes
(32), the model hangs off GPT-2 XL; no model currently does, the largest gap being 13. For every field the model adds or switches to, one more edge
(`trait`) comes from the earliest placed model that already carried it. Generation is one more
than the largest generation among a node's parents and is the column in the figure.

Result: 103 models, 33 folded as scale copies, 70 drawn nodes, 86 edges after reduction (63
parent, 23 trait), 9 generations.

## Caveats

The gallery is a curated sample, so a trait's "origin" is its first appearance among these
103 models; the literature may hold earlier ones. A nearest-config edge measures similarity
between two configs; which papers a team actually read is a separate question.

## Layout of the repository

```
data/cards.json        gallery card list with config / report links
data/configs/*.json    one config per model (with _provenance where non-canonical)
data/metadata.json     per-model canonical config + derived traits (all 103 models)
data/graph.json        83 nodes, edges with per-edge justification, trait weights, generations
web/index.html         standalone viewer
web/lineage.js         renderer (no dependencies)
web/data.js            trimmed graph for the viewer
```

## Credits

Model list and config links: Sebastian Raschka, *LLM Architecture Gallery*, 2026.
Configs: the respective Hugging Face repositories. Figure style follows the exploration
lineage DAGs of an ongoing paper project. Code is MIT licensed; the configs keep their
original licences.
