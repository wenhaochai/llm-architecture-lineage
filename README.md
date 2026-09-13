# LLM Architecture Lineage

An interactive lineage graph of the open-weight models in Sebastian Raschka's
[LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/),
built from each model's `config.json` alone: 103 gallery cards, 83 nodes once the sizes of one
release are folded into the largest.

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
schema.py          alias table: 429 architecture keys -> 174 canonical fields in 14 sections; design/scale/tuning kinds
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

The 103 configs use 571 distinct keys. `key_taxonomy.py` lists by hand the 142 that carry no
architecture: 68 decoding defaults, tokenizer ids and Hugging Face bookkeeping; 40 dtype,
kernel, parallelism and implementation switches; 26 training-only settings; 8 multimodal
leftovers. `schema.py` renames the remaining 429 architecture keys into 174 canonical fields
(100 of them merge several spellings: seven names for the number of active experts, six for
the norm epsilon), each assigned to one of 14 sections. `KEYS.md` lists every key's fate.
Each canonical field is tagged design (67), scale (66) or tuning (41) in `schema.py`. The change
line under a model compares canonical configs with the first parent: design fields count when
their value differs (layer schedules by the kinds of layer they contain), scale fields never
count by value and 17 of them count on appearance because they mark a mechanism (`PRESENCE`),
tuning fields and MTP heads (`NOT_A_CHANGE`) never count. A wider or deeper copy of the same
design therefore reads as no change. `extract.py` writes the canonical config per model (`config_canonical`, with the original
spelling in `config_canonical_raw_key`) next to the derived traits used by the score. Where a
modeling class fixes a trait the config does not spell out (QK-Norm in Gemma 3), the trait is
attached by `model_type` in the `IMPLIED` table.

### 2b. One node per model name

Twelve releases appear in several sizes (six Qwen3 models, two GPT-OSS, ...). `SIZE_VARIANTS`
in `build_graph.py` keeps the largest of each, so 103 cards become 83 nodes; folded sizes are
recorded on the kept node.

### 3. The graph: online insertion in release order

`build_graph.py` (mirrored live in `web/lineage.js`, where the threshold is a slider) takes the
83 models in release order; on one day the larger model comes first. GPT-2 XL opens the graph.
Every later model is compared with each placed model that is not a scale copy, using the change
list above, and attaches under the one with the fewest changes (ties: same modeling class, then
same organisation, then the later release). Zero changes make it a scale copy: a diamond in the
figure, never a parent. If even the closest placed model needs more than `ORIGIN_THRESHOLD`
changes (12), the model hangs off GPT-2 XL. For every field the model adds or switches to, one
more edge (`trait`) comes from the earliest placed model that already carried it. Generation is
one more than the largest generation among a node's parents and is the column in the figure.

Result at the default threshold: 83 nodes, 189 edges (80 parent, 107 trait, 2 origin), 18 scale
copies, 12 generations.

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
