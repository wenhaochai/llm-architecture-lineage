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
schema.py          alias table: 428 architecture keys -> 174 canonical fields in 14 sections
extract.py         configs       -> data/metadata.json       (canonical config + derived traits per model)
build_graph.py     metadata      -> data/graph.json          (83 nodes, edges, generations)
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

The 103 configs use 571 distinct keys. `key_taxonomy.py` lists by hand the 143 that carry no
architecture: 69 decoding defaults, tokenizer ids and Hugging Face bookkeeping; 40 dtype,
kernel, parallelism and implementation switches; 26 training-only settings; 8 multimodal
leftovers. `schema.py` renames the remaining 428 architecture keys into 174 canonical fields
(100 of them merge several spellings: seven names for the number of active experts, six for
the norm epsilon), each assigned to one of 14 sections. `KEYS.md` lists every key's fate.
`extract.py` writes the canonical config per model (`config_canonical`, with the original
spelling in `config_canonical_raw_key`) next to the derived traits used by the score. Where a
modeling class fixes a trait the config does not spell out (QK-Norm in Gemma 3), the trait is
attached by `model_type` in the `IMPLIED` table.

### 2b. One node per model name

Twelve releases appear in several sizes (six Qwen3 models, two GPT-OSS, ...). `SIZE_VARIANTS`
in `build_graph.py` keeps the largest of each, so 103 cards become 83 nodes; folded sizes are
recorded on the kept node.

### 3. Parents

Every edge means the same thing: the child's config is one step away from the parent's. Parents are chosen as follows. Each model becomes a set of about 30 categorical traits. For a candidate parent `a`
released strictly before `b`:

```
score(a -> b) = 10 * weighted Jaccard(traits(a), traits(b))
              +  4 * log-ratio closeness over layers, width, heads, KV heads, head_dim, experts, top-k, vocab
              +  3 if identical Hugging Face architecture class
              +  2 if identical model_type
              +  2 if identical vocab_size
              +  1 if same organisation
```

* The highest-scoring earlier model is the first parent when trait overlap reaches 0.62
  (0.45 within the same organisation); otherwise the model hangs off GPT-2 XL, the single
  origin (`origin`). `graph.json` tags
  such an edge `nearest`, or `same-code` when both configs load into the same modeling class.
* Same-day releases of one modeling class under different names attach to the larger one (`variant`).
* For every notable trait a model carries that its first parent lacks, one more edge comes
  from the earliest gallery model that carried it (`trait`).

The tags record how an edge was derived and are exposed in the detail panel; the figure
draws every edge the same way.

Every edge runs forward in release time, so the graph is acyclic. Generation is 0 for GPT-2 XL
and one more than the largest generation among a node's parents. The trait set still comes from
the first normalization pass; the canonical fields are displayed but do not yet enter the score.

Result: 83 nodes, 139 edges, 10 generations, one root.

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
