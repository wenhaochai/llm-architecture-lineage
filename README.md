# LLM Architecture Lineage

An interactive lineage graph of the 103 open-weight models in Sebastian Raschka's
[LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/),
built from each model's `config.json` alone.

**Live page:** https://wenhaochai.com/blogs/llm-architecture-lineage.html
**Standalone viewer:** open `web/index.html` in a browser (no server needed).

Each model is linked to its closest architectural ancestors, and the figure is laid
out by generation: the three roots sit in generation 0, and every other model sits one
column past its latest parent. No edge comes from commentary about who borrowed what;
the graph is a deterministic function of the configs.

## Pipeline

```
fetch_gallery.py   gallery page  -> data/cards.json          (103 cards, config.json links)
fetch_configs.py   Hugging Face  -> data/configs/<key>.json  (103 configs, provenance recorded)
extract.py         configs       -> data/metadata.json       (one normalized schema, ~110 fields)
build_graph.py     metadata      -> data/graph.json          (nodes, edges, generations)
export_web.py      graph.json    -> web/data.js              (trimmed data for the viewer)
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

### 2. Normalized metadata

Config keys differ by model family: the 103 files use 571 distinct keys. `extract.py`
maps them onto one schema of about 110 fields: depth, width, heads, KV heads, head dimension, FFN size, MoE
expert counts and shared experts, MLA latent ranks, sliding-window layer schedules,
hybrid mixer type and ratio (Gated DeltaNet, Kimi Delta Attention, Mamba-2, Lightning
Attention, LIV convolution, mLSTM), sparse-attention indexers, MTP heads, QK-Norm, gated
attention, NoPE, cross-layer KV sharing, per-layer embeddings, looped depth sharing,
hyper-connections, RoPE parameters, router settings, and more. The complete original
config is kept per model (`config_full`, long lists summarised as value counts, checkpoint
quantization blocks reduced to method and bit width). Where the modeling class fixes a trait that the config does
not spell out (QK-Norm in Gemma 3, for example), the trait is attached by `model_type`
in the `IMPLIED` table and marked as implied. The gallery's own attention / layer-mix
text is stored only as a cross-check and never used to build edges.

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
  (0.45 within the same organisation); otherwise the model is a root. `graph.json` tags
  such an edge `nearest`, or `same-code` when both configs load into the same modeling class.
* Same-day siblings of one modeling class attach to the largest sibling (`variant`).
* For every notable trait a model carries that its first parent lacks, one more edge comes
  from the earliest gallery model that carried it (`trait`).

The tags record how an edge was derived and are exposed in the detail panel; the figure
draws every edge the same way.

Every edge runs forward in release time, so the graph is acyclic. Generation is 0 for a
root and one more than the largest generation among a node's parents.

Result: 103 nodes, 162 edges, 11 generations, 3 roots (GPT-2 XL, DeepSeek V3, xLSTM 7B).

## Caveats

The gallery is a curated sample, so a trait's "origin" is its first appearance among these
103 models; the literature may hold earlier ones. A nearest-config edge measures similarity
between two configs; which papers a team actually read is a separate question.

## Layout of the repository

```
data/cards.json        gallery card list with config / report links
data/configs/*.json    one config per model (with _provenance where non-canonical)
data/metadata.json     normalized per-model metadata, gallery text kept as cross-check
data/graph.json        nodes, edges with per-edge justification, trait weights, generations
web/index.html         standalone viewer
web/lineage.js         renderer (no dependencies)
web/data.js            trimmed graph for the viewer
```

## Credits

Model list and config links: Sebastian Raschka, *LLM Architecture Gallery*, 2026.
Configs: the respective Hugging Face repositories. Figure style follows the exploration
lineage DAGs of an ongoing paper project. Code is MIT licensed; the configs keep their
original licences.
