#!/usr/bin/env python3
"""Build the lineage DAG from metadata.json only (no narrative sources).

Rules (all deterministic, documented in the page's Method panel):
  1. Trait set  T(m): categorical architectural traits derived from the config.
  2. Score(a -> b) for a released strictly before b:
        10 * weighted-Jaccard(T(a), T(b))
      +  4 * numeric closeness (layers, hidden, heads, kv heads, head_dim, experts, top-k, vocab)
      +  3 if same HF architecture class (identical modeling code)
      +  2 if same model_type family
      +  2 if identical vocab_size (same tokenizer)
      +  1 if same organisation
  3. Primary parent of b = argmax_a Score(a -> b), if Jaccard >= 0.62 (0.45 when same org), else b is a root.
     Edge type "nearest" (or "same-code" when architecture class is identical).
  4. Same-day siblings with identical model_type + org attach to the largest one
     of the day (edge type "variant") instead of competing for earlier parents.
  5. Trait-origin edges: for each notable trait of b that its primary parent
     lacks, add an edge from the earliest earlier model carrying that trait
     (ties -> highest Score). Edge type "trait".
  6. Generation (the x position in the figure): roots are generation 0; every
     other model is one generation past the largest generation among its parents.
"""
import json, math, os, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(f"{ROOT}/data/metadata.json"))
M.sort(key=lambda m: (m["date"], m["key"]))
by = {m["key"]: m for m in M}

# ----------------------------------------------------------------------------
# 1. trait sets
# ----------------------------------------------------------------------------
W = {  # weights per trait dimension
    "attn": 3, "moe": 3, "hybrid": 3, "sparse": 2, "swa": 2, "swa_ratio": 1, "mtp": 1.5, "qk_norm": 1, "gated_attn": 1.5,
    "nope": 1, "pos": 1, "norm": 0.5, "act": 0.5, "shared_expert": 1, "dense_prefix": 1, "expert_granularity": 1,
    "partial_rotary": 0.5, "tie": 0.3, "mhc": 1.5, "kv_sharing": 1.5, "ple": 1.5, "looped": 2, "shortconv": 1,
    "sinks": 1, "chunked": 1, "latent_moe": 1.5, "enc_dec": 2, "softcap": 0.5, "multimodal": 0.3, "index_share": 1,
}
NOTABLE = {"attn", "moe", "hybrid", "sparse", "swa", "mtp", "qk_norm", "gated_attn", "nope", "mhc", "kv_sharing", "ple",
           "looped", "shortconv", "sinks", "chunked", "latent_moe", "enc_dec", "index_share", "shared_expert", "dense_prefix"}
LABEL = {
    "attn": "attention", "moe": "FFN", "hybrid": "sequence mixer", "sparse": "sparse attention", "swa": "sliding window",
    "swa_ratio": "local:global ratio", "mtp": "MTP head", "qk_norm": "QK-Norm", "gated_attn": "gated attention", "nope": "NoPE",
    "pos": "positional encoding", "norm": "norm", "act": "activation", "shared_expert": "shared expert", "dense_prefix": "dense prefix layers",
    "expert_granularity": "expert granularity", "partial_rotary": "partial RoPE", "tie": "tied embeddings", "mhc": "mHC / hyper-connections",
    "kv_sharing": "cross-layer KV sharing", "ple": "per-layer embeddings", "looped": "looped depth sharing", "shortconv": "short conv",
    "sinks": "attention sinks", "chunked": "chunked attention", "latent_moe": "latent MoE", "enc_dec": "encoder-decoder", "softcap": "logit softcapping",
    "multimodal": "multimodal wrapper", "index_share": "IndexShare",
}


def act_class(a):
    a = (a or "").lower()
    if "gelu" in a:
        return "gelu"
    if "relu2" in a or "squared" in a:
        return "relu2"
    if "poly" in a:
        return "polynorm"
    if a in ("silu", "swiglu", "swish", "swiglu_clamped", "situ"):
        return "silu/swiglu" if a != "swiglu_clamped" else "clamped swiglu"
    return a or "silu/swiglu"


def granularity(n):
    if not n:
        return None
    return "coarse (<=16)" if n <= 16 else ("mid (<128)" if n < 128 else "fine (>=128)")


def traits(m):
    t = {}
    t["attn"] = m["attention_class"]
    t["moe"] = "MoE" if m["is_moe"] else "dense"
    t["hybrid"] = m["hybrid_mixer"] or "attention-only"
    t["sparse"] = m["sparse_attention"] or "none"
    t["swa"] = "yes" if m["swa"] else "no"
    if m["swa"] and m["swa_ratio"]:
        r = m["swa_ratio"]
        t["swa_ratio"] = "1:1" if r < 2 else ("3:1" if r < 4 else ("5:1" if r < 6 else "6:1+"))
    t["mtp"] = "yes" if m["mtp"] else "no"
    t["qk_norm"] = "yes" if m["qk_norm"] else "no"
    t["gated_attn"] = "yes" if m["gated_attention"] else "no"
    t["nope"] = m["nope"]
    t["pos"] = m["position_encoding"]
    t["norm"] = m["norm_type"]
    t["act"] = act_class(m["activation"])
    if m["is_moe"]:
        t["shared_expert"] = "yes" if (m["shared_experts"] or 0) > 0 else "no"
        t["dense_prefix"] = "yes" if (m["dense_prefix_layers"] or 0) > 0 else "no"
        t["expert_granularity"] = granularity(m["num_experts"])
    t["partial_rotary"] = "yes" if m["partial_rotary"] else "no"
    t["tie"] = "yes" if m["tie_embeddings"] else "no"
    for k, f in [("mhc", "mhc"), ("kv_sharing", "kv_sharing"), ("ple", "per_layer_embeddings"), ("looped", "looped"), ("shortconv", "shortconv"),
                 ("sinks", "attention_sinks"), ("chunked", "chunked_attention"), ("latent_moe", "latent_moe"), ("enc_dec", "encoder_decoder"),
                 ("softcap", "logit_softcapping"), ("multimodal", "multimodal"), ("index_share", "index_share")]:
        if m.get(f):
            t[k] = "yes"
    return {k: v for k, v in t.items() if v is not None}


for m in M:
    m["traits"] = traits(m)


def jaccard(a, b):
    keys = set(a) | set(b)
    inter = sum(W.get(k, 1) for k in keys if k in a and k in b and a[k] == b[k])
    union = sum(W.get(k, 1) for k in keys)
    return inter / union if union else 0.0


def num_close(a, b):
    fields = ["num_layers", "hidden_size", "num_heads", "num_kv_heads", "head_dim", "num_experts", "experts_per_tok", "vocab_size"]
    s, n = 0.0, 0
    for f in fields:
        x, y = a.get(f), b.get(f)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x > 0 and y > 0:
            s += math.exp(-abs(math.log(x / y)))
            n += 1
    return s / n if n else 0.0


def score(a, b):
    j = jaccard(a["traits"], b["traits"])
    nc = num_close(a, b)
    bonus = {}
    if a["architecture_class"] and a["architecture_class"] == b["architecture_class"]:
        bonus["same architecture class"] = 3
    if a["model_type"] and a["model_type"] == b["model_type"]:
        bonus["same model_type"] = 2
    if a["vocab_size"] and a["vocab_size"] == b["vocab_size"]:
        bonus["identical vocab_size"] = 2
    if a["org"] == b["org"]:
        bonus["same organisation"] = 1
    total = 10 * j + 4 * nc + sum(bonus.values())
    return total, j, nc, bonus


def size_key(m):
    return (m.get("num_layers") or 0) * (m.get("hidden_size") or 0) * (1 + (m.get("num_experts") or 0))


# ----------------------------------------------------------------------------
# 2-4. parents
# ----------------------------------------------------------------------------
edges = []
# same-day sibling hubs
groups = collections.defaultdict(list)
for m in M:
    groups[(m["date"], m["model_type"], m["org"])].append(m)
hub_of = {}
for g in groups.values():
    if len(g) > 1:
        hub = max(g, key=size_key)
        for m in g:
            if m is not hub:
                hub_of[m["key"]] = hub["key"]

JACC_MIN = 0.62      # general threshold
JACC_MIN_SAME_ORG = 0.45  # a same-organisation predecessor needs less architectural overlap
for i, b in enumerate(M):
    if b["key"] in hub_of:
        h = by[hub_of[b["key"]]]
        tot, j, nc, bonus = score(h, b)
        edges.append({"source": h["key"], "target": b["key"], "type": "variant", "score": round(tot, 2), "jaccard": round(j, 3),
                      "numeric": round(nc, 3), "bonus": bonus, "shared": sorted(k for k in b["traits"] if h["traits"].get(k) == b["traits"][k]),
                      "differs": sorted(k for k in set(b["traits"]) | set(h["traits"]) if h["traits"].get(k) != b["traits"].get(k)),
                      "why": "same-day release, identical modeling class and organisation; attached to the largest sibling"})
        b["primary_parent"] = h["key"]
        continue
    cands = [a for a in M[:i] if a["date"] < b["date"]]
    if not cands:
        b["primary_parent"] = None
        b["root_reason"] = "first model in the gallery"
        continue
    scored = sorted(((score(a, b), a) for a in cands), key=lambda x: -x[0][0])
    (tot, j, nc, bonus), a = scored[0]
    b["candidates"] = [{"key": x["key"], "score": round(s[0], 2), "jaccard": round(s[1], 3)} for s, x in scored[:5]]
    thr = JACC_MIN_SAME_ORG if a["org"] == b["org"] else JACC_MIN
    if j < thr:
        b["primary_parent"] = None
        b["root_reason"] = f"no earlier model shares enough architecture (best weighted Jaccard {j:.2f} < {thr})"
    else:
        etype = "same-code" if "same architecture class" in bonus else "nearest"
        edges.append({"source": a["key"], "target": b["key"], "type": etype, "score": round(tot, 2), "jaccard": round(j, 3), "numeric": round(nc, 3),
                      "bonus": bonus, "shared": sorted(k for k in b["traits"] if a["traits"].get(k) == b["traits"][k]),
                      "differs": sorted(k for k in set(b["traits"]) | set(a["traits"]) if a["traits"].get(k) != b["traits"].get(k)),
                      "why": "highest config-similarity score among all earlier models"})
        b["primary_parent"] = a["key"]

# ----------------------------------------------------------------------------
# 5. trait-origin edges
# ----------------------------------------------------------------------------
NEG = {"no", "none", "dense", "attention-only", "rope", "rmsnorm", "silu/swiglu"}
for i, b in enumerate(M):
    if b["key"] in hub_of:
        continue
    p = by.get(b["primary_parent"]) if b.get("primary_parent") else None
    novel = [k for k in NOTABLE if k in b["traits"] and b["traits"][k] not in NEG and (p is None or p["traits"].get(k) != b["traits"][k])]
    origins = collections.defaultdict(list)
    for k in novel:
        v = b["traits"][k]
        carriers = [a for a in M[:i] if a["date"] < b["date"] and a["traits"].get(k) == v]
        if not carriers:
            b.setdefault("introduces", []).append(f"{LABEL[k]} = {v}")
            continue
        first_date = min(a["date"] for a in carriers)
        firsts = [a for a in carriers if a["date"] == first_date]
        origin = max(firsts, key=lambda a: score(a, b)[0])
        if origin["key"] == b.get("primary_parent"):
            continue
        origins[origin["key"]].append(f"{LABEL[k]} = {v}")
    for src, labels in origins.items():
        a = by[src]
        tot, j, nc, bonus = score(a, b)
        edges.append({"source": src, "target": b["key"], "type": "trait", "score": round(tot, 2), "jaccard": round(j, 3), "numeric": round(nc, 3),
                      "bonus": bonus, "traits": labels, "why": "earliest gallery model carrying: " + "; ".join(labels)})

# ----------------------------------------------------------------------------
# lineage lanes for layout: follow primary parents to a root
# ----------------------------------------------------------------------------
def root_of(k):
    seen = set()
    while by[k].get("primary_parent") and k not in seen:
        seen.add(k)
        k = by[k]["primary_parent"]
    return k


for m in M:
    m["lane_root"] = root_of(m["key"])
    m["month"] = m["date"][:7]

# generation: roots sit at 0; every other node sits one past its latest parent (over all edge types)
gen = {}
parents_by = collections.defaultdict(list)
for e in edges:
    parents_by[e["target"]].append(e["source"])


def generation(k):
    if k not in gen:
        gen[k] = 0 if not parents_by[k] else 1 + max(generation(p) for p in parents_by[k])
    return gen[k]


for m in M:
    m["generation"] = generation(m["key"])

children = collections.Counter(e["source"] for e in edges)
for m in M:
    m["is_terminal"] = children[m["key"]] == 0
    m["is_root"] = not m.get("primary_parent")

graph = {"nodes": M, "edges": edges, "trait_weights": W, "trait_labels": LABEL, "jaccard_min": JACC_MIN, "jaccard_min_same_org": JACC_MIN_SAME_ORG,
         "generated": "2026-09-13", "source": "https://sebastianraschka.com/llm-architecture-gallery/ (card list + config.json links); HuggingFace config.json per model"}
json.dump(graph, open(f"{ROOT}/data/graph.json", "w"), indent=1, ensure_ascii=False)

# report
print(f"nodes {len(M)} edges {len(edges)} :", collections.Counter(e['type'] for e in edges))
print("roots:", [m["key"] for m in M if m["is_root"]])
for e in edges:
    if e["type"] != "variant":
        print(f"{e['type']:9s} {e['source']:28s} -> {e['target']:28s} {e['score']:5.2f} j={e['jaccard']:.2f} {e.get('traits','')}")
