#!/usr/bin/env python3
"""Build the lineage DAG from metadata.json by online insertion in release order.

Mirror of the construction in web/lineage.js (the page rebuilds the graph live so the
origin threshold can be tuned there; this script writes the same graph at the default).

  0. All 103 gallery models take part; sizes of one release fold together through rule 3 when they
     share the design.
  1. Models are taken in release order. GPT-2 XL, the first, is the single origin.
  2. The model in hand is compared with every model already placed that is not a scale
     copy. The comparison is the design-change list of schema.py: design fields count when
     their value differs (per-layer schedules compared by the kinds of layer they contain),
     scale fields never count by value and the PRESENCE subset counts on appearance, tuning
     fields and NOT_A_CHANGE fields never count.
  3. The placed model needing the fewest changes becomes the parent. Ties: same modeling
     class, then same organisation, then the later release. Zero changes make the new model a
     scale copy: it can never be a parent and is folded into its parent's node, which keeps
     the earliest name and lists the copies as aliases.
  4. If even the closest placed model needs more than ORIGIN_THRESHOLD changes, the model
     hangs off GPT-2 XL instead.
  5. For every field the model adds or switches to, one more edge comes from the earliest
     placed model that already carried it (second kind of edge, "trait").
  6. Generation = 1 + max generation over all parents; it is the column in the figure.
"""
import json, os, collections
import schema

ROOT = os.path.dirname(os.path.abspath(__file__))
ORIGIN_THRESHOLD = 32

M_ALL = json.load(open(f"{ROOT}/data/metadata.json"))


def params_total(m):
    """Total parameter count parsed from the gallery scale string, e.g. '284B total, 30B active' -> 284e9."""
    import re
    s = (m.get("gallery") or {}).get("scale") or ""
    mt = re.match(r"\s*([\d.]+)\s*([BMT])", s)
    if not mt:
        return 0
    return float(mt.group(1)) * {"M": 1e6, "B": 1e9, "T": 1e12}[mt.group(2)]


M_ALL.sort(key=lambda m: (m["date"], -params_total(m), m["key"]))

M = M_ALL
by = {m["key"]: m for m in M}

ORDER = [f for g in schema.GROUPS.values() for f in g]


def absent(v):
    if v is None or v is False or v == 0 or v == [] or v == "":
        return True
    if isinstance(v, dict) and not v.get("_list_len") and "_raw" not in v and not v:
        return True
    return False


def shape(v):
    if isinstance(v, dict) and v.get("_list_len"):
        return sorted(v["_counts"])
    if isinstance(v, list):
        return sorted(v, key=str)
    return v


def same(a, b):
    return json.dumps(shape(a), sort_keys=True) == json.dumps(shape(b), sort_keys=True)


def changes(x, p):
    a, b, out = p["config_canonical"], x["config_canonical"], []
    for k in ORDER:
        if k in schema.NOT_A_CHANGE or k in schema.TUNING or (k in schema.SCALE and k not in schema.PRESENCE):
            continue
        va, vb = a.get(k), b.get(k)
        has_a, has_b = not absent(va), not absent(vb)
        if has_a and has_b:
            if k not in schema.SCALE and not same(va, vb):
                out.append({"field": k, "kind": "change"})
        elif has_b:
            out.append({"field": k, "kind": "add"})
        elif has_a:
            out.append({"field": k, "kind": "drop"})
    return out


edges = []
parents = collections.defaultdict(list)
children = collections.defaultdict(list)


def add_edge(src, dst, etype, fields, n):
    e = {"source": src["key"], "target": dst["key"], "type": etype, "fields": fields, "n": n}
    edges.append(e); parents[dst["key"]].append(e); children[src["key"]].append(e)


origin = M[0]
origin.update(primary_parent=None, scale_copy=False, changes=[], generation=0, is_root=True)
placed = [origin]
for x in M[1:]:
    x["is_root"] = False
    eligible = [p for p in placed if not p["scale_copy"]]
    best, best_n = None, None
    for p in eligible:
        n = len(changes(x, p))
        if best is None or n < best_n:
            best, best_n = p, n
        elif n == best_n:
            key = lambda c: ((c["architecture_class"] == x["architecture_class"]), (c["org"] == x["org"]), c["date"])
            if key(p) > key(best):
                best = p
    if best is None or best_n > ORIGIN_THRESHOLD:
        x["primary_parent"], x["changes"], x["scale_copy"] = origin["key"], changes(x, origin), False
        add_edge(origin, x, "origin", [], len(x["changes"]))
    else:
        x["primary_parent"], x["changes"], x["scale_copy"] = best["key"], changes(x, best), best_n == 0
        add_edge(best, x, "parent", [], best_n)
    origins = collections.OrderedDict()
    for c in x["changes"]:
        if c["kind"] == "drop":
            continue
        src = None
        for p in eligible:
            if p["key"] == x["primary_parent"]:
                continue
            v = p["config_canonical"].get(c["field"])
            if absent(v):
                continue
            if c["kind"] == "change" and not same(v, x["config_canonical"][c["field"]]):
                continue
            if src is None or (p["date"], p["key"]) < (src["date"], src["key"]):
                src = p
        if src is not None:
            origins.setdefault(src["key"], []).append(c["field"])
    for k, fields in origins.items():
        add_edge(by[k], x, "trait", fields, len(fields))
    x["generation"] = 1 + max(by[e["source"]]["generation"] for e in parents[x["key"]])
    placed.append(x)

# a scale copy is the same design: fold it into the model it copies, which keeps the earliest name
for m in M:
    m["aliases"], m["merged_into"] = [], None
for m in M:
    if m["scale_copy"]:
        p = by[m["primary_parent"]]
        p["aliases"].append({"key": m["key"], "title": m["title"], "date": m["date"], "org": m["org"], "scale": m["gallery"].get("scale")})
        m["merged_into"] = p["key"]
        edges = [e for e in edges if e["target"] != m["key"]]
        children[p["key"]] = [e for e in children[p["key"]] if e["target"] != m["key"]]
for m in M:
    m["is_terminal"] = not children[m["key"]]

graph = {"nodes": M, "edges": edges, "n_gallery": len(M_ALL), "origin_threshold": ORIGIN_THRESHOLD,
         "generated": "2026-09-13", "source": "https://sebastianraschka.com/llm-architecture-gallery/ (card list + config.json links); HuggingFace config.json per model"}
json.dump(graph, open(f"{ROOT}/data/graph.json", "w"), indent=1, ensure_ascii=False)

drawn = [m for m in M if not m["merged_into"]]
print(f"models {len(M)} drawn nodes {len(drawn)} edges {len(edges)} :", collections.Counter(e["type"] for e in edges),
      "| scale copies folded in:", sum(1 for m in M if m["scale_copy"]), "| generations:", max(m["generation"] for m in drawn) + 1)
for e in edges:
    if e["type"] != "trait":
        print(f"{e['type']:7s} {e['source']:28s} -> {e['target']:28s} n={e['n']}")
