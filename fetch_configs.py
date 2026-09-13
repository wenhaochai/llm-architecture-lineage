#!/usr/bin/env python3
"""Step 2: download every model's config.json into data/configs/<key>.json.

The gallery links the canonical Hugging Face repository for each card. Eleven of
those repositories are gated (HTTP 401 without accepting a licence), so this
script falls back, in order, to:
  * an ungated mirror that redistributes the identical config.json (Llama, Gemma,
    Tiny Aya, Antares);
  * the repository's native params.json converted field by field (Mistral Large 3);
  * the base model named in adapter_config.json when the card is a LoRA adapter
    (BTL-3 -> Qwen/Qwen3.6-27B);
  * a reconstruction from the model card + technical report when the model states
    that it adopts another gallery model's architecture unchanged (Soofi-S ->
    Nemotron 3 Nano; values checked against arXiv 2607.09424 Table 1).
Every non-canonical source is recorded in the saved file under "_provenance".
"""
import json, os, sys, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.abspath(__file__))
CARDS = os.path.join(ROOT, "data", "cards.json")
OUT = os.path.join(ROOT, "data", "configs")

# cards without a config link on the gallery page
EXTRA_URL = {
    "llama-3-2-1b": "https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json",
    "llama-3-2-3b": "https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json",
    "mistral-large-3": "https://huggingface.co/mistralai/Mistral-Large-3-675B-Base-2512/blob/main/params.json",
    "btl-3-27b": "https://huggingface.co/badtheorylabs/BTL-3/blob/main/adapter_config.json",
}
# ungated mirrors carrying the identical config.json
MIRROR = {
    "llama-3-8b": "unsloth/llama-3-8b",
    "llama-3-2-1b": "unsloth/Llama-3.2-1B",
    "llama-3-2-3b": "unsloth/Llama-3.2-3B",
    "gemma-3-27b": "unsloth/gemma-3-27b-it",
    "gemma-3-270m": "unsloth/gemma-3-270m-it",
    "llama-4-maverick": "unsloth/Llama-4-Maverick-17B-128E-Instruct",
    "tiny-aya-3-35b": "AMAImedia/Tiny-Aya-3.3B-L2-Thinker-BF16-GGUF",
    "antares-1b": "DKAvocadoo/antares-1b-mlx-bf16",
}
# LoRA adapters: the architecture is the base model's
ADAPTER_BASE = {"btl-3-27b": ("Qwen/Qwen3.6-27B", "qwen3-6-27b")}
# gated with no mirror; architecture declared identical to another gallery model
RECONSTRUCT = {
    "soofi-s-30b-a3b": {
        "from": "nemotron-3-nano-30b-a3b",
        "override": {"architectures": ["NemotronHForCausalLM"], "max_position_embeddings": 1048576},
        "note": "gated repo Soofi-Project/Soofi-S-Base; its model card states 'Nemotron 3 Nano reference architecture, adopted without "
                "modification'; values cross-checked with arXiv 2607.09424 Table 1 (52 layers = 23 Mamba-2 + 23 MoE + 6 GQA, d=2688, "
                "32 Q heads, 2 KV heads, head_dim 128, 128 routed + 2 shared experts, top-6, expert dim 1856, NoPE, RMSNorm, untied); "
                "HF GGUF metadata reports architecture nemotron_h_moe, context_length 1048576",
    }
}


def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, ""


def resolve(url):
    return url.replace("/blob/", "/resolve/")


def mistral_params_to_config(p):
    m = p["moe"]
    return {
        "_provenance": "converted from mistralai/Mistral-Large-3-675B-Base-2512 params.json (repo ships no HF config.json)",
        "architectures": ["MistralLarge3ForCausalLM"], "model_type": "mistral_large3",
        "hidden_size": p["dim"], "num_hidden_layers": p["n_layers"], "intermediate_size": p["hidden_dim"],
        "num_attention_heads": p["n_heads"], "num_key_value_heads": p["n_kv_heads"], "head_dim": p["head_dim"],
        "rope_theta": p["rope_theta"], "rms_norm_eps": p["norm_eps"], "vocab_size": p["vocab_size"], "tie_word_embeddings": p["tied_embeddings"],
        "max_position_embeddings": p["max_position_embeddings"], "rope_scaling": {"type": "yarn", **p["yarn"]},
        "q_lora_rank": p["q_lora_rank"], "qk_rope_head_dim": p["qk_rope_head_dim"], "qk_nope_head_dim": p["qk_nope_head_dim"],
        "kv_lora_rank": p["kv_lora_rank"], "v_head_dim": p["v_head_dim"],
        "first_k_dense_replace": m["first_k_dense_replace"], "n_routed_experts": m["num_experts"], "num_experts_per_tok": m["num_experts_per_tok"],
        "n_group": m["num_expert_groups"], "topk_group": m["num_expert_groups_per_tok"], "routed_scaling_factor": m["routed_scale"],
        "moe_intermediate_size": m["expert_hidden_dim"], "n_shared_experts": m["num_shared_experts"], "moe_layer_freq": m["route_every_n"],
        "hidden_act": "silu", "vision_config": p.get("vision_encoder"),
    }


def main():
    cards = json.load(open(CARDS))
    os.makedirs(OUT, exist_ok=True)
    pending = {}
    for c in cards:
        key, url = c["key"], c["config_url"] or EXTRA_URL.get(c["key"])
        path = os.path.join(OUT, key + ".json")
        if key in RECONSTRUCT or key in ADAPTER_BASE:
            pending[key] = url
            continue
        if url and url.endswith("params.json"):
            code, body = get(resolve(url))
            if code == 200:
                json.dump(mistral_params_to_config(json.loads(body)), open(path, "w"), indent=1)
                print(f"{key:36s} params.json -> converted")
                continue
        code, body = get(resolve(url)) if url else (0, "")
        src = url
        if code != 200 and key in MIRROR:
            src = f"https://huggingface.co/{MIRROR[key]}/resolve/main/config.json"
            code, body = get(src)
        if code == 200:
            cfg = json.loads(body)
            if src != url:
                cfg["_provenance"] = f"canonical repo is gated ({url}); identical config.json taken from ungated mirror {src}"
            json.dump(cfg, open(path, "w"), indent=1, ensure_ascii=False)
            print(f"{key:36s} {'mirror' if src != url else 'ok'}")
        else:
            print(f"{key:36s} FAILED {code} {url}", file=sys.stderr)
    for key, (base_repo, base_key) in ADAPTER_BASE.items():
        cfg = json.load(open(os.path.join(OUT, base_key + ".json")))
        cfg["_provenance"] = f"{key} is a PEFT LoRA adapter; adapter_config.json base_model_name_or_path = {base_repo}; base config copied from data/configs/{base_key}.json"
        json.dump(cfg, open(os.path.join(OUT, key + ".json"), "w"), indent=1, ensure_ascii=False)
        print(f"{key:36s} adapter -> {base_key}")
    for key, r in RECONSTRUCT.items():
        cfg = json.load(open(os.path.join(OUT, r["from"] + ".json")))
        for k in ("_name_or_path", "transformers_version", "quantization_config"):
            cfg.pop(k, None)
        cfg.update(r["override"])
        cfg["_provenance"] = "reconstructed: " + r["note"]
        json.dump(cfg, open(os.path.join(OUT, key + ".json"), "w"), indent=1, ensure_ascii=False)
        print(f"{key:36s} reconstructed from {r['from']}")


if __name__ == "__main__":
    main()
