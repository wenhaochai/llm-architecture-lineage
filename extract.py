#!/usr/bin/env python3
"""Extract normalized architecture metadata from every HF config.json in configs/.

Output: metadata.json (list of dicts, one per gallery card) with
  - raw structural numbers (layers, hidden, heads, kv heads, experts, ...)
  - derived categorical traits (attention class, MoE, SWA, hybrid mixer, MTP, ...)
  - provenance (which repo/file the config came from, and whether a trait is
    read directly from a config key or implied by the modeling class / model_type).
"""
import json, glob, os, re, collections
from key_taxonomy import DROP as NON_ARCH
import schema

ROOT = os.path.dirname(os.path.abspath(__file__))
cards = json.load(open(f"{ROOT}/data/cards.json"))


def first(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def text_cfg(c):
    for k in ("text_config", "language_config"):
        if isinstance(c.get(k), dict):
            return c[k]
    return c


# Traits that the modeling class fixes but the config does not spell out.
# key: model_type (of the text config or outer) -> dict of implied traits
IMPLIED = {
    "gemma3_text": {"qk_norm": True, "swa": True, "swa_pattern": 6, "norm": "rmsnorm", "act": "gelu"},
    "gemma3": {"qk_norm": True, "swa": True, "swa_pattern": 6, "norm": "rmsnorm", "act": "gelu"},
    "gemma4_text": {"qk_norm": True, "norm": "rmsnorm", "act": "gelu"},
    "gemma4_unified_text": {"qk_norm": True, "norm": "rmsnorm", "act": "gelu"},
    "olmo2": {"qk_norm": True, "norm": "rmsnorm", "post_norm": True},
    "olmo3": {"qk_norm": True, "norm": "rmsnorm", "post_norm": True},
    "qwen3": {"qk_norm": True},
    "qwen3_moe": {"qk_norm": True},
    "qwen3_next": {"qk_norm": True, "gated_attention": True, "hybrid": "gated_deltanet"},
    "qwen3_5_text": {"qk_norm": True, "gated_attention": True, "hybrid": "gated_deltanet"},
    "qwen3_5_moe_text": {"qk_norm": True, "gated_attention": True, "hybrid": "gated_deltanet"},
    "qwen4_exp_text": {"qk_norm": True, "gated_attention": True, "hybrid": "gated_deltanet", "sparse_attention": "QSA"},
    "kimi_linear": {"hybrid": "kda", "gated_attention": True, "shortconv": True},
    "kimi_k3": {"hybrid": "kda", "gated_attention": True, "shortconv": True},
    "solar_open2": {"hybrid": "kda", "gated_attention": True, "shortconv": True},
    "glm5_next_text": {"hybrid": "kda", "shortconv": True},
    "nemotron_h": {"hybrid": "mamba2", "nope": "full", "norm": "rmsnorm"},
    "granitemoehybrid": {"norm": "rmsnorm"},
    "lfm2": {"hybrid": "liv_conv", "norm": "rmsnorm"},
    "lfm2_moe": {"hybrid": "liv_conv", "norm": "rmsnorm"},
    "xlstm": {"hybrid": "mlstm", "recurrent_only": True, "norm": "rmsnorm", "nope": "full", "attention": "none (mLSTM)"},
    "cohere2": {"nope": "global", "norm": "layernorm", "swa": True},
    "cohere2_moe": {"nope": "global", "norm": "layernorm", "swa": True},
    "gpt_oss": {"attention_sinks": True, "swa": True, "act": "swiglu_clamped"},
    "gpt2": {"norm": "layernorm", "pos": "learned_absolute", "act": "gelu", "mha": True},
    "deepseek_v3": {"mla": True},
    "deepseek_v32": {"mla": True, "sparse_attention": "DSA"},
    "glm_moe_dsa": {"mla": True, "sparse_attention": "DSA"},
    "deepseek_v4": {"attention": "CSA/HCA", "sparse_attention": "DSA", "mhc": True},
    "deepseek_v41_text": {"attention": "CSA2", "sparse_attention": "DSA", "mhc": True, "encoder_decoder": True},
    "kimi_k2": {"mla": True},
    "bailing_hybrid": {"mla": True},
    "smollm3": {"nope": "partial"},
    "ouro": {"looped": True},
    "nanbeige": {"looped": True},
    "zaya": {"attention": "CCA"},
    "minimax_m3_vl": {"sparse_attention": "MSA", "qk_norm": True},
    "minimax_m2": {"qk_norm": True},
    "Motif": {"mla": True, "mhc": True, "act": "polynorm", "swa": True},
    "hy_v4": {"mla": True, "sparse_attention": "DSA", "gated_attention": True, "mhc": True},
    "hy_v3": {"qk_norm": True},
    "afmoe": {"gated_attention": True, "swa": True, "nope": "global", "qk_norm": True},
    "laguna": {"gated_attention": True, "qk_norm": True, "swa": True},
    "muse_glimmer_text": {"gated_attention": True, "qk_norm": True, "swa": True, "nope": "global"},
    "inkling_mm_model": {"qk_norm": True, "swa": True, "shortconv": True, "pos": "learned_relative_bias"},
    "step3p5": {"swa": True},
    "mimo_v2_flash": {"swa": True},
    "mimo_v2": {"swa": True},
    "mellum": {"swa": True},
    "sarvam_mla": {"mla": True, "nope": "partial"},
    "sarvam_moe": {"qk_norm": True},
    "mistral_large3": {"mla": True},
    "llama4_text": {"nope": "partial", "shared_experts": 1, "moe_interleave": 2},
    "phi3": {},
}


def count_layer_types(lt):
    return collections.Counter(lt) if isinstance(lt, list) else collections.Counter()


def extract(card):
    key = card["key"]
    path = f"{ROOT}/data/configs/{key}.json"
    c = json.load(open(path))
    tc = text_cfg(c)
    outer_mt = c.get("model_type")
    mt = tc.get("model_type") or outer_mt
    archs = c.get("architectures") or []
    imp = dict(IMPLIED.get(mt, {}))
    imp.update(IMPLIED.get(outer_mt, {}) if outer_mt != mt else {})

    m = {
        "key": key,
        "title": card["title"],
        "base_title": card["base"],
        "org": card["company"],
        "date": card["date"],
        "gallery": {  # kept only as an independent cross-check, never used to build edges
            "decoder": card["decoder"], "attention": card["attention"], "layer_mix": card["layer_mix"],
            "scale": card["scale"], "context": card["context"],
            "concepts": card["concepts"], "card_url": "https://sebastianraschka.com/llm-architecture-gallery/#card-" + key,
            "config_url": card.get("config_url"), "report_url": card.get("report_url"), "aai": card.get("aai"),
        },
        "config_source": card.get("config_url") or c.get("_provenance") or "",
        "provenance": c.get("_provenance") or c.get("_lineage_note") or "",
        "model_type": mt, "outer_model_type": outer_mt, "architecture_class": archs[0] if archs else None,
    }

    # ---- structural numbers -------------------------------------------------
    hidden = first(tc, "hidden_size", "n_embd", "dim", "embedding_dim", "block_dim")
    layers = first(tc, "num_hidden_layers", "n_layer", "num_layers", "num_blocks")
    if isinstance(tc.get("layers_block_type"), list):
        layers = len(tc["layers_block_type"])
    heads = first(tc, "num_attention_heads", "n_head", "num_heads")
    kv = first(tc, "num_key_value_heads", "n_kv_heads", "num_attention_groups")
    head_dim = first(tc, "head_dim", "qk_head_dim", "kv_channels")
    if head_dim is None and hidden and heads:
        head_dim = hidden // heads
    inter = first(tc, "intermediate_size", "ffn_hidden_size", "hidden_dim", "intermediate_size_mlp", "mlp_intermediate_size", "block_ff_dim")
    m.update(hidden_size=hidden, num_layers=layers, num_heads=heads, num_kv_heads=kv, head_dim=head_dim,
             intermediate_size=inter,
             vocab_size=first(tc, "vocab_size"), max_position_embeddings=first(tc, "max_position_embeddings", "n_positions", "max_seq_len"),
             rope_theta=first(tc, "rope_theta"), tie_embeddings=first(tc, "tie_word_embeddings", "tied_embeddings", "tie_embedding"),
             hidden_act=first(tc, "hidden_act", "hidden_activation", "activation_function", "mlp_hidden_act"),
             norm_eps=first(tc, "rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon", "norm_eps", "layernorm_epsilon"),
             attention_bias=first(tc, "attention_bias", "use_qkv_bias", "q_bias", "use_bias"),
             rope_scaling=(first(tc, "rope_scaling") or first(tc, "rope_parameters") or {}))
    if isinstance(m["rope_scaling"], dict):
        m["rope_scaling_type"] = m["rope_scaling"].get("rope_type") or m["rope_scaling"].get("type")
        if m["rope_theta"] is None:
            m["rope_theta"] = m["rope_scaling"].get("rope_theta")
    else:
        m["rope_scaling_type"] = None
    m["rope_scaling"] = None  # drop the blob

    lt = first(tc, "layer_types", "layers_block_type", "attn_type_list")
    lt_counts = count_layer_types(lt)

    # ---- MoE ---------------------------------------------------------------
    n_exp = first(tc, "n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts", "num_routed_experts")
    topk = first(tc, "num_experts_per_tok", "experts_per_token", "moe_top_k", "moe_topk", "top_k_experts", "experts_top_k", "num_experts_per_token")
    shared = first(tc, "n_shared_experts", "num_shared_experts")
    if shared is None and imp.get("shared_experts"):
        shared = imp["shared_experts"]
    if shared is None:
        sie = first(tc, "shared_expert_intermediate_size", "moe_shared_expert_intermediate_size", "share_expert_dim", "shared_intermediate_size")
        if sie:
            shared = 1
    moe_inter = first(tc, "moe_intermediate_size", "expert_intermediate_size", "expert_ffn_hidden_size", "expert_hidden_dim", "routed_expert_hidden_size")
    dense_prefix = first(tc, "first_k_dense_replace", "num_dense_layers", "n_dense_first_layers")
    is_moe = bool(n_exp and n_exp > 1)
    if mt == "lfm2" or mt == "granitemoehybrid" and not n_exp:
        is_moe = False
    m.update(is_moe=is_moe, num_experts=n_exp if is_moe else None, experts_per_tok=topk if is_moe else None,
             shared_experts=(shared or 0) if is_moe else None, moe_intermediate_size=moe_inter if is_moe else None,
             dense_prefix_layers=dense_prefix if is_moe else None,
             latent_moe=bool(first(tc, "moe_latent_size")),
             moe_layers=(len(tc["moe_layers"]) if isinstance(tc.get("moe_layers"), list) else (sum(1 for x in tc["moe_layer_freq"] if x) if isinstance(tc.get("moe_layer_freq"), list) else (lt_counts.get("moe") or (sum(1 for x in tc["mlp_layer_types"] if x == "sparse") if isinstance(tc.get("mlp_layer_types"), list) else None)))) if is_moe else None,
             zero_experts=first(tc, "zero_expert_num"),
             expert_groups=first(tc, "n_group", "num_expert_groups", "num_expert_group") if is_moe else None,
             router_scoring=first(tc, "scoring_func", "score_function", "score_func", "moe_router_activation_func", "moe_router_activation") if is_moe else None)

    # ---- attention class ----------------------------------------------------
    mla = bool(first(tc, "kv_lora_rank")) or imp.get("mla", False) or bool(first(tc, "use_mla"))
    attn = imp.get("attention")
    if not attn:
        if mla:
            attn = "MLA"
        elif imp.get("mha") or (kv is not None and heads is not None and kv == heads):
            attn = "MHA"
        elif kv == 1:
            attn = "MQA"
        elif kv is not None and heads is not None and kv < heads:
            attn = "GQA"
        else:
            attn = "unknown"
    m.update(attention_class=attn, mla=mla, q_lora_rank=first(tc, "q_lora_rank"), kv_lora_rank=first(tc, "kv_lora_rank"),
             qk_rope_head_dim=first(tc, "qk_rope_head_dim"), qk_nope_head_dim=first(tc, "qk_nope_head_dim"), v_head_dim=first(tc, "v_head_dim"))
    if heads and kv and not mla:
        m["gqa_ratio"] = heads / kv

    # ---- layer-type schedule -------------------------------------------------
    hop = first(tc, "hybrid_override_pattern")
    mix = {}
    if hop:
        mix = {"mamba2": hop.count("M"), "attention": hop.count("*"), "moe": hop.count("E"), "mlp": hop.count("-")}
    elif lt_counts:
        mix = dict(lt_counts)
    m["layer_schedule"] = mix

    # sliding window
    sw = first(tc, "sliding_window", "sliding_window_size")
    if isinstance(sw, int) and sw <= 0:
        sw = None
    swa_layers = lt_counts.get("sliding_attention", 0) + lt_counts.get("sliding_window", 0) + lt_counts.get("swa", 0) + lt_counts.get("local", 0)
    full_layers = lt_counts.get("full_attention", 0) + lt_counts.get("attention", 0) + lt_counts.get("global", 0) + lt_counts.get("full", 0)
    hlp = first(tc, "hybrid_layer_pattern")  # MiMo: 1 = sliding, 0 = full
    if isinstance(hlp, list) and set(hlp) <= {0, 1}:
        swa_layers, full_layers = hlp.count(1), hlp.count(0)
    lli = first(tc, "local_layer_ids")  # Inkling
    if isinstance(lli, list) and layers:
        swa_layers, full_layers = len(lli), layers - len(lli)
    int_pat = None
    for k in ("sliding_window_pattern", "_sliding_window_pattern", "layer_switch", "global_attn_every_n_layers", "global_attn_every_n", "sliding_window_period"):
        v = tc.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v > 1:
            int_pat = v
            break
    int_pat = int_pat or imp.get("swa_pattern")
    has_swa = bool(imp.get("swa")) or swa_layers > 0 or bool(first(tc, "use_sliding_window")) or bool(int_pat)
    if mt in ("qwen3", "qwen3_moe", "qwen2", "minimax_m2", "glm4_moe") and not first(tc, "use_sliding_window"):
        has_swa = False  # HF ships sliding_window keys on these but the released models do not use them
    if mt in ("mistral", "mistral4") and not sw:
        has_swa = False
    if mt == "git":  # Grok: sliding_window_size=-1, global_attn_every_n=1 -> every layer global
        has_swa = False
    if mt == "deepseek_v41_text" and sw:  # SWA-only layers + CSA2/SWA layers (compress_ratios)
        has_swa = True
        cr = first(tc, "compress_ratios") or []
        swa_layers, full_layers = sum(1 for x in cr if x == 0), sum(1 for x in cr if x != 0)
        int_pat = None
    if has_swa and not swa_layers and int_pat and layers:
        full_layers = layers // int_pat
        swa_layers = layers - full_layers
    swa_ratio = round(swa_layers / full_layers, 2) if (swa_layers and full_layers) else (int_pat - 1 if int_pat else None)
    if mt == 'deepseek_v41_text':
        swa_ratio = None  # SWA-only layers + CSA2/SWA layers; ratio not comparable
    m.update(swa=has_swa, sliding_window=sw if has_swa else None, swa_local_layers=swa_layers if has_swa and swa_layers else None,
             swa_global_layers=full_layers if has_swa and swa_layers else None, swa_ratio=swa_ratio if has_swa else None,
             chunked_attention=bool(first(tc, "attention_chunk_size")) and mt == "llama4_text")

    # hybrid sequence mixers
    hybrid = imp.get("hybrid")
    lin = lt_counts.get("linear_attention", 0) + lt_counts.get("kda", 0) + lt_counts.get("linear", 0) + lt_counts.get("conv", 0)
    if not hybrid:
        if lt_counts.get("linear_attention"):
            hybrid = "gated_deltanet"
        elif lt_counts.get("mamba"):
            hybrid = "mamba2"
        elif lt_counts.get("conv"):
            hybrid = "liv_conv"
        elif first(tc, "linear_attn_config"):
            hybrid = "kda"
    if mt == "bailing_hybrid":
        if archs and "V3" in archs[0]:
            hybrid = "kda"
        else:
            hybrid = "lightning"
    if mt == "granitemoehybrid" and not lt_counts.get("mamba"):
        hybrid = None
    m["hybrid_mixer"] = hybrid
    lin_n, full_n = None, None
    lac = first(tc, "linear_attn_config")
    if hybrid:
        if isinstance(lac, dict) and isinstance(lac.get("kda_layers"), list):
            lin_n, full_n = len(lac["kda_layers"]), len(lac.get("full_attn_layers") or [])
        elif isinstance(first(tc, "gqa_layers"), list) and layers:
            full_n = len(tc["gqa_layers"]); lin_n = layers - full_n
        elif first(tc, "layer_group_size") and layers:
            full_n = layers // tc["layer_group_size"]; lin_n = layers - full_n
        elif lt_counts.get("linear_attention"):
            lin_n = lt_counts["linear_attention"]; full_n = layers - lin_n if layers else None
        elif first(tc, "full_attention_interval") and layers:
            full_n = layers // tc["full_attention_interval"]; lin_n = layers - full_n
        elif hop:
            lin_n, full_n = mix.get("mamba2"), mix.get("attention")
        elif lt_counts.get("mamba"):
            lin_n, full_n = lt_counts["mamba"], lt_counts.get("attention", 0)
        elif lt_counts.get("conv"):
            lin_n, full_n = lt_counts["conv"], lt_counts.get("full_attention", 0)
        elif m["recurrent_only"] if "recurrent_only" in m else imp.get("recurrent_only"):
            lin_n, full_n = layers, 0
    m["linear_layers"] = lin_n
    m["full_attention_layers_in_hybrid"] = full_n
    m["hybrid_ratio"] = round(lin_n / full_n, 2) if (lin_n and full_n) else None
    m["recurrent_only"] = bool(imp.get("recurrent_only"))

    # sparse attention / indexers
    sparse = imp.get("sparse_attention")
    if not sparse and (first(tc, "index_topk") or first(tc, "indexer_budget") or first(tc, "sparse_attention_config") or first(tc, "use_dsa")):
        sparse = "DSA"
    m["sparse_attention"] = sparse
    m["index_topk"] = first(tc, "index_topk", "indexer_budget")
    m["index_share"] = bool(first(tc, "index_source_layer_ids", "index_topk_pattern", "indexer_types")) or bool(first(tc, "index_share_for_mtp_iteration"))
    # cross-layer KV sharing
    m["kv_sharing"] = bool(first(tc, "num_kv_shared_layers", "kv_source_layer_ids"))
    m["kv_shared_layers"] = first(tc, "num_kv_shared_layers")
    # per-layer embeddings
    m["per_layer_embeddings"] = bool(first(tc, "hidden_size_per_layer_input", "ple_embed_dim"))
    # mHC / hyper-connections / attention residuals
    m["mhc"] = bool(imp.get("mhc")) or bool(first(tc, "mhc", "hc_mult", "mhc_enabled", "enable_ihc"))
    m["attention_residuals"] = bool(first(tc, "attn_res_block_size"))
    m["hc_streams"] = first(tc, "hc_mult", "hc_count", "mhc_expansion_rate")
    # MTP
    mtp = first(tc, "num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_modules", "mtp_transformer_layers", "mtp_layers_block_type")
    if isinstance(mtp, list):
        mtp = len(mtp)
    if mtp is None and (first(tc, "use_mtp") or isinstance(first(tc, "mtp"), dict)):
        mtp = 1
    m["mtp_layers"] = mtp or 0
    m["mtp"] = bool(mtp)
    # QK norm
    qk = first(tc, "use_qk_norm", "qk_norm", "add_qk_norm")
    qkt = first(tc, "qk_norm_type")
    m["qk_norm"] = bool(imp.get("qk_norm")) or bool(qk) or (qkt not in (None, "none", False))
    # gated attention
    ga = first(tc, "attn_output_gate", "attention_output_gate", "use_gqa_gate", "use_head_wise_attn_gate", "gated_mla", "mla_use_output_gate", "headwise_attn_output_gate", "elementwise_attn_output_gate")
    ogt = first(tc, "output_gate_type", "gating_type")
    m["gated_attention"] = bool(imp.get("gated_attention")) or bool(ga) or (ogt not in (None, "none", False))
    # positional encoding
    nope = imp.get("nope")
    if first(tc, "no_rope_layers") or first(tc, "no_rope_layer_interval") or first(tc, "use_rope_layers") is not None and any(not x for x in (first(tc, "use_rope_layers") or [True])):
        nope = nope or "partial"
    if first(tc, "use_rope") is False or first(tc, "use_pos_enc") is False or first(tc, "position_embedding_type") in ("none", "nope") or first(tc, "mla_use_nope") or first(tc, "use_mla_nope"):
        nope = nope or "partial" if mla else (nope or "full")
    pos = imp.get("pos") or ("rope" if (m["rope_theta"] or first(tc, "rope_scaling") or first(tc, "rope_parameters") or mla) else ("none" if nope == "full" else "rope"))
    if nope == "full":
        pos = "none"
    m["position_encoding"] = pos
    m["nope"] = nope or "none"
    prf = first(tc, "partial_rotary_factor", "rotary_pct")
    m["partial_rotary"] = (prf is not None and prf not in (1, 1.0)) or bool(first(tc, "rotary_dim")) or bool(first(tc, "partial_rotary_factors"))
    # misc
    m["attention_sinks"] = bool(imp.get("attention_sinks")) or bool(first(tc, "learnable_sink", "add_full_attention_sink_bias", "add_swa_attention_sink_bias", "sink", "shared_expert_sink"))
    m["looped"] = bool(imp.get("looped")) or bool(first(tc, "num_loops", "total_ut_steps"))
    m["loop_passes"] = first(tc, "num_loops", "total_ut_steps")
    m["shortconv"] = bool(imp.get("shortconv")) or bool(first(tc, "use_sconv", "short_conv_kernel_size"))
    m["logit_softcapping"] = bool(first(tc, "final_logit_softcapping", "output_logit_soft_cap"))
    m["attn_softcapping"] = bool(first(tc, "attn_logit_softcapping"))
    m["norm_type"] = imp.get("norm") or ("layernorm" if first(tc, "layer_norm_eps", "layer_norm_epsilon") and not first(tc, "rms_norm_eps") else "rmsnorm")
    m["post_norm"] = bool(imp.get("post_norm"))
    m["encoder_decoder"] = bool(imp.get("encoder_decoder")) or bool(first(tc, "is_encoder_decoder")) and mt != "gpt2"
    m["activation"] = imp.get("act") or (m["hidden_act"] or "").lower() or None
    m["multimodal"] = tc is not c  # text config nested inside a multimodal wrapper

    # ---- more normalised fields (kept for the metadata table) ----------------------
    rs = first(tc, "rope_scaling") or first(tc, "rope_parameters") or {}
    m["rope_scaling_factor"] = rs.get("factor") if isinstance(rs, dict) else None
    m["original_max_position_embeddings"] = rs.get("original_max_position_embeddings") if isinstance(rs, dict) else None
    m["partial_rotary_factor"] = first(tc, "partial_rotary_factor", "rotary_pct")
    m["rotary_dim"] = first(tc, "rotary_dim")
    m["attention_dropout"] = first(tc, "attention_dropout")
    m["initializer_range"] = first(tc, "initializer_range")
    m["mlp_bias"] = first(tc, "mlp_bias")
    m["moe_layer_freq"] = first(tc, "moe_layer_freq") if not isinstance(first(tc, "moe_layer_freq"), list) else None
    m["routed_scaling_factor"] = first(tc, "routed_scaling_factor", "route_scale", "moe_router_scaling_factor", "router_scaling_factor")
    m["norm_topk_prob"] = first(tc, "norm_topk_prob", "moe_renormalize", "norm_expert_weight")
    m["topk_method"] = first(tc, "topk_method")
    m["topk_group"] = first(tc, "topk_group", "num_expert_groups_per_tok")
    m["router_aux_loss_coef"] = first(tc, "router_aux_loss_coef", "aux_loss_alpha", "load_balance_coeff")
    m["shared_expert_intermediate_size"] = first(tc, "shared_expert_intermediate_size", "moe_shared_expert_intermediate_size", "share_expert_dim", "shared_intermediate_size")
    m["moe_latent_size"] = first(tc, "moe_latent_size")
    m["index_n_heads"] = first(tc, "index_n_heads", "indexer_n_heads")
    m["index_head_dim"] = first(tc, "index_head_dim", "indexer_head_dim")
    m["compress_ratios"] = sorted(set(first(tc, "compress_ratios") or [])) or None
    m["mamba_num_heads"] = first(tc, "mamba_num_heads", "mamba_n_heads")
    m["mamba_head_dim"] = first(tc, "mamba_head_dim", "mamba_d_head")
    m["ssm_state_size"] = first(tc, "ssm_state_size", "mamba_d_state")
    m["mamba_n_groups"] = first(tc, "n_groups", "mamba_n_groups")
    m["conv_kernel"] = first(tc, "conv_kernel", "mamba_d_conv", "linear_conv_kernel_dim", "short_conv_kernel_size", "conv_L_cache", "sconv_kernel_size")
    m["linear_num_key_heads"] = first(tc, "linear_num_key_heads")
    m["linear_num_value_heads"] = first(tc, "linear_num_value_heads")
    m["linear_key_head_dim"] = first(tc, "linear_key_head_dim")
    m["linear_value_head_dim"] = first(tc, "linear_value_head_dim")
    lac = first(tc, "linear_attn_config")
    if isinstance(lac, dict):
        m["linear_num_value_heads"] = m["linear_num_value_heads"] or lac.get("num_heads")
        m["linear_value_head_dim"] = m["linear_value_head_dim"] or lac.get("head_dim")
    m["kv_shared_layers"] = first(tc, "num_kv_shared_layers") or (len(first(tc, "kv_source_layer_ids") or []) or None)
    m["hidden_size_per_layer_input"] = first(tc, "hidden_size_per_layer_input", "ple_embed_dim")
    m["final_logit_softcapping"] = first(tc, "final_logit_softcapping", "output_logit_soft_cap")
    m["attn_logit_softcapping"] = first(tc, "attn_logit_softcapping")
    m["query_pre_attn_scalar"] = first(tc, "query_pre_attn_scalar")
    m["swiglu_limit"] = first(tc, "swiglu_limit")
    m["dtype"] = first(tc, "torch_dtype", "dtype") or first(c, "torch_dtype", "dtype")
    m["transformers_version"] = first(tc, "transformers_version") or first(c, "transformers_version")
    m["bos_token_id"] = first(tc, "bos_token_id"); m["eos_token_id"] = first(tc, "eos_token_id"); m["pad_token_id"] = first(tc, "pad_token_id")
    m["num_config_keys"] = len(tc)

    # ---- the complete text config, long lists summarised as value counts -------------
    def compact(v):
        if isinstance(v, list) and len(v) > 12 and all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
            cnt = collections.Counter(str(x) for x in v)
            return {"_list_len": len(v), "_counts": dict(cnt.most_common())}
        if isinstance(v, dict):
            return {k: compact(x) for k, x in v.items()}
        if isinstance(v, list):
            return [compact(x) for x in v]
        return v
    def quant_summary(q):
        if not isinstance(q, dict):
            return q
        bits = None
        for g in (q.get("config_groups") or {}).values():
            if isinstance(g, dict) and isinstance(g.get("weights"), dict):
                bits = g["weights"].get("num_bits")
        return {"quant_method": q.get("quant_method") or q.get("format"), "weight_bits": bits or q.get("bits"), "_note": "checkpoint quantization, not architecture; details omitted"}
    DROP = ("architectures", "auto_map", "transformers.js_config", "chat_template", "processor_config")
    m["config_full"] = {k: (quant_summary(v) if k == "quantization_config" else compact(v)) for k, v in tc.items() if k not in DROP}
    if tc is not c:
        m["config_wrapper"] = {k: compact(v) for k, v in c.items() if not isinstance(v, dict) or k in ("vision_config", "audio_config")}
        m["config_wrapper"] = {k: ("<omitted sub-config>" if isinstance(v, dict) and k in ("vision_config", "audio_config") else v) for k, v in m["config_wrapper"].items()}
    # ---- canonical config: every architecture key renamed through schema.ALIASES ------
    norm = collections.OrderedDict()
    raw_used = {}
    for k, v in tc.items():
        if k in NON_ARCH:
            continue
        canon = schema.RAW_TO_CANONICAL.get(k)
        if canon is None:
            raise KeyError(f"{key}: architecture key {k!r} missing from schema.ALIASES")
        if canon in norm:  # several spellings present in one config -> keep each under its raw name
            if not isinstance(norm[canon], dict) or "_raw" not in norm[canon]:
                norm[canon] = {"_raw": {raw_used[canon]: norm[canon]}}
            norm[canon]["_raw"][k] = compact(v)
        else:
            norm[canon] = compact(v)
            raw_used[canon] = k
    # ---- nested configs and layer-index lists -> flat fields and kind-count schedules ----
    def schedule(counts, total):
        return {"_list_len": total, "_counts": {k: v for k, v in counts.items() if v}}
    def is_index_list(v):
        return isinstance(v, dict) and v.get("_list_len") and all(c == 1 for c in v["_counts"].values()) and all(str(k).isdigit() for k in v["_counts"])
    def put(canon, value, source):
        if canon not in norm and value not in (None, "", [], {}):
            norm[canon] = value; raw_used[canon] = source
    lac = tc.get("linear_attn_config")
    if isinstance(lac, dict):
        norm.pop("linear_attention_config", None); raw_used.pop("linear_attention_config", None)
        if isinstance(lac.get("kda_layers"), list) or isinstance(lac.get("full_attn_layers"), list):
            n_kda, n_full = len(lac.get("kda_layers") or []), len(lac.get("full_attn_layers") or [])
            norm["layer_types"] = schedule({"kda": n_kda, "full_attention": n_full}, n_kda + n_full); raw_used["layer_types"] = "linear_attn_config.kda_layers / full_attn_layers"
        put("linear_num_value_heads", lac.get("num_heads"), "linear_attn_config.num_heads")
        put("linear_num_key_heads", lac.get("num_kv_heads"), "linear_attn_config.num_kv_heads")
        put("linear_value_head_dim", lac.get("head_dim"), "linear_attn_config.head_dim")
        put("short_conv_kernel", lac.get("short_conv_kernel_size"), "linear_attn_config.short_conv_kernel_size")
        extra = {k: v for k, v in lac.items() if k not in ("kda_layers", "full_attn_layers", "num_heads", "num_kv_heads", "head_dim", "short_conv_kernel_size")}
        if extra:
            if "kda_config" in norm:
                cur = norm["kda_config"]; cur = cur["_raw"] if isinstance(cur, dict) and "_raw" in cur else {raw_used["kda_config"]: cur}
                cur.update(extra); norm["kda_config"] = {"_raw": cur}
            else:
                norm["kda_config"] = {"_raw": extra}; raw_used["kda_config"] = "linear_attn_config"
    aos = tc.get("attention_other_setting")
    if isinstance(aos, dict):
        norm.pop("swa_attention_config", None); raw_used.pop("swa_attention_config", None)
        put("swa_num_heads", aos.get("num_attention_heads"), "attention_other_setting.num_attention_heads")
        put("swa_num_kv_heads", aos.get("num_attention_groups"), "attention_other_setting.num_attention_groups")
        put("swa_head_dim", aos.get("head_dim"), "attention_other_setting.head_dim")
    sac = tc.get("sparse_attention_config")
    if isinstance(sac, dict):
        norm["sparse_attention"] = m["sparse_attention"] or "sparse"; raw_used["sparse_attention"] = "sparse_attention_config"
        if sac.get("sparse_topk_blocks") and sac.get("sparse_block_size"):
            put("index_topk", sac["sparse_topk_blocks"] * sac["sparse_block_size"], "sparse_attention_config.sparse_topk_blocks × sparse_block_size")
        put("index_num_heads", sac.get("sparse_num_index_heads"), "sparse_attention_config.sparse_num_index_heads")
        put("index_head_dim", sac.get("sparse_index_dim"), "sparse_attention_config.sparse_index_dim")
        put("index_layer_types", sac.get("sparse_attention_freq"), "sparse_attention_config.sparse_attention_freq")
        rest = {k: v for k, v in sac.items() if k not in ("sparse_topk_blocks", "sparse_block_size", "sparse_num_index_heads", "sparse_index_dim", "sparse_attention_freq", "use_sparse_attention")}
        if rest:
            put("candidate_selection", {"_raw": rest}, "sparse_attention_config")
    rs = norm.get("rope_scaling")
    if isinstance(rs, dict) and "_raw" not in rs:
        src = raw_used.get("rope_scaling", "rope_scaling")
        rs = dict(rs)
        put("rope_scaling_type", rs.pop("rope_type", None) or rs.pop("type", None), src + ".rope_type"); rs.pop("type", None)
        put("rope_scaling_factor", rs.pop("factor", None), src + ".factor")
        put("rope_original_max_position", rs.pop("original_max_position_embeddings", None), src + ".original_max_position_embeddings")
        put("rope_theta", rs.pop("rope_theta", None), src + ".rope_theta")
        put("partial_rotary_factor", rs.pop("partial_rotary_factor", None), src + ".partial_rotary_factor")
        for lt_key, canon in (("full_attention", "rope_theta"), ("hybrid", "rope_theta"), ("sliding_attention", "rope_theta_local"), ("hybrid_sliding", "rope_theta_local")):
            sub = rs.pop(lt_key, None)
            if isinstance(sub, dict):
                put(canon, sub.get("rope_theta"), src + "." + lt_key + ".rope_theta")
                put("partial_rotary_factor", sub.get("partial_rotary_factor"), src + "." + lt_key + ".partial_rotary_factor")
                put("rope_scaling_type", sub.get("rope_type"), src + "." + lt_key + ".rope_type")
        if rs:
            if "rope_scaling_params" in norm:
                cur = norm["rope_scaling_params"]; cur = cur["_raw"] if isinstance(cur, dict) and "_raw" in cur else {raw_used["rope_scaling_params"]: cur}
                cur.update(rs); norm["rope_scaling_params"] = {"_raw": cur}
            else:
                norm["rope_scaling_params"] = {"_raw": rs}; raw_used["rope_scaling_params"] = src
        norm.pop("rope_scaling", None); raw_used.pop("rope_scaling", None)
    # layer_types: always a kind-count schedule with readable kind names
    lt_raw = raw_used.get("layer_types")
    ltv = norm.get("layer_types")
    if lt_raw == "hybrid_override_pattern" or lt_raw == "layers_block_type":
        mix = m["layer_schedule"]
        norm["layer_types"] = schedule({"mamba2": mix.get("mamba2", mix.get("mamba", 0)), "attention": mix.get("attention", 0), "moe": mix.get("moe", 0), "mlp": mix.get("mlp", 0)}, sum(v for v in mix.values() if isinstance(v, int)))
    elif lt_raw == "hybrid_layer_pattern" and isinstance(ltv, dict):
        c = ltv["_counts"]; norm["layer_types"] = schedule({"sliding_attention": c.get("1", 0), "full_attention": c.get("0", 0)}, ltv["_list_len"])
    elif lt_raw == "local_layer_ids" and is_index_list(ltv) and layers:
        norm["layer_types"] = schedule({"sliding_attention": ltv["_list_len"], "full_attention": layers - ltv["_list_len"]}, layers)
    if "layer_types" not in norm and isinstance(tc.get("gqa_layers"), list) and layers:
        n_full = len(tc["gqa_layers"]); norm["layer_types"] = schedule({"full_attention": n_full, "kda": layers - n_full}, layers); raw_used["layer_types"] = "gqa_layers"
    if raw_used.get("hybrid_attention_interval") == "gqa_layers":
        norm["hybrid_attention_interval"] = tc.get("gqa_interval", norm["hybrid_attention_interval"]); raw_used["hybrid_attention_interval"] = "gqa_interval"
    # remaining index lists: keep the count and the range instead of one bucket per index
    for k, v in list(norm.items()):
        if is_index_list(v):
            idx = sorted(int(i) for i in v["_counts"])
            step = idx[1] - idx[0] if len(idx) > 1 and all(b - a == idx[1] - idx[0] for a, b in zip(idx, idx[1:])) else None
            norm[k] = {"_indices": len(idx), "_min": idx[0], "_max": idx[-1], "_step": step}

    # traits the modeling class implies (QK-Norm in Gemma 3 / OLMo 2, gated attention in Qwen3-Next, ...) and
    # ratios that only exist as counts in the raw config become canonical fields too, so two configs are
    # compared on what the model computes rather than on which keys its family happened to spell out
    def fill(canon, value, source):
        if canon not in norm and value not in (None, False, "", 0, "none"):
            norm[canon] = value
            raw_used[canon] = source
    fill("attention_kind", m["attention_class"], "derived")
    fill("sequence_mixer", m["hybrid_mixer"] or "attention", "derived")
    if m["hybrid_ratio"]:
        fill("mixer_attention_ratio", round(m["hybrid_ratio"]), "derived")
    if m["swa"] and m["swa_ratio"]:
        fill("local_global_ratio", round(m["swa_ratio"]), "derived")
    imp_src = f"implied by model_type {mt}"
    fill("qk_norm", m["qk_norm"], imp_src)
    fill("gated_attention", m["gated_attention"], imp_src)
    fill("sparse_attention", m["sparse_attention"], imp_src)
    fill("mla", m["mla"], imp_src)
    fill("attention_sinks", m["attention_sinks"], imp_src)
    fill("hyper_connections", m["mhc"], imp_src)
    fill("attention_residuals", m["attention_residuals"], "derived")
    fill("moe", m["is_moe"], "derived")
    fill("norm_type", m["norm_type"], imp_src)
    fill("position_encoding_type", m["position_encoding"], imp_src)
    if m["nope"] != "none":
        fill("nope_layers", m["nope"], imp_src)
    if m["looped"]:
        fill("loop_passes", m["loop_passes"] or True, imp_src)
    if m["kv_sharing"]:
        fill("kv_shared_layers", m["kv_shared_layers"] or True, imp_src)
    if m["per_layer_embeddings"]:
        fill("per_layer_embedding_dim", m["hidden_size_per_layer_input"] or True, imp_src)
    if m["encoder_decoder"]:
        fill("bidirectional_attention", True, imp_src)
    # a named gate type means the attention is gated; a disabled or non-positive window means no window
    if norm.get("gated_attention_type") and "gated_attention" not in norm:
        norm["gated_attention"] = True; raw_used["gated_attention"] = raw_used["gated_attention_type"]
    if isinstance(norm.get("sliding_window"), (int, float)) and norm["sliding_window"] <= 0:
        del norm["sliding_window"]; raw_used.pop("sliding_window", None)
    if norm.get("sliding_window_enabled") is False or (not m["swa"] and mt in ("qwen3", "qwen3_moe", "qwen2", "minimax_m2", "glm4_moe", "mistral", "mistral4", "git")):
        for k in ("sliding_window", "sliding_window_enabled", "sliding_window_max_layers"):
            norm.pop(k, None); raw_used.pop(k, None)
    if not m["hybrid_mixer"]:  # leftover Mamba defaults in a pure-attention config (Antares) describe no layer
        for k in ("mamba_num_heads", "mamba_head_dim", "mamba_state_size", "mamba_num_groups", "mamba_expand", "mamba_activation", "mamba_proj_bias", "mamba_time_step"):
            norm.pop(k, None); raw_used.pop(k, None)
        for k in ("short_conv_kernel", "conv_bias"):
            if str(raw_used.get(k, "")).startswith("mamba"):
                norm.pop(k, None); raw_used.pop(k, None)
    # bare switches (use_rmsnorm=True, use_pos_enc=True, use_dsa=True) name no design; give them the
    # value the rest of the gallery uses for the same thing, so spellings never read as changes
    if norm.get("sparse_attention") is True:
        norm["sparse_attention"] = m["sparse_attention"] or "sparse"
    if norm.get("norm_type") in (True, "layer_norm", "rms_norm", "RMSNorm", "LayerNorm"):
        norm["norm_type"] = m["norm_type"]
    if norm.get("position_encoding_type") in (True, False, "rope_gptj"):
        norm["position_encoding_type"] = m["position_encoding"]
    if norm.get("router_scoring") is True:
        norm["router_scoring"] = "sigmoid"
    if norm.get("router_topk_method") is True:
        norm["router_topk_method"] = "grouped"
    if norm.get("router_topk_method") in ("sigmoid", "softmax"):  # expert_selection_fn names the scoring, not the top-k rule
        if "router_scoring" not in norm:
            norm["router_scoring"] = norm["router_topk_method"]; raw_used["router_scoring"] = raw_used["router_topk_method"]
        del norm["router_topk_method"]; raw_used.pop("router_topk_method", None)
    if norm.get("bidirectional_attention") == "vision":  # bidirectional only over image tokens: not a text-decoder design
        del norm["bidirectional_attention"]; raw_used.pop("bidirectional_attention", None)
    if isinstance(norm.get("parallel_block"), str):
        norm["parallel_block"] = "parallel" in norm["parallel_block"].lower()
    if norm.get("gated_attention_type") is True:
        del norm["gated_attention_type"]; raw_used.pop("gated_attention_type", None)
    # null and empty values carry no information; several spellings with one value collapse to that value
    for k in list(norm):
        v = norm[k]
        if v is None or v == [] or v == {} or v == "":
            del norm[k]; raw_used.pop(k, None); continue
        if isinstance(v, dict) and "_raw" in v:
            v["_raw"] = {rk: rv for rk, rv in v["_raw"].items() if rv not in (None, [], {}, "")}
            if not v["_raw"]:
                del norm[k]; raw_used.pop(k, None); continue
            if len(v["_raw"]) == 1:
                (rk, rv), = v["_raw"].items(); norm[k] = rv; raw_used[k] = rk; continue
            vals = list(v["_raw"].values())
            if all(json.dumps(x, sort_keys=True) == json.dumps(vals[0], sort_keys=True) for x in vals):
                norm[k] = vals[0]; raw_used[k] = " / ".join(v["_raw"])
    m["config_canonical"] = dict(norm)
    m["config_canonical_raw_key"] = raw_used
    m["num_architecture_keys"] = sum(1 for k in tc if k not in NON_ARCH)
    return m


out = [extract(cd) for cd in cards]
json.dump(out, open(f"{ROOT}/data/metadata.json", "w"), indent=1, ensure_ascii=False)

# ---- compact table for eyeballing against the gallery's own descriptions ----
hdr = ["key", "mt", "attn", "moe", "E/k", "swa", "ratio", "hybrid", "hratio", "sparse", "mtp", "qk", "gate", "nope", "kvsh", "mhc", "L", "d", "H/KV", "| gallery attention / layer mix"]
print("\t".join(hdr))
for m in sorted(out, key=lambda x: x["date"]):
    print("\t".join(str(x) for x in [
        m["key"][:26], (m["model_type"] or "")[:14], m["attention_class"], "M" if m["is_moe"] else "-",
        f"{m['num_experts']}/{m['experts_per_tok']}+{m['shared_experts']}" if m["is_moe"] else "",
        "S" if m["swa"] else "-", m["swa_ratio"] or "", m["hybrid_mixer"] or "", m["hybrid_ratio"] or "", m["sparse_attention"] or "",
        "T" if m["mtp"] else "-", "Q" if m["qk_norm"] else "-", "G" if m["gated_attention"] else "-", m["nope"], "K" if m["kv_sharing"] else "-", "H" if m["mhc"] else "-",
        m["num_layers"], m["hidden_size"], f"{m['num_heads']}/{m['num_kv_heads']}",
        "| " + str(m["gallery"]["attention"])[:50] + " / " + str(m["gallery"]["layer_mix"])[:40]]))
