"""Canonical names for the architecture keys that survive key_taxonomy.DROP.

Every raw key is listed exactly once. Keys that name the same quantity under
different spellings collapse into one canonical field. Family-specific knobs keep
a descriptive canonical name of their own; the GROUPS table says which section of
the architecture each canonical field belongs to.
"""

ALIASES = {
    # ---- derived (no raw key of their own; filled by extract.py from the modeling class or from ratios) ----
    "attention_kind": [],          # MHA / GQA / MQA / MLA / CSA ... from head counts and latent ranks
    "local_global_ratio": [],      # sliding-window : global layers, rounded
    "sequence_mixer": [],          # attention | gated_deltanet | kda | mamba2 | lightning | liv_conv | mlstm
    "mixer_attention_ratio": [],   # mixer : attention layers in a hybrid, rounded
    # ---- shape ------------------------------------------------------------------
    "hidden_size": ["hidden_size", "n_embd", "embedding_dim", "block_dim"],
    "num_layers": ["num_hidden_layers", "n_layer", "num_layers", "num_blocks"],
    "intermediate_size": ["intermediate_size", "ffn_hidden_size", "intermediate_size_mlp", "mlp_intermediate_size", "block_ff_dim", "dense_intermediate_size"],
    "dense_prefix_intermediate_size": ["prefix_dense_intermediate_size"],
    "ffn_width_multiplier": ["ffn_proj_factor", "block_ffn_dim_multiplier"],
    "ffn_round_to_multiple": ["ffn_round_up_to_multiple_of", "block_multiple_of", "block_auto_adjust_ff_dim", "mlstm_round_up_to_multiple_of"],
    "vocab_size": ["vocab_size"],
    "vocab_size_unpadded": ["unpadded_vocab_size"],
    "max_position_embeddings": ["max_position_embeddings", "n_positions", "n_ctx", "max_seq_len", "model_max_length"],
    "tie_embeddings": ["tie_word_embeddings", "tie_embedding", "use_embedding_sharing"],
    "lm_head_bias": ["lm_head_bias"],
    "mlp_bias": ["mlp_bias"],
    "parallel_block": ["use_parallel_block", "transformer_block_type"],
    # ---- normalisation & activation ---------------------------------------------
    "norm_eps": ["rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon", "norm_eps", "layernorm_epsilon", "eps", "block_norm_eps"],
    "norm_type": ["norm_type", "normalization_function", "use_rmsnorm", "use_gemma_norm"],
    "norm_eps_post": ["post_norm_eps"],
    "norm_eps_cell": ["cell_norm_eps"],
    "norm_extra_placement": ["add_post_norm", "add_out_norm", "add_post_blocks_norm", "use_embed_norm", "up_proj_norm", "value_norm", "skip_loop_final_norm"],
    "norm_beta_attention": ["layernorm_full_attention_beta"],
    "norm_beta_linear_attention": ["layernorm_linear_attention_beta"],
    "norm_beta_mlp": ["layernorm_mlp_beta"],
    "norm_zero_centered": ["zero_centered"],
    "activation": ["hidden_act", "hidden_activation", "activation_function", "mlp_hidden_act"],
    "activation_gated": ["use_gated_activation", "block_use_swiglu"],
    "activation_clamp": ["swiglu_limit", "swiglu_limits", "swiglu_alpha", "hidden_clamp"],
    "activation_clamp_shared_expert": ["swiglu_limits_shared", "share_expert_swiglu_limit_list"],
    "activation_clamp_experts": ["expert_swiglu_limit_list"],
    "activation_situ_beta": ["activation_situ_beta", "activation_situ_linear_beta"],
    "polynorm_output_scale": ["polynorm_output_scale", "polynorm_output_scale_per_layer"],
    "polynorm_bias_clamp": ["polynorm_bias_clamp"],
    "ngpt": ["use_nGPT"],
    # ---- attention: heads & projections -----------------------------------------
    "num_heads": ["num_attention_heads", "n_head", "num_heads"],
    "num_kv_heads": ["num_key_value_heads", "num_attention_groups"],
    "head_dim": ["head_dim", "qk_head_dim", "kv_channels", "q_head_dim"],
    "v_head_dim": ["v_head_dim"],
    "attention_bias": ["attention_bias", "use_qkv_bias", "use_bias", "q_bias", "o_bias"],
    "attention_impl_type": ["att_impl_type", "attention_cls"],
    "attention_scale": ["query_pre_attn_scalar", "attn_scale", "qk_scale_factor", "attention_multiplier"],
    "attention_value_scale": ["attention_value_scale"],
    "attention_logit_softcapping": ["attn_logit_softcapping"],
    "attention_temperature_tuning": ["attn_temperature_tuning", "attn_temperature_len", "floor_scale"],
    "attention_log_scaling": ["log_scaling_alpha", "log_scaling_n_floor"],
    "attention_heads_per_layer": ["num_attention_heads_per_layer"],
    "attention_differential": ["diff_v2", "k_ratio", "num_noise_heads"],
    "attention_cca_steps": ["cca_time0", "cca_time1"],
    # ---- attention: gating & QK-Norm ---------------------------------------------
    "gated_attention": ["attn_output_gate", "attention_output_gate", "use_gqa_gate", "use_head_wise_attn_gate", "headwise_attn_output_gate", "elementwise_attn_output_gate", "gated_mla", "mla_use_output_gate"],
    "gated_attention_type": ["output_gate_type", "gating", "gating_type", "gating_types", "gated_attention_proj_granularity_type"],
    "qk_norm": ["use_qk_norm", "qk_norm", "add_qk_norm"],
    "qk_norm_type": ["qk_norm_type"],
    # ---- attention: latent (MLA family) -------------------------------------------
    "mla": ["use_mla"],
    "q_lora_rank": ["q_lora_rank"],
    "kv_lora_rank": ["kv_lora_rank"],
    "o_lora_rank": ["o_lora_rank"],
    "o_groups": ["o_groups"],
    "qk_rope_head_dim": ["qk_rope_head_dim"],
    "qk_nope_head_dim": ["qk_nope_head_dim"],
    "mla_scale_lora": ["mla_scale_q_lora", "mla_scale_kv_lora"],
    "mla_nope": ["mla_use_nope", "use_mla_nope"],
    # ---- attention: sliding window / layer schedule -------------------------------
    "layer_types": ["layer_types", "attn_type_list", "hybrid_layer_pattern", "local_layer_ids", "order_of_interleaved_layers", "layers_block_type", "hybrid_override_pattern"],
    "sliding_window": ["sliding_window", "sliding_window_size"],
    "sliding_window_enabled": ["use_sliding_window"],
    "sliding_window_pattern": ["sliding_window_pattern", "_sliding_window_pattern", "layer_switch", "sliding_window_period", "global_attn_every_n", "global_attn_every_n_layers", "prefix_dense_sliding_window_pattern"],
    "sliding_window_max_layers": ["max_window_layers"],
    "chunked_attention_size": ["attention_chunk_size"],
    "swa_num_heads": ["swa_num_attention_heads"],
    "swa_num_kv_heads": ["swa_num_key_value_heads"],
    "swa_head_dim": ["swa_head_dim", "swa_v_head_dim"],
    "swa_rope_theta": ["swa_rope_theta"],
    "swa_attention_config": ["attention_other_setting"],
    "global_head_dim": ["global_head_dim"],
    "global_num_kv_heads": ["num_global_key_value_heads"],
    "global_kv_unified": ["attention_k_eq_v"],
    "kv_shared_layers": ["num_kv_shared_layers", "kv_source_layer_ids"],
    "attention_sinks": ["sink", "learnable_sink", "add_full_attention_sink_bias", "add_swa_attention_sink_bias"],
    "bidirectional_attention": ["use_bidirectional_attention"],
    # ---- attention: sparse / indexer ----------------------------------------------
    "sparse_attention": ["use_dsa", "sparse_attention_config"],
    "index_topk": ["index_topk", "indexer_budget"],
    "index_num_heads": ["index_n_heads", "indexer_n_heads"],
    "index_head_dim": ["index_head_dim", "indexer_head_dim"],
    "index_kv_heads": ["indexer_kv_heads"],
    "index_layer_types": ["indexer_types", "index_topk_pattern", "index_topk_freq", "index_skip_topk_offset"],
    "index_kpool": ["index_kpool", "index_kpool_always_select_tail", "index_kpool_compress", "indexer_compress_ratio"],
    "index_share_layers": ["index_source_layer_ids", "index_share_for_mtp_iteration"],
    "index_rope_interleave": ["indexer_rope_interleave"],
    "compress_ratios": ["compress_ratios"],
    "compress_rope_theta": ["compress_rope_theta"],
    "candidate_selection": ["candidate_source_layer_id", "candidate_topk_blocks", "candidate_block_size"],
    # ---- positions --------------------------------------------------------------
    "position_encoding_type": ["position_embedding_type", "use_pos_enc", "use_rope", "rope_style"],
    "rope_theta": ["rope_theta", "default_theta"],
    "rope_theta_per_layer": ["layer_rope_theta"],
    "rope_theta_local": ["rope_local_base_freq"],
    "rope_scaling": ["rope_scaling", "rope_parameters"],
    "rope_scaling_type": ["rope_type"],
    "rope_scaling_factor": ["rope_factor", "scaling_factor"],
    "rope_scaling_params": ["mscale", "beta_fast", "beta_slow", "attn_factor", "extrapolation_factor"],
    "rope_original_max_position": ["original_max_position_embeddings", "initial_context_length", "original_seq_len"],
    "rope_scaling_layer_types": ["yarn_only_types"],
    "partial_rotary_factor": ["partial_rotary_factor", "partial_rotary_factors", "rotary_pct", "rotary_dim"],
    "rope_interleave": ["rope_interleave"],
    "nope_layers": ["no_rope_layers", "no_rope_layer_interval", "use_rope_layers"],
    "relative_position_bias": ["d_rel", "rel_extent"],
    # ---- feed-forward: MoE ------------------------------------------------------
    "moe": ["use_moe", "enable_moe_block"],
    "num_experts": ["n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts"],
    "experts_per_tok": ["num_experts_per_tok", "experts_per_token", "num_experts_per_token", "moe_top_k", "moe_topk", "top_k_experts", "experts_top_k"],
    "num_shared_experts": ["n_shared_experts", "num_shared_experts"],
    "shared_expert_intermediate_size": ["shared_expert_intermediate_size", "moe_shared_expert_intermediate_size", "share_expert_dim", "shared_intermediate_size"],
    "shared_expert_mode": ["shared_expert_combination_strategy", "shared_moe_mode", "shared_expert_sink"],
    "moe_intermediate_size": ["moe_intermediate_size", "expert_intermediate_size", "expert_ffn_hidden_size", "expert_hidden_dim", "routed_expert_hidden_size"],
    "dense_prefix_layers": ["first_k_dense_replace", "num_dense_layers", "n_dense_first_layers"],
    "moe_layer_schedule": ["moe_layer_freq", "moe_layers", "moe_layers_enum", "moe_every_n_layer", "moe_layer_offset", "interleave_moe_layer_step", "decoder_sparse_step", "mlp_only_layers", "mlp_layer_types", "dense_mlp_idx"],
    "router_scoring": ["scoring_func", "score_function", "score_func", "moe_router_activation", "moe_router_activation_func", "moe_router_use_sigmoid", "gate_activation"],
    "router_topk_method": ["topk_method", "expert_selection_fn", "use_grouped_topk"],
    "router_num_groups": ["n_group", "num_expert_group", "num_expert_groups"],
    "router_topk_groups": ["topk_group", "num_limited_groups"],
    "router_normalize_weights": ["norm_topk_prob", "moe_renormalize", "norm_expert_weight", "route_norm", "norm_after_topk"],
    "router_scaling_factor": ["routed_scaling_factor", "route_scale", "moe_routed_scaling_factor", "moe_router_scaling_factor", "router_scaling_factor", "use_global_scale"],
    "router_expert_bias": ["moe_router_enable_expert_bias", "use_routing_bias", "use_expert_bias", "use_moe_router_bias", "use_gate_bias"],
    "router_hidden_size": ["router_hidden_size"],
    "router_input_scaling": ["scale_router_input", "moe_apply_router_weight_on_input", "score_before_experts"],
    "router_logit_softcapping": ["router_logit_softcapping", "moe_router_logit_softcapping"],
    "moe_latent_size": ["moe_latent_size"],
    "moe_latent_norm": ["latent_moe_use_norm"],
    "zero_experts": ["zero_expert_num", "zero_expert_type"],
    "residual_moe": ["residual_moe"],
    "double_wide_mlp": ["use_double_wide_mlp"],
    # ---- hybrid sequence mixers ---------------------------------------------------
    "linear_attention_config": ["linear_attn_config"],
    "hybrid_attention_interval": ["full_attention_interval", "gqa_interval", "layer_group_size", "gqa_layers", "hybrid_block_size"],
    "linear_num_key_heads": ["linear_num_key_heads", "num_kv_heads_for_linear_attn"],
    "linear_num_value_heads": ["linear_num_value_heads"],
    "linear_key_head_dim": ["linear_key_head_dim"],
    "linear_value_head_dim": ["linear_value_head_dim"],
    "linear_group_norm_size": ["group_norm_size"],
    "linear_activation": ["linear_silu"],
    "kda_config": ["kda_lower_bound", "kda_safe_gate", "kda_allow_neg_eigval", "kda_use_full_proj", "use_kda_lora", "no_kda_lora"],
    "short_conv_kernel": ["linear_conv_kernel_dim", "short_conv_kernel_size", "sconv_kernel_size", "use_sconv", "conv_kernel", "mamba_d_conv", "conv_L_cache"],
    "conv_dim": ["conv_dim"],
    "conv_bias": ["use_conv_bias", "mamba_conv_bias", "conv_bias"],
    "mamba_num_heads": ["mamba_num_heads", "mamba_n_heads"],
    "mamba_head_dim": ["mamba_head_dim", "mamba_d_head"],
    "mamba_state_size": ["ssm_state_size", "mamba_d_state"],
    "mamba_num_groups": ["n_groups", "mamba_n_groups"],
    "mamba_expand": ["expand", "mamba_expand"],
    "mamba_activation": ["mamba_hidden_act"],
    "mamba_proj_bias": ["mamba_proj_bias"],
    "mamba_time_step": ["time_step_min", "time_step_max", "time_step_floor", "time_step_limit", "time_step_rank"],
    "mlstm_dims": ["qk_dim_factor", "v_dim_factor"],
    "mlstm_gate_softcap": ["gate_soft_cap"],
    "hash_layers": ["num_hash_layers"],
    # ---- residual stream ----------------------------------------------------------
    "hyper_connections": ["mhc", "mhc_enabled", "enable_ihc"],
    "hyper_connection_streams": ["hc_mult", "hc_count", "mhc_expansion_rate"],
    "hyper_connection_params": ["hc_eps", "hc_sinkhorn_iters", "mhc_sinkhorn_iters", "hc_lowrank", "hc_magnitude"],
    "attention_residual_block": ["attn_res_block_size"],
    "residual_multiplier": ["residual_multiplier"],
    "embedding_multiplier": ["embedding_multiplier", "embedding_multiplier_scale"],
    "output_multiplier": ["logits_scaling", "output_multiplier", "output_multiplier_scale", "logits_mup_width_multiplier", "logit_scale"],
    "mup": ["mup_enabled"],
    "final_logit_softcapping": ["final_logit_softcapping", "output_logit_soft_cap"],
    # ---- looped depth -------------------------------------------------------------
    "loop_passes": ["num_loops", "total_ut_steps"],
    "loop_exit_threshold": ["early_exit_threshold"],
    # ---- per-layer embeddings -----------------------------------------------------
    "per_layer_embedding_dim": ["hidden_size_per_layer_input", "ple_embed_dim"],
    "per_layer_embedding_vocab": ["vocab_size_per_layer_input"],
    "per_layer_embedding_layers": ["ple_layer_ids"],
    "per_layer_embedding_conv": ["ple_conv_kernel_size"],
    # ---- n-gram memories ----------------------------------------------------------
    "engram_layers": ["engram_layer_ids"],
    "engram_size": ["engram_num_embeddings", "engram_vocab_size", "engram_compressed_vocab_size", "engram_max_ngram_size"],
    "engram_heads": ["engram_n_heads", "engram_head_dim"],
    "ngram_embedding": ["ngram_size", "ngram_vocab_size_base", "ngram_vocab_size_ratio", "make_ngram_vocab_size_divisible_by", "split_ngram_parts", "heads_per_ngram", "emb_neighbor_num", "emb_split_num"],
    "dspark": ["dspark_block_size", "dspark_markov_rank", "dspark_n_routed_experts", "dspark_num_experts_per_tok", "dspark_target_layer_ids"],
}

# Sections follow the forward pass of one decoder block: embeddings in, token mixing, channel
# mixing, the residual stream around them, training-only heads, and the code identity.
# Keys are "Section › Subsection"; the order here is the order of the model card and of the change list.
GROUPS = {
    "Embeddings & output": ["vocab_size", "vocab_size_unpadded", "tie_embeddings", "lm_head_bias", "embedding_multiplier", "output_multiplier", "final_logit_softcapping",
                            "per_layer_embedding_dim", "per_layer_embedding_vocab", "per_layer_embedding_layers", "per_layer_embedding_conv",
                            "engram_layers", "engram_size", "engram_heads", "ngram_embedding", "dspark"],
    "Token mixing › Mixer kind & layer schedule": ["sequence_mixer", "layer_types", "mixer_attention_ratio", "hybrid_attention_interval", "local_global_ratio",
                            "sliding_window", "sliding_window_enabled", "sliding_window_pattern", "sliding_window_max_layers", "chunked_attention_size", "bidirectional_attention"],
    "Token mixing › Attention heads & projections": ["attention_kind", "num_heads", "num_kv_heads", "head_dim", "v_head_dim", "attention_bias", "attention_impl_type",
                            "gated_attention", "gated_attention_type", "qk_norm", "qk_norm_type", "attention_scale", "attention_value_scale", "attention_logit_softcapping",
                            "attention_temperature_tuning", "attention_log_scaling", "attention_differential", "attention_cca_steps", "attention_heads_per_layer",
                            "swa_num_heads", "swa_num_kv_heads", "swa_head_dim", "swa_rope_theta", "swa_attention_config", "global_head_dim", "global_num_kv_heads",
                            "global_kv_unified", "kv_shared_layers", "attention_sinks"],
    "Token mixing › Latent attention": ["mla", "q_lora_rank", "kv_lora_rank", "o_lora_rank", "o_groups", "qk_rope_head_dim", "qk_nope_head_dim", "mla_scale_lora", "mla_nope"],
    "Token mixing › Sparse attention & indexer": ["sparse_attention", "index_topk", "index_num_heads", "index_head_dim", "index_kv_heads", "index_layer_types", "index_kpool",
                            "index_share_layers", "index_rope_interleave", "compress_ratios", "compress_rope_theta", "candidate_selection"],
    "Token mixing › Linear & recurrent mixers": ["linear_attention_config", "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
                            "linear_group_norm_size", "linear_activation", "kda_config", "short_conv_kernel", "conv_dim", "conv_bias", "mamba_num_heads", "mamba_head_dim",
                            "mamba_state_size", "mamba_num_groups", "mamba_expand", "mamba_activation", "mamba_proj_bias", "mamba_time_step", "mlstm_dims", "mlstm_gate_softcap", "hash_layers"],
    "Token mixing › Positions": ["max_position_embeddings", "position_encoding_type", "rope_theta", "rope_theta_per_layer", "rope_theta_local", "rope_scaling", "rope_scaling_type", "rope_scaling_factor",
                            "rope_scaling_params", "rope_original_max_position", "rope_scaling_layer_types", "partial_rotary_factor", "rope_interleave", "nope_layers", "relative_position_bias"],
    "Channel mixing › Dense FFN": ["intermediate_size", "dense_prefix_intermediate_size", "ffn_width_multiplier", "ffn_round_to_multiple", "mlp_bias", "activation", "activation_gated",
                            "activation_clamp", "activation_clamp_shared_expert", "activation_clamp_experts", "activation_situ_beta", "polynorm_output_scale", "polynorm_bias_clamp", "double_wide_mlp"],
    "Channel mixing › Experts": ["moe", "num_experts", "experts_per_tok", "num_shared_experts", "shared_expert_intermediate_size", "shared_expert_mode", "moe_intermediate_size",
                            "dense_prefix_layers", "moe_layer_schedule", "moe_latent_size", "moe_latent_norm", "zero_experts", "residual_moe"],
    "Channel mixing › Router": ["router_scoring", "router_topk_method", "router_num_groups", "router_topk_groups", "router_normalize_weights", "router_scaling_factor",
                            "router_expert_bias", "router_hidden_size", "router_input_scaling", "router_logit_softcapping"],
    "Block structure & residual stream": ["num_layers", "hidden_size", "norm_type", "norm_eps", "norm_eps_post", "norm_eps_cell", "norm_extra_placement", "norm_beta_attention",
                            "norm_beta_linear_attention", "norm_beta_mlp", "norm_zero_centered", "ngpt", "parallel_block", "hyper_connections", "hyper_connection_streams",
                            "hyper_connection_params", "attention_residual_block", "residual_multiplier", "mup", "loop_passes", "loop_exit_threshold"],
}
SECTIONS = {}
for _g in GROUPS:
    SECTIONS.setdefault(_g.split(" › ")[0], []).append(_g)

DERIVED = {"attention_kind", "local_global_ratio", "sequence_mixer", "mixer_attention_ratio"}
RAW_TO_CANONICAL = {raw: canon for canon, raws in ALIASES.items() for raw in raws}
CANONICAL_GROUP = {canon: g for g, canons in GROUPS.items() for canon in canons}


# ---- which canonical fields count as a design change ---------------------------------
# SCALE: sizes and counts that grow with the model. A change in value is a bigger or smaller
# model, not a different architecture; only appearing or disappearing counts.
SCALE = {
    "hidden_size", "num_layers", "intermediate_size", "dense_prefix_intermediate_size", "vocab_size", "vocab_size_unpadded",
    "max_position_embeddings", "num_heads", "num_kv_heads", "head_dim", "v_head_dim", "q_lora_rank", "kv_lora_rank", "o_lora_rank",
    "o_groups", "qk_rope_head_dim", "qk_nope_head_dim", "num_experts", "experts_per_tok", "num_shared_experts",
    "shared_expert_intermediate_size", "moe_intermediate_size", "dense_prefix_layers", "router_num_groups", "router_topk_groups",
    "moe_latent_size", "router_hidden_size", "index_topk", "index_num_heads", "index_head_dim", "index_kv_heads", "sliding_window",
    "sliding_window_max_layers", "chunked_attention_size", "swa_num_heads", "swa_num_kv_heads", "swa_head_dim", "global_head_dim",
    "global_num_kv_heads", "kv_shared_layers", "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
    "linear_value_head_dim", "linear_group_norm_size", "mamba_num_heads", "mamba_head_dim", "mamba_state_size", "mamba_num_groups",
    "mamba_expand", "conv_dim", "short_conv_kernel", "hyper_connection_streams", "attention_residual_block", "per_layer_embedding_dim",
    "per_layer_embedding_vocab", "engram_size", "engram_heads", "loop_passes", "attention_heads_per_layer",
    "attention_cca_steps", "relative_position_bias", "ffn_round_to_multiple", "mlstm_dims", "ffn_width_multiplier",
}
# TUNING: continuous hyper-parameters (epsilons, thetas, scales, clamps). Same rule as SCALE.
TUNING = {
    "norm_eps", "norm_eps_post", "norm_eps_cell", "norm_beta_attention", "norm_beta_linear_attention", "norm_beta_mlp",
    "activation_clamp", "activation_clamp_shared_expert", "activation_clamp_experts", "activation_situ_beta", "polynorm_output_scale",
    "polynorm_bias_clamp", "attention_scale", "attention_value_scale", "attention_logit_softcapping", "attention_temperature_tuning",
    "attention_log_scaling", "rope_theta", "rope_theta_per_layer", "rope_theta_local", "rope_scaling_factor", "rope_scaling_params",
    "rope_original_max_position", "partial_rotary_factor", "swa_rope_theta", "compress_rope_theta", "router_scaling_factor",
    "router_logit_softcapping", "mamba_time_step", "mlstm_gate_softcap", "hyper_connection_params", "residual_multiplier",
    "embedding_multiplier", "output_multiplier", "final_logit_softcapping", "loop_exit_threshold", "kda_config", "candidate_selection",
    "index_kpool", "ngram_embedding", "dspark",
}
DESIGN = set(ALIASES) - SCALE - TUNING
assert SCALE <= set(ALIASES) and TUNING <= set(ALIASES) and not (SCALE & TUNING)

# Never reported as a change: multi-token-prediction heads are a training aid, and identity fields name the code.
NOT_A_CHANGE = set()  # kept for the change rule; multi-token-prediction heads and identity keys are dropped upstream
# Scale fields whose appearance marks a mechanism (their value never counts, their presence does).
PRESENCE = {"kv_lora_rank", "q_lora_rank", "index_topk", "mamba_num_heads", "mamba_state_size", "linear_num_value_heads",
            "kv_shared_layers", "per_layer_embedding_dim", "hyper_connection_streams", "loop_passes", "moe_latent_size", "num_experts",
            "engram_size", "sliding_window", "chunked_attention_size", "num_shared_experts", "dense_prefix_layers"}
assert NOT_A_CHANGE <= set(ALIASES) and PRESENCE <= SCALE
