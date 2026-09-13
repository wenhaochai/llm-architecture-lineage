# From 571 config keys to the canonical fields

The 103 configs use **571 distinct keys** on their text sub-config. `key_taxonomy.py` drops **154** that carry no architecture, and `schema.py` renames the remaining **417** into **168** canonical fields, every raw key covered exactly once. Nested sub-configs (`linear_attn_config`, `rope_scaling`, `attention_other_setting`, `sparse_attention_config`) are flattened into the same fields, per-layer index lists become layer schedules by kind, four canonical fields are derived (`attention_kind`, `sequence_mixer`, `local_global_ratio`, `mixer_attention_ratio`), and traits the modeling class implies are filled in by `extract.py` with the implying model type recorded as their source.

Fields sit in four sections that follow the forward pass of a decoder block; token mixing and channel mixing have subsections. Each field is tagged **design** (66), **scale** (65) or **tuning** (41). The change list between a model and a candidate parent: a design field counts when its value differs (per-layer schedules compared by the kinds of layer they contain); a scale field never counts by value, and 17 of them (`PRESENCE`) count when they appear because their presence marks a mechanism; tuning fields never count. Candidate parents are ranked by the number of changes among the mechanism-level fields (`MECHANISM`, 16 fields) first, then by the total.

## Dropped

### Generation, tokenizer, Hugging Face bookkeeping (68)

`eos_token_id` ×90, `transformers_version` ×73, `bos_token_id` ×70, `pad_token_id` ×67, `auto_map` ×33, `output_router_logits` ×22, `_name_or_path` ×8, `max_length` ×5, `sep_token_id` ×5, `task_specific_params` ×5, `add_cross_attention` ×4, `bad_words_ids` ×4, `begin_suppress_tokens` ×4, `chunk_size_feed_forward` ×4, `cross_attention_hidden_size` ×4, `decoder_start_token_id` ×4, `diversity_penalty` ×4, `do_sample` ×4, `early_stopping` ×4, `encoder_no_repeat_ngram_size` ×4, `exponential_decay_length_penalty` ×4, `finetuning_task` ×4, `forced_bos_token_id` ×4, `forced_eos_token_id` ×4, `id2label` ×4, `is_decoder` ×4, `is_encoder_decoder` ×4, `label2id` ×4, `length_penalty` ×4, `min_length` ×4, `no_repeat_ngram_size` ×4, `num_beam_groups` ×4, `num_beams` ×4, `num_return_sequences` ×4, `output_attentions` ×4, `output_hidden_states` ×4, `output_scores` ×4, `prefix` ×4, `problem_type` ×4, `pruned_heads` ×4, `remove_invalid_values` ×4, `repetition_penalty` ×4, `return_dict` ×4, `return_dict_in_generate` ×4, `suppress_tokens` ×4, `temperature` ×4, `tf_legacy_loss` ×4, `tie_encoder_decoder` ×4, `tokenizer_class` ×4, `top_k` ×4, `top_p` ×4, `torchscript` ×4, `typical_p` ×4, `use_bfloat16` ×4, `unsloth_fixed` ×3, `_provenance` ×2, `dspark_noise_token_id`, `engram_pad_token_id`, `eod_token_id`, `force_bos_token_insert`, `output_past`, `summary_activation`, `summary_first_dropout`, `summary_proj_to_labels`, `summary_type`, `summary_use_proj`, `transformers.js_config`, `unsloth_version`

### Runtime dtype, kernels, parallelism, implementation switches (40)

`use_cache` ×96, `torch_dtype` ×49, `dtype` ×42, `quantization_config` ×21, `pretraining_tp` ×14, `ep_size` ×9, `chunk_size` ×7, `mamba_ssm_dtype` ×7, `num_logits_to_keep` ×6, `residual_in_fp32` ×6, `use_mamba_kernels` ×6, `mamba_ssm_cache_dtype` ×5, `router_dtype` ×4, `moe_shared_expert_overlap` ×3, `use_grouped_mm` ×3, `use_parallel_embedding` ×3, `attention_projection_layout` ×2, `attn_implementation` ×2, `cache_implementation` ×2, `enable_lm_head_fp32` ×2, `expert_dtype` ×2, `moe_router_dtype` ×2, `_attn_implementation`, `_debug_force_load_balance`, `add_forward_backend_padding`, `autocast_kernel_dtype`, `bitwise_backward_align`, `chunkwise_kernel`, `enable_attention_fp32_softmax`, `enable_moe_fp32_combine`, `for_llm_compressor`, `inference_state_dtype`, `mamba_chunk_size`, `mode`, `need_fp32_gate`, `norm_reduction_force_float32`, `return_last_states`, `sequence_kernel`, `step_kernel`, `weight_mode`

### Training-only (26)

`initializer_range` ×90, `attention_dropout` ×87, `router_aux_loss_coef` ×19, `seq_aux` ×7, `hidden_dropout` ×6, `rescale_prenorm_residual` ×6, `embedding_dropout` ×5, `output_dropout` ×5, `aux_loss_alpha` ×4, `embd_pdrop` ×3, `mtp_loss_scaling_factor` ×3, `resid_pdrop` ×3, `block_mlp_init_scale` ×2, `block_out_init_scale` ×2, `block_use_xavier_init` ×2, `conv_use_xavier_init` ×2, `init_method` ×2, `load_balance_coeff` ×2, `router_jitter_noise` ×2, `add_embedding_dropout`, `attn_pdrop`, `igate_bias_init_range`, `learnable_sink_init`, `loop_loss_weights`, `mhc_identity_init`, `mtp_loss_factor`

### Multi-token-prediction heads (10)

`num_nextn_predict_layers` ×31, `mtp_num_hidden_layers` ×7, `mtp_use_dedicated_embeddings` ×7, `num_mtp_modules` ×4, `mtp_transformer_layers` ×3, `use_mtp` ×3, `mtp_layers_block_type` ×2, `mtp`, `mtp_hybrid_override_pattern`, `mtp_use_kda`

### Multimodal leftovers (8)

`vision_config` ×2, `audio_config`, `image_token_id`, `processor_config`, `video_token_id`, `vision_end_token_id`, `vision_model_type`, `vision_start_token_id`

### Identity of the modeling code (2)

`model_type` ×100, `architectures` ×83

## Canonical fields

| section | canonical field | kind | raw spellings (×configs) |
|---|---|---|---|
| Embeddings & output | `vocab_size` | scale | `vocab_size` ×103 |
| Embeddings & output | `vocab_size_unpadded` | scale | `unpadded_vocab_size` ×1 |
| Embeddings & output | `tie_embeddings` | design | `tie_word_embeddings` ×89, `tie_embedding` ×2, `use_embedding_sharing` ×2 |
| Embeddings & output | `lm_head_bias` | design | `lm_head_bias` ×1 |
| Embeddings & output | `embedding_multiplier` | tuning | `embedding_multiplier` ×2, `embedding_multiplier_scale` ×1 |
| Embeddings & output | `output_multiplier` | tuning | `logits_scaling` ×2, `output_multiplier` ×1, `output_multiplier_scale` ×1, `logits_mup_width_multiplier` ×1, `logit_scale` ×3 |
| Embeddings & output | `final_logit_softcapping` | tuning | `final_logit_softcapping` ×10, `output_logit_soft_cap` ×1 |
| Embeddings & output | `per_layer_embedding_dim` | scale · presence | `hidden_size_per_layer_input` ×5, `ple_embed_dim` ×1 |
| Embeddings & output | `per_layer_embedding_vocab` | scale | `vocab_size_per_layer_input` ×5 |
| Embeddings & output | `per_layer_embedding_layers` | design | `ple_layer_ids` ×1 |
| Embeddings & output | `per_layer_embedding_conv` | design | `ple_conv_kernel_size` ×1 |
| Embeddings & output | `engram_layers` | design | `engram_layer_ids` ×1 |
| Embeddings & output | `engram_size` | scale · presence | `engram_num_embeddings` ×1, `engram_vocab_size` ×1, `engram_compressed_vocab_size` ×1, `engram_max_ngram_size` ×1 |
| Embeddings & output | `engram_heads` | scale | `engram_n_heads` ×1, `engram_head_dim` ×1 |
| Embeddings & output | `ngram_embedding` | tuning | `ngram_size` ×1, `ngram_vocab_size_base` ×1, `ngram_vocab_size_ratio` ×1, `make_ngram_vocab_size_divisible_by` ×1, `split_ngram_parts` ×1, `heads_per_ngram` ×1, `emb_neighbor_num` ×1, `emb_split_num` ×1 |
| Embeddings & output | `dspark` | tuning | `dspark_block_size` ×1, `dspark_markov_rank` ×1, `dspark_n_routed_experts` ×1, `dspark_num_experts_per_tok` ×1, `dspark_target_layer_ids` ×1 |
| Token mixing › Mixer kind & layer schedule | `sequence_mixer` | design | *derived* |
| Token mixing › Mixer kind & layer schedule | `layer_types` | design | `layer_types` ×35, `attn_type_list` ×3, `hybrid_layer_pattern` ×3, `local_layer_ids` ×1, `order_of_interleaved_layers` ×2, `layers_block_type` ×2, `hybrid_override_pattern` ×4 |
| Token mixing › Mixer kind & layer schedule | `mixer_attention_ratio` | design | *derived* |
| Token mixing › Mixer kind & layer schedule | `hybrid_attention_interval` | design | `full_attention_interval` ×8, `gqa_interval` ×1, `layer_group_size` ×3, `gqa_layers` ×1, `hybrid_block_size` ×1 |
| Token mixing › Mixer kind & layer schedule | `local_global_ratio` | design | *derived* |
| Token mixing › Mixer kind & layer schedule | `sliding_window` | scale · presence | `sliding_window` ×49, `sliding_window_size` ×5 |
| Token mixing › Mixer kind & layer schedule | `sliding_window_enabled` | design | `use_sliding_window` ×13 |
| Token mixing › Mixer kind & layer schedule | `sliding_window_pattern` | design | `sliding_window_pattern` ×2, `_sliding_window_pattern` ×3, `layer_switch` ×2, `sliding_window_period` ×1, `global_attn_every_n` ×1, `global_attn_every_n_layers` ×1, `prefix_dense_sliding_window_pattern` ×2 |
| Token mixing › Mixer kind & layer schedule | `sliding_window_max_layers` | scale | `max_window_layers` ×15 |
| Token mixing › Mixer kind & layer schedule | `chunked_attention_size` | scale · presence | `attention_chunk_size` ×4 |
| Token mixing › Mixer kind & layer schedule | `bidirectional_attention` | design | `use_bidirectional_attention` ×6 |
| Token mixing › Attention heads & projections | `attention_kind` | design | *derived* |
| Token mixing › Attention heads & projections | `num_heads` | scale | `num_attention_heads` ×101, `n_head` ×1, `num_heads` ×3 |
| Token mixing › Attention heads & projections | `num_kv_heads` | scale | `num_key_value_heads` ×98, `num_attention_groups` ×1 |
| Token mixing › Attention heads & projections | `head_dim` | scale | `head_dim` ×81, `qk_head_dim` ×9, `kv_channels` ×1, `q_head_dim` ×1 |
| Token mixing › Attention heads & projections | `v_head_dim` | scale | `v_head_dim` ×25 |
| Token mixing › Attention heads & projections | `attention_bias` | design | `attention_bias` ×75, `use_qkv_bias` ×4, `use_bias` ×11, `q_bias` ×1, `o_bias` ×1 |
| Token mixing › Attention heads & projections | `attention_impl_type` | design | `att_impl_type` ×1, `attention_cls` ×1 |
| Token mixing › Attention heads & projections | `gated_attention` | design | `attn_output_gate` ×6, `attention_output_gate` ×1, `use_gqa_gate` ×1, `use_head_wise_attn_gate` ×1, `headwise_attn_output_gate` ×1, `elementwise_attn_output_gate` ×1, `gated_mla` ×1, `mla_use_output_gate` ×1 |
| Token mixing › Attention heads & projections | `gated_attention_type` | design | `output_gate_type` ×4, `gating` ×3, `gating_type` ×1, `gating_types` ×2, `gated_attention_proj_granularity_type` ×1 |
| Token mixing › Attention heads & projections | `qk_norm` | design | `use_qk_norm` ×18, `qk_norm` ×1, `add_qk_norm` ×1 |
| Token mixing › Attention heads & projections | `qk_norm_type` | design | `qk_norm_type` ×4 |
| Token mixing › Attention heads & projections | `attention_scale` | tuning | `query_pre_attn_scalar` ×2, `attn_scale` ×1, `qk_scale_factor` ×1, `attention_multiplier` ×2 |
| Token mixing › Attention heads & projections | `attention_value_scale` | tuning | `attention_value_scale` ×3 |
| Token mixing › Attention heads & projections | `attention_logit_softcapping` | tuning | `attn_logit_softcapping` ×3 |
| Token mixing › Attention heads & projections | `attention_temperature_tuning` | tuning | `attn_temperature_tuning` ×1, `attn_temperature_len` ×1, `floor_scale` ×1 |
| Token mixing › Attention heads & projections | `attention_log_scaling` | tuning | `log_scaling_alpha` ×1, `log_scaling_n_floor` ×1 |
| Token mixing › Attention heads & projections | `attention_differential` | design | `diff_v2` ×1, `k_ratio` ×1, `num_noise_heads` ×1 |
| Token mixing › Attention heads & projections | `attention_cca_steps` | scale | `cca_time0` ×1, `cca_time1` ×1 |
| Token mixing › Attention heads & projections | `attention_heads_per_layer` | scale | `num_attention_heads_per_layer` ×3 |
| Token mixing › Attention heads & projections | `swa_num_heads` | scale | `swa_num_attention_heads` ×4 |
| Token mixing › Attention heads & projections | `swa_num_kv_heads` | scale | `swa_num_key_value_heads` ×4 |
| Token mixing › Attention heads & projections | `swa_head_dim` | scale | `swa_head_dim` ×4, `swa_v_head_dim` ×3 |
| Token mixing › Attention heads & projections | `swa_rope_theta` | tuning | `swa_rope_theta` ×4 |
| Token mixing › Attention heads & projections | `swa_attention_config` | design | `attention_other_setting` ×1 |
| Token mixing › Attention heads & projections | `global_head_dim` | scale | `global_head_dim` ×5 |
| Token mixing › Attention heads & projections | `global_num_kv_heads` | scale | `num_global_key_value_heads` ×5 |
| Token mixing › Attention heads & projections | `global_kv_unified` | design | `attention_k_eq_v` ×5 |
| Token mixing › Attention heads & projections | `kv_shared_layers` | scale · presence | `num_kv_shared_layers` ×5, `kv_source_layer_ids` ×1 |
| Token mixing › Attention heads & projections | `attention_sinks` | design | `sink` ×1, `learnable_sink` ×1, `add_full_attention_sink_bias` ×3, `add_swa_attention_sink_bias` ×3 |
| Token mixing › Latent attention | `mla` | design | `use_mla` ×1 |
| Token mixing › Latent attention | `q_lora_rank` | scale · presence | `q_lora_rank` ×24 |
| Token mixing › Latent attention | `kv_lora_rank` | scale · presence | `kv_lora_rank` ×22 |
| Token mixing › Latent attention | `o_lora_rank` | scale | `o_lora_rank` ×3 |
| Token mixing › Latent attention | `o_groups` | scale | `o_groups` ×3 |
| Token mixing › Latent attention | `qk_rope_head_dim` | scale | `qk_rope_head_dim` ×25 |
| Token mixing › Latent attention | `qk_nope_head_dim` | scale | `qk_nope_head_dim` ×21 |
| Token mixing › Latent attention | `mla_scale_lora` | design | `mla_scale_q_lora` ×1, `mla_scale_kv_lora` ×1 |
| Token mixing › Latent attention | `mla_nope` | design | `mla_use_nope` ×3, `use_mla_nope` ×1 |
| Token mixing › Sparse attention & indexer | `sparse_attention` | design | `use_dsa` ×1, `sparse_attention_config` ×1 |
| Token mixing › Sparse attention & indexer | `index_topk` | scale · presence | `index_topk` ×9, `indexer_budget` ×1 |
| Token mixing › Sparse attention & indexer | `index_num_heads` | scale | `index_n_heads` ×9, `indexer_n_heads` ×1 |
| Token mixing › Sparse attention & indexer | `index_head_dim` | scale | `index_head_dim` ×9, `indexer_head_dim` ×1 |
| Token mixing › Sparse attention & indexer | `index_kv_heads` | scale | `indexer_kv_heads` ×1 |
| Token mixing › Sparse attention & indexer | `index_layer_types` | design | `indexer_types` ×3, `index_topk_pattern` ×1, `index_topk_freq` ×1, `index_skip_topk_offset` ×1 |
| Token mixing › Sparse attention & indexer | `index_kpool` | tuning | `index_kpool` ×1, `index_kpool_always_select_tail` ×1, `index_kpool_compress` ×1, `indexer_compress_ratio` ×1 |
| Token mixing › Sparse attention & indexer | `index_share_layers` | design | `index_source_layer_ids` ×1, `index_share_for_mtp_iteration` ×2 |
| Token mixing › Sparse attention & indexer | `index_rope_interleave` | design | `indexer_rope_interleave` ×4 |
| Token mixing › Sparse attention & indexer | `compress_ratios` | design | `compress_ratios` ×3 |
| Token mixing › Sparse attention & indexer | `compress_rope_theta` | tuning | `compress_rope_theta` ×3 |
| Token mixing › Sparse attention & indexer | `candidate_selection` | tuning | `candidate_source_layer_id` ×1, `candidate_topk_blocks` ×1, `candidate_block_size` ×1 |
| Token mixing › Linear & recurrent mixers | `linear_attention_config` | design | `linear_attn_config` ×4 |
| Token mixing › Linear & recurrent mixers | `linear_num_key_heads` | scale | `linear_num_key_heads` ×8, `num_kv_heads_for_linear_attn` ×3 |
| Token mixing › Linear & recurrent mixers | `linear_num_value_heads` | scale · presence | `linear_num_value_heads` ×8 |
| Token mixing › Linear & recurrent mixers | `linear_key_head_dim` | scale | `linear_key_head_dim` ×8 |
| Token mixing › Linear & recurrent mixers | `linear_value_head_dim` | scale | `linear_value_head_dim` ×8 |
| Token mixing › Linear & recurrent mixers | `linear_group_norm_size` | scale | `group_norm_size` ×3 |
| Token mixing › Linear & recurrent mixers | `linear_activation` | design | `linear_silu` ×3 |
| Token mixing › Linear & recurrent mixers | `kda_config` | tuning | `kda_lower_bound` ×1, `kda_safe_gate` ×1, `kda_allow_neg_eigval` ×1, `kda_use_full_proj` ×1, `use_kda_lora` ×1, `no_kda_lora` ×1 |
| Token mixing › Linear & recurrent mixers | `short_conv_kernel` | scale | `linear_conv_kernel_dim` ×8, `short_conv_kernel_size` ×1, `sconv_kernel_size` ×1, `use_sconv` ×1, `conv_kernel` ×6, `mamba_d_conv` ×1, `conv_L_cache` ×3 |
| Token mixing › Linear & recurrent mixers | `conv_dim` | scale | `conv_dim` ×2 |
| Token mixing › Linear & recurrent mixers | `conv_bias` | design | `use_conv_bias` ×6, `mamba_conv_bias` ×1, `conv_bias` ×3 |
| Token mixing › Linear & recurrent mixers | `mamba_num_heads` | scale · presence | `mamba_num_heads` ×6, `mamba_n_heads` ×1 |
| Token mixing › Linear & recurrent mixers | `mamba_head_dim` | scale | `mamba_head_dim` ×6, `mamba_d_head` ×1 |
| Token mixing › Linear & recurrent mixers | `mamba_state_size` | scale · presence | `ssm_state_size` ×6, `mamba_d_state` ×1 |
| Token mixing › Linear & recurrent mixers | `mamba_num_groups` | scale | `n_groups` ×6, `mamba_n_groups` ×1 |
| Token mixing › Linear & recurrent mixers | `mamba_expand` | scale | `expand` ×6, `mamba_expand` ×1 |
| Token mixing › Linear & recurrent mixers | `mamba_activation` | design | `mamba_hidden_act` ×6 |
| Token mixing › Linear & recurrent mixers | `mamba_proj_bias` | design | `mamba_proj_bias` ×7 |
| Token mixing › Linear & recurrent mixers | `mamba_time_step` | tuning | `time_step_min` ×7, `time_step_max` ×7, `time_step_floor` ×6, `time_step_limit` ×2, `time_step_rank` ×1 |
| Token mixing › Linear & recurrent mixers | `mlstm_dims` | scale | `qk_dim_factor` ×1, `v_dim_factor` ×1 |
| Token mixing › Linear & recurrent mixers | `mlstm_gate_softcap` | tuning | `gate_soft_cap` ×1 |
| Token mixing › Linear & recurrent mixers | `hash_layers` | design | `num_hash_layers` ×2 |
| Token mixing › Positions | `max_position_embeddings` | scale | `max_position_embeddings` ×99, `n_positions` ×1, `n_ctx` ×1, `max_seq_len` ×1, `model_max_length` ×2 |
| Token mixing › Positions | `position_encoding_type` | design | `position_embedding_type` ×3, `use_pos_enc` ×2, `use_rope` ×1, `rope_style` ×1 |
| Token mixing › Positions | `rope_theta` | tuning | `rope_theta` ×71, `default_theta` ×1 |
| Token mixing › Positions | `rope_theta_per_layer` | tuning | `layer_rope_theta` ×1 |
| Token mixing › Positions | `rope_theta_local` | tuning | `rope_local_base_freq` ×2 |
| Token mixing › Positions | `rope_scaling` | design | `rope_scaling` ×55, `rope_parameters` ×27 |
| Token mixing › Positions | `rope_scaling_type` | design | `rope_type` ×1 |
| Token mixing › Positions | `rope_scaling_factor` | tuning | `rope_factor` ×1, `scaling_factor` ×1 |
| Token mixing › Positions | `rope_scaling_params` | tuning | `mscale` ×1, `beta_fast` ×1, `beta_slow` ×1, `attn_factor` ×1, `extrapolation_factor` ×1 |
| Token mixing › Positions | `rope_original_max_position` | tuning | `original_max_position_embeddings` ×2, `initial_context_length` ×2, `original_seq_len` ×1 |
| Token mixing › Positions | `rope_scaling_layer_types` | design | `yarn_only_types` ×1 |
| Token mixing › Positions | `partial_rotary_factor` | tuning | `partial_rotary_factor` ×26, `partial_rotary_factors` ×1, `rotary_pct` ×2, `rotary_dim` ×7 |
| Token mixing › Positions | `rope_interleave` | design | `rope_interleave` ×7 |
| Token mixing › Positions | `nope_layers` | design | `no_rope_layers` ×1, `no_rope_layer_interval` ×1, `use_rope_layers` ×1 |
| Token mixing › Positions | `relative_position_bias` | scale | `d_rel` ×1, `rel_extent` ×1 |
| Channel mixing › Dense FFN | `intermediate_size` | scale | `intermediate_size` ×92, `ffn_hidden_size` ×1, `intermediate_size_mlp` ×1, `mlp_intermediate_size` ×1, `block_ff_dim` ×2, `dense_intermediate_size` ×2 |
| Channel mixing › Dense FFN | `dense_prefix_intermediate_size` | scale | `prefix_dense_intermediate_size` ×2 |
| Channel mixing › Dense FFN | `ffn_width_multiplier` | scale | `ffn_proj_factor` ×1, `block_ffn_dim_multiplier` ×2 |
| Channel mixing › Dense FFN | `ffn_round_to_multiple` | scale | `ffn_round_up_to_multiple_of` ×1, `block_multiple_of` ×2, `block_auto_adjust_ff_dim` ×2, `mlstm_round_up_to_multiple_of` ×1 |
| Channel mixing › Dense FFN | `mlp_bias` | design | `mlp_bias` ×12 |
| Channel mixing › Dense FFN | `activation` | design | `hidden_act` ×76, `hidden_activation` ×8, `activation_function` ×1, `mlp_hidden_act` ×6 |
| Channel mixing › Dense FFN | `activation_gated` | design | `use_gated_activation` ×3, `block_use_swiglu` ×2 |
| Channel mixing › Dense FFN | `activation_clamp` | tuning | `swiglu_limit` ×8, `swiglu_limits` ×1, `swiglu_alpha` ×1, `hidden_clamp` ×1 |
| Channel mixing › Dense FFN | `activation_clamp_shared_expert` | tuning | `swiglu_limits_shared` ×1, `share_expert_swiglu_limit_list` ×1 |
| Channel mixing › Dense FFN | `activation_clamp_experts` | tuning | `expert_swiglu_limit_list` ×1 |
| Channel mixing › Dense FFN | `activation_situ_beta` | tuning | `activation_situ_beta` ×1, `activation_situ_linear_beta` ×1 |
| Channel mixing › Dense FFN | `polynorm_output_scale` | tuning | `polynorm_output_scale` ×1, `polynorm_output_scale_per_layer` ×1 |
| Channel mixing › Dense FFN | `polynorm_bias_clamp` | tuning | `polynorm_bias_clamp` ×1 |
| Channel mixing › Dense FFN | `double_wide_mlp` | design | `use_double_wide_mlp` ×5 |
| Channel mixing › Experts | `moe` | design | `use_moe` ×1, `enable_moe_block` ×5 |
| Channel mixing › Experts | `num_experts` | scale · presence | `n_routed_experts` ×32, `num_experts` ×31, `num_local_experts` ×9, `moe_num_experts` ×1 |
| Channel mixing › Experts | `experts_per_tok` | scale | `num_experts_per_tok` ×63, `experts_per_token` ×2, `num_experts_per_token` ×2, `moe_top_k` ×1, `moe_topk` ×1, `top_k_experts` ×5, `experts_top_k` ×1 |
| Channel mixing › Experts | `num_shared_experts` | scale · presence | `n_shared_experts` ×32, `num_shared_experts` ×12 |
| Channel mixing › Experts | `shared_expert_intermediate_size` | scale | `shared_expert_intermediate_size` ×8, `moe_shared_expert_intermediate_size` ×9, `share_expert_dim` ×1, `shared_intermediate_size` ×5 |
| Channel mixing › Experts | `shared_expert_mode` | design | `shared_expert_combination_strategy` ×2, `shared_moe_mode` ×1, `shared_expert_sink` ×1 |
| Channel mixing › Experts | `moe_intermediate_size` | scale | `moe_intermediate_size` ×58, `expert_intermediate_size` ×3, `expert_ffn_hidden_size` ×1, `expert_hidden_dim` ×1, `routed_expert_hidden_size` ×1 |
| Channel mixing › Experts | `dense_prefix_layers` | scale · presence | `first_k_dense_replace` ×28, `num_dense_layers` ×2, `n_dense_first_layers` ×1 |
| Channel mixing › Experts | `moe_layer_schedule` | design | `moe_layer_freq` ×17, `moe_layers` ×1, `moe_layers_enum` ×1, `moe_every_n_layer` ×1, `moe_layer_offset` ×1, `interleave_moe_layer_step` ×2, `decoder_sparse_step` ×6, `mlp_only_layers` ×7, `mlp_layer_types` ×7, `dense_mlp_idx` ×1 |
| Channel mixing › Experts | `moe_latent_size` | scale · presence | `moe_latent_size` ×3 |
| Channel mixing › Experts | `moe_latent_norm` | design | `latent_moe_use_norm` ×1 |
| Channel mixing › Experts | `zero_experts` | design | `zero_expert_num` ×1, `zero_expert_type` ×1 |
| Channel mixing › Experts | `residual_moe` | design | `residual_moe` ×1 |
| Channel mixing › Router | `router_scoring` | design | `scoring_func` ×24, `score_function` ×4, `score_func` ×2, `moe_router_activation` ×1, `moe_router_activation_func` ×2, `moe_router_use_sigmoid` ×1, `gate_activation` ×1 |
| Channel mixing › Router | `router_topk_method` | design | `topk_method` ×21, `expert_selection_fn` ×2, `use_grouped_topk` ×2 |
| Channel mixing › Router | `router_num_groups` | scale | `n_group` ×31, `num_expert_group` ×2, `num_expert_groups` ×1 |
| Channel mixing › Router | `router_topk_groups` | scale | `topk_group` ×33, `num_limited_groups` ×1 |
| Channel mixing › Router | `router_normalize_weights` | design | `norm_topk_prob` ×41, `moe_renormalize` ×2, `norm_expert_weight` ×1, `route_norm` ×3, `norm_after_topk` ×1 |
| Channel mixing › Router | `router_scaling_factor` | tuning | `routed_scaling_factor` ×40, `route_scale` ×3, `moe_routed_scaling_factor` ×3, `moe_router_scaling_factor` ×1, `router_scaling_factor` ×1, `use_global_scale` ×1 |
| Channel mixing › Router | `router_expert_bias` | design | `moe_router_enable_expert_bias` ×6, `use_routing_bias` ×4, `use_expert_bias` ×1, `use_moe_router_bias` ×1, `use_gate_bias` ×1 |
| Channel mixing › Router | `router_hidden_size` | scale | `router_hidden_size` ×1 |
| Channel mixing › Router | `router_input_scaling` | design | `scale_router_input` ×1, `moe_apply_router_weight_on_input` ×3, `score_before_experts` ×1 |
| Channel mixing › Router | `router_logit_softcapping` | tuning | `router_logit_softcapping` ×1, `moe_router_logit_softcapping` ×1 |
| Block structure & residual stream | `num_layers` | scale | `num_hidden_layers` ×99, `n_layer` ×1, `num_layers` ×1, `num_blocks` ×1 |
| Block structure & residual stream | `hidden_size` | scale | `hidden_size` ×101, `n_embd` ×1, `embedding_dim` ×1, `block_dim` ×2 |
| Block structure & residual stream | `norm_type` | design | `norm_type` ×2, `normalization_function` ×1, `use_rmsnorm` ×1, `use_gemma_norm` ×1 |
| Block structure & residual stream | `norm_eps` | tuning | `rms_norm_eps` ×89, `layer_norm_eps` ×3, `layer_norm_epsilon` ×7, `norm_eps` ×9, `layernorm_epsilon` ×3, `eps` ×1, `block_norm_eps` ×2 |
| Block structure & residual stream | `norm_eps_post` | tuning | `post_norm_eps` ×1 |
| Block structure & residual stream | `norm_eps_cell` | tuning | `cell_norm_eps` ×1 |
| Block structure & residual stream | `norm_extra_placement` | design | `add_post_norm` ×1, `add_out_norm` ×1, `add_post_blocks_norm` ×1, `use_embed_norm` ×1, `up_proj_norm` ×1, `value_norm` ×1, `skip_loop_final_norm` ×1 |
| Block structure & residual stream | `norm_beta_attention` | tuning | `layernorm_full_attention_beta` ×1 |
| Block structure & residual stream | `norm_beta_linear_attention` | tuning | `layernorm_linear_attention_beta` ×1 |
| Block structure & residual stream | `norm_beta_mlp` | tuning | `layernorm_mlp_beta` ×1 |
| Block structure & residual stream | `norm_zero_centered` | design | `zero_centered` ×1 |
| Block structure & residual stream | `ngpt` | design | `use_nGPT` ×1 |
| Block structure & residual stream | `parallel_block` | design | `use_parallel_block` ×1, `transformer_block_type` ×2 |
| Block structure & residual stream | `hyper_connections` | design | `mhc` ×1, `mhc_enabled` ×1, `enable_ihc` ×1 |
| Block structure & residual stream | `hyper_connection_streams` | scale · presence | `hc_mult` ×5, `hc_count` ×1, `mhc_expansion_rate` ×1 |
| Block structure & residual stream | `hyper_connection_params` | tuning | `hc_eps` ×5, `hc_sinkhorn_iters` ×4, `mhc_sinkhorn_iters` ×1, `hc_lowrank` ×1, `hc_magnitude` ×1 |
| Block structure & residual stream | `attention_residual_block` | scale | `attn_res_block_size` ×1 |
| Block structure & residual stream | `residual_multiplier` | tuning | `residual_multiplier` ×2 |
| Block structure & residual stream | `mup` | design | `mup_enabled` ×1 |
| Block structure & residual stream | `loop_passes` | scale · presence | `num_loops` ×1, `total_ut_steps` ×1 |
| Block structure & residual stream | `loop_exit_threshold` | tuning | `early_exit_threshold` ×1 |
