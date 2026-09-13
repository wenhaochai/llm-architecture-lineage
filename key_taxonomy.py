"""Hand classification of every key found in the 103 text configs.

Four groups are dropped because they carry no architecture information:
  GENERATION  decoding defaults, tokenizer ids, Hugging Face bookkeeping, output flags
  RUNTIME     dtype, kernels, parallelism, inference caching, implementation switches
  TRAINING    initialisation, dropout, auxiliary losses, loss weights
  VISION      multimodal wrapper leftovers
Everything else is architecture and stays.
"""

GENERATION = {
    "_name_or_path", "_provenance", "add_cross_attention", "auto_map", "bad_words_ids", "begin_suppress_tokens", "bos_token_id",
    "chunk_size_feed_forward", "cross_attention_hidden_size", "decoder_start_token_id", "diversity_penalty", "do_sample", "early_stopping",
    "encoder_no_repeat_ngram_size", "eod_token_id", "eos_token_id", "exponential_decay_length_penalty", "finetuning_task", "force_bos_token_insert",
    "forced_bos_token_id", "forced_eos_token_id", "id2label", "is_decoder", "is_encoder_decoder", "label2id", "length_penalty", "max_length",
    "min_length", "no_repeat_ngram_size", "num_beam_groups", "num_beams", "num_return_sequences", "output_attentions",
    "output_hidden_states", "output_past", "output_router_logits", "output_scores", "pad_token_id", "prefix", "problem_type", "pruned_heads",
    "remove_invalid_values", "repetition_penalty", "return_dict", "return_dict_in_generate", "sep_token_id", "suppress_tokens",
    "task_specific_params", "temperature", "tf_legacy_loss", "tie_encoder_decoder", "tokenizer_class", "top_k", "top_p", "torchscript",
    "typical_p", "use_bfloat16", "transformers_version", "transformers.js_config", "unsloth_fixed", "unsloth_version",
    "summary_activation", "summary_first_dropout", "summary_proj_to_labels", "summary_type", "summary_use_proj",
    "dspark_noise_token_id", "engram_pad_token_id",
}
RUNTIME = {
    "_attn_implementation", "attn_implementation", "attention_projection_layout", "autocast_kernel_dtype", "bitwise_backward_align",
    "cache_implementation", "chunk_size", "mamba_chunk_size", "chunkwise_kernel", "dtype", "torch_dtype", "enable_attention_fp32_softmax",
    "enable_lm_head_fp32", "enable_moe_fp32_combine", "ep_size", "expert_dtype", "for_llm_compressor", "inference_state_dtype",
    "mamba_ssm_cache_dtype", "mamba_ssm_dtype", "mode", "moe_router_dtype", "router_dtype", "need_fp32_gate", "norm_reduction_force_float32",
    "num_logits_to_keep", "pretraining_tp", "residual_in_fp32", "return_last_states", "sequence_kernel", "step_kernel", "use_cache",
    "use_grouped_mm", "use_mamba_kernels", "use_parallel_embedding", "weight_mode", "_debug_force_load_balance", "quantization_config",
    "add_forward_backend_padding", "moe_shared_expert_overlap",
}
TRAINING = {
    "initializer_range", "init_method", "attention_dropout", "hidden_dropout", "embedding_dropout", "output_dropout", "attn_pdrop",
    "embd_pdrop", "resid_pdrop", "add_embedding_dropout", "router_aux_loss_coef", "aux_loss_alpha", "load_balance_coeff", "seq_aux",
    "router_jitter_noise", "mtp_loss_scaling_factor", "mtp_loss_factor", "loop_loss_weights", "igate_bias_init_range", "mhc_identity_init",
    "learnable_sink_init", "block_mlp_init_scale", "block_out_init_scale", "block_use_xavier_init", "conv_use_xavier_init",
    "rescale_prenorm_residual",
}
VISION = {
    "audio_config", "vision_config", "processor_config", "vision_model_type", "image_token_id", "video_token_id",
    "vision_start_token_id", "vision_end_token_id",
}
IDENTITY = {"model_type", "architectures"}          # name the modeling code, not the architecture
TRAINING_HEADS = {"num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_modules", "mtp_transformer_layers", "use_mtp", "mtp",
                  "mtp_layers_block_type", "mtp_hybrid_override_pattern", "mtp_use_dedicated_embeddings", "mtp_use_kda"}  # multi-token-prediction heads
DROP = GENERATION | RUNTIME | TRAINING | VISION | IDENTITY | TRAINING_HEADS
