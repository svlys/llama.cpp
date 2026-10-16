#include "models.h"

// Released checkpoint: video_fusion_layer_freq is 1 only at these layers.
static bool youtu_omni_fusion_layer(int il) {
    return il == 13 || il == 14 || il == 26 || il == 27;
}

ggml_cgraph * clip_graph_youtu_omni::build() {
    const int n_rot = d_head / 2;
    const int per   = n_rot / 3;
    const int rem   = n_rot - per * 3;
    int sections[4] = {per, per, per + rem, 0};
    const float rope_theta = hparams.rope_theta > 0.0f ? hparams.rope_theta : 10000.0f;

    ggml_tensor * freq = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_rot);
    ggml_set_name(freq, "youtu_omni_rope_factors");
    ggml_set_input(freq);

    if (model.modality == CLIP_MODALITY_AUDIO) {
        // Three stride-2 convs over 100-frame chunks, then the shared transformer.
        // Padded tail frames are dropped before attention, matching padded_mask_after_cnn.
        // Each remaining chunk is its own bidirectional segment. RoPE t restarts at 0
        // inside the chunk; h and w stay 0. m_dim is 0, so the M channel is unused.
        GGML_ASSERT(model.conv2d_1_w && model.conv2d_2_w && model.conv2d_3_w);
        GGML_ASSERT(model.conv_out_w);
        GGML_ASSERT(model.mm_input_norm_w && model.mm_merger_fc1_w && model.mm_merger_fc2_w);

        const int64_t n_frames = img.nx();
        const int64_t chunk    = 100;
        GGML_ASSERT(n_frames > 0 && n_frames % chunk == 0);
        const int64_t n_chunks = n_frames / chunk;
        const int n_valid = img.audio_n_frames > 0 ? img.audio_n_frames : (int) n_frames;
        const int n_keep  = clip_youtu_omni_audio_tokens(n_valid);
        GGML_ASSERT(n_keep > 0 && n_keep <= n_chunks * 13);
        const int n_tail = n_keep - (int) (n_chunks - 1) * 13;
        GGML_ASSERT(n_tail > 0 && n_tail <= 13);

        ggml_tensor * inp = build_inp_raw(1);
        inp = ggml_view_4d(ctx0, inp,
            chunk, inp->ne[1], 1, n_chunks,
            n_frames * (int64_t) sizeof(float),
            chunk    * (int64_t) sizeof(float),
            chunk    * (int64_t) sizeof(float),
            0);
        inp = ggml_cont(ctx0, inp);

        auto conv_block = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
            x = ggml_conv_2d(ctx0, w, x, 2, 2, 1, 1, 1, 1);
            if (b) {
                x = ggml_add(ctx0, x, ggml_reshape_4d(ctx0, b, 1, 1, x->ne[2], 1));
            }
            return ggml_gelu_erf(ctx0, x);
        };
        inp = conv_block(inp, model.conv2d_1_w, model.conv2d_1_b);
        inp = conv_block(inp, model.conv2d_2_w, model.conv2d_2_b);
        inp = conv_block(inp, model.conv2d_3_w, model.conv2d_3_b);
        cb(inp, "after_conv_blocks", -1);

        // [OW, OH, OC, n_chunks] -> [OH, OC, OW, n_chunks] -> [OH*OC, OW*n_chunks]
        inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 2, 0, 1, 3));
        inp = ggml_reshape_2d(ctx0, inp, inp->ne[0] * inp->ne[1], inp->ne[2] * inp->ne[3]);
        inp = ggml_mul_mat(ctx0, model.conv_out_w, inp);
        cb(inp, "after_conv_out", -1);

        if (n_keep < n_chunks * 13) {
            auto slice_rows = [&](int64_t row0, int64_t nrows) {
                return ggml_cont(ctx0, ggml_view_2d(ctx0, inp, inp->ne[0], nrows, inp->nb[1], row0 * inp->nb[1]));
            };
            ggml_tensor * kept = nullptr;
            for (int64_t c = 0; c < n_chunks; ++c) {
                const int64_t n_this = (c + 1 == n_chunks) ? n_tail : 13;
                ggml_tensor * part = slice_rows(c * 13, n_this);
                kept = kept ? ggml_concat(ctx0, kept, part, 1) : part;
            }
            inp = kept;
        }
        GGML_ASSERT(inp->ne[1] == n_keep);

        ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_keep * 4);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
            return ggml_rope_multi(
                ctx0, cur, positions, freq,
                n_rot, sections, GGML_ROPE_TYPE_VISION,
                32768, rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        };

        build_vit_opts opts;
        opts.attn_mask = build_inp_attn_mask(n_keep, n_keep);
        ggml_tensor * cur = build_vit(
            inp, n_keep, NORM_TYPE_RMS, FFN_SILU, nullptr, add_pos, opts);

        // temporal_merge_size is 1: RMSNorm over hidden, then a bias-free GELU MLP.
        cur = build_norm(cur, model.mm_input_norm_w, nullptr, NORM_TYPE_RMS, eps, -1);
        cur = build_ffn(cur,
            model.mm_merger_fc1_w, nullptr,
            nullptr, nullptr,
            model.mm_merger_fc2_w, nullptr,
            FFN_GELU_ERF, -1);

        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    // Image path of the shared omni encoder.
    // Patches arrive in raster order from conv2d. The merger consumes 2x2
    // blocks, so tokens are gathered into that order before the transformer.
    // RoPE is chunked MTHW with m_dim = 0: sections [T, H, W] over head_dim/2,
    // each section using its own theta base. ggml_rope_multi(VISION) resets
    // theta per section but scales it by the full n_dims, so a constant
    // freq_factors tensor restores the per-section bases.
    GGML_ASSERT(model.patch_embeddings_0 != nullptr);
    GGML_ASSERT(model.mm_input_norm_w != nullptr);
    GGML_ASSERT(model.mm_merger_fc1_w != nullptr);
    GGML_ASSERT(model.mm_merger_fc2_w != nullptr);
    GGML_ASSERT(hparams.n_merge == 2);
    GGML_ASSERT(n_patches_x % 2 == 0 && n_patches_y % 2 == 0);

    ggml_tensor * inp = build_inp();

    ggml_tensor * order = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(order, "youtu_omni_order");
    ggml_set_input(order);
    inp = ggml_get_rows(ctx0, inp, order);
    inp = ggml_reshape_3d(ctx0, inp, n_embd, n_patches, n_batch);

    // Video: one frame, then that second of audio, in one transformer.
    // Layers 13, 14, 26 and 27 attend across both. Every other layer keeps
    // them in separate segments. A 1 fps group is exactly this pair, and both
    // time cursors restart at 0, so the positions match either mask.
    int n_audio = 0;
    if (img.youtu_joint_mel) {
        GGML_ASSERT(n_batch == 1);
        GGML_ASSERT(model.conv2d_1_w && model.conv2d_2_w && model.conv2d_3_w);
        GGML_ASSERT(model.conv_out_w);
        GGML_ASSERT(model.youtu_a_norm_w && model.youtu_a_fc1_w && model.youtu_a_fc2_w);

        const int64_t n_frames = img.youtu_joint_mel_frames;
        const int64_t n_mel    = img.youtu_joint_mel_bins;
        const int64_t chunk    = 100;
        GGML_ASSERT(n_frames > 0 && n_frames % chunk == 0);
        GGML_ASSERT(n_mel > 0);
        const int n_valid = img.youtu_joint_mel_valid > 0 ? img.youtu_joint_mel_valid : (int) n_frames;
        n_audio = clip_youtu_omni_audio_tokens(n_valid);
        const int64_t n_chunks = n_frames / chunk;
        GGML_ASSERT(n_audio > 0 && n_audio <= n_chunks * 13);
        const int n_tail = n_audio - (int) (n_chunks - 1) * 13;
        GGML_ASSERT(n_tail > 0 && n_tail <= 13);

        ggml_tensor * aud = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_frames, n_mel, 1, 1);
        ggml_set_name(aud, "youtu_joint_mel");
        ggml_set_input(aud);
        aud = ggml_view_4d(ctx0, aud,
            chunk, aud->ne[1], 1, n_chunks,
            n_frames * (int64_t) sizeof(float),
            chunk    * (int64_t) sizeof(float),
            chunk    * (int64_t) sizeof(float),
            0);
        aud = ggml_cont(ctx0, aud);

        auto conv_block = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
            x = ggml_conv_2d(ctx0, w, x, 2, 2, 1, 1, 1, 1);
            if (b) {
                x = ggml_add(ctx0, x, ggml_reshape_4d(ctx0, b, 1, 1, x->ne[2], 1));
            }
            return ggml_gelu_erf(ctx0, x);
        };
        aud = conv_block(aud, model.conv2d_1_w, model.conv2d_1_b);
        aud = conv_block(aud, model.conv2d_2_w, model.conv2d_2_b);
        aud = conv_block(aud, model.conv2d_3_w, model.conv2d_3_b);
        aud = ggml_cont(ctx0, ggml_permute(ctx0, aud, 2, 0, 1, 3));
        aud = ggml_reshape_2d(ctx0, aud, aud->ne[0] * aud->ne[1], aud->ne[2] * aud->ne[3]);
        aud = ggml_mul_mat(ctx0, model.conv_out_w, aud);

        if (n_audio < n_chunks * 13) {
            auto slice_rows = [&](int64_t row0, int64_t nrows) {
                return ggml_cont(ctx0, ggml_view_2d(ctx0, aud, aud->ne[0], nrows, aud->nb[1], row0 * aud->nb[1]));
            };
            ggml_tensor * kept = nullptr;
            for (int64_t c = 0; c < n_chunks; ++c) {
                const int64_t n_this = (c + 1 == n_chunks) ? n_tail : 13;
                ggml_tensor * part = slice_rows(c * 13, n_this);
                kept = kept ? ggml_concat(ctx0, kept, part, 1) : part;
            }
            aud = kept;
        }
        GGML_ASSERT(aud->ne[1] == n_audio);

        inp = ggml_reshape_2d(ctx0, inp, n_embd, n_patches);
        inp = ggml_concat(ctx0, ggml_cont(ctx0, inp), aud, 1);
        inp = ggml_reshape_3d(ctx0, inp, n_embd, n_patches + n_audio, 1);
    }

    const int n_seq = n_patches + n_audio;
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_seq * 4);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return ggml_rope_multi(
            ctx0, cur, positions, freq,
            n_rot, sections, GGML_ROPE_TYPE_VISION,
            32768, rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    };

    build_vit_opts opts;
    if (n_audio > 0) {
        ggml_tensor * mask = build_inp_attn_mask(n_seq, n_seq);
        opts.attn_mask_layers.resize(n_layer);
        for (int il = 0; il < n_layer; il++) {
            opts.attn_mask_layers[il] = youtu_omni_fusion_layer(il) ? nullptr : mask;
        }
    }

    ggml_tensor * cur = build_vit(
        inp, n_seq, NORM_TYPE_RMS, FFN_SILU, nullptr, add_pos, opts);

    const int merge2 = hparams.n_merge * hparams.n_merge;
    if (n_audio == 0) {
        cur = ggml_reshape_2d(ctx0, cur, n_embd * merge2, n_patches / merge2);
        cur = build_norm(cur, model.mm_input_norm_w, nullptr, NORM_TYPE_RMS, eps, -1);
        cur = build_ffn(cur,
            model.mm_merger_fc1_w, nullptr,
            nullptr, nullptr,
            model.mm_merger_fc2_w, nullptr,
            FFN_GELU_ERF, -1);
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq);
    ggml_tensor * vis = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, n_patches, cur->nb[1], 0));
    ggml_tensor * aud = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, n_audio, cur->nb[1], n_patches * cur->nb[1]));

    vis = ggml_reshape_2d(ctx0, vis, n_embd * merge2, n_patches / merge2);
    vis = build_norm(vis, model.mm_input_norm_w, nullptr, NORM_TYPE_RMS, eps, -1);
    vis = build_ffn(vis,
        model.mm_merger_fc1_w, nullptr,
        nullptr, nullptr,
        model.mm_merger_fc2_w, nullptr,
        FFN_GELU_ERF, -1);

    aud = build_norm(aud, model.youtu_a_norm_w, nullptr, NORM_TYPE_RMS, eps, -1);
    aud = build_ffn(aud,
        model.youtu_a_fc1_w, nullptr,
        nullptr, nullptr,
        model.youtu_a_fc2_w, nullptr,
        FFN_GELU_ERF, -1);

    cur = ggml_concat(ctx0, vis, aud, 1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
