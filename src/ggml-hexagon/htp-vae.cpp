#include "htp-vae.h"
#include "htp/conv2d-ops.h"
#include "htp/groupnorm-ops.h"

#include <algorithm>

static bool conv2d_layout(
        const struct ggml_hexagon_session * sess,
        const struct ggml_tensor * op,
        struct htp_conv2d_kernel_params * kparams) {
    const bool causal3d = op->op == GGML_OP_CONV_3D_CAUSAL;
    const uint32_t kh = (uint32_t) op->src[0]->ne[1];
    const uint32_t kw = (uint32_t) op->src[0]->ne[0];
    const uint32_t kd = causal3d ? (uint32_t) op->src[0]->ne[2] : 1u;
    const uint32_t ic = causal3d ? (uint32_t) ggml_get_op_params_i32(op, 6) :
                                   (uint32_t) op->src[0]->ne[2];
    const uint32_t oc = causal3d ? (uint32_t) (op->src[0]->ne[3] / ic) :
                                   (uint32_t) op->src[0]->ne[3];
    const uint32_t ic_padded = causal3d ? (ic + 31u) & ~31u : ic;
    const uint32_t ow = (uint32_t) op->ne[0];
    const uint32_t oh = (uint32_t) op->ne[1];
    const uint32_t input_element_size = (uint32_t) ggml_type_size(op->src[1]->type);
    const uint32_t padded_ow =
        (ow + HTP_CONV2D_TILE_W - 1u) & ~(HTP_CONV2D_TILE_W - 1u);

    const size_t vtcm_total = htp_session_vtcm_size(sess);
    const size_t bias_bytes = op->src[2] && opt_dma64 ? (size_t) oc * sizeof(float) : 0;
    if (vtcm_total <= bias_bytes) return false;
    const size_t vtcm_size = vtcm_total - bias_bytes;

    auto build_layout = [&](struct htp_conv2d_kernel_params * result,
                            uint32_t n_tiles_per_block,
                            uint32_t tile_w, uint32_t tile_h) {
        htp_conv2d_layout_build(result, kh, kw, ic_padded, oc,
                                n_tiles_per_block, tile_w, tile_h,
                                htp_session_n_threads(sess), input_element_size,
                                causal3d && opt_dma64 ? htp_session_n_threads(sess) : 0);
        result->kd = kd;
        result->ic = ic;
        result->ic_padded = ic_padded;
        if (causal3d) {
            result->flags |= HTP_CONV2D_CAUSAL3D;
        }
        return result->vtcm_size;
    };

    auto max_tile_rows = [&](uint32_t n_tiles_per_block, uint32_t tile_w,
                             struct htp_conv2d_kernel_params * result) {
        struct htp_conv2d_kernel_params candidate;
        if (build_layout(&candidate, n_tiles_per_block, tile_w, 1) > vtcm_size) {
            return 0u;
        }

        uint32_t lo = 1;
        uint32_t hi = oh;
        while (lo < hi) {
            const uint32_t mid = lo + (hi - lo + 1) / 2;
            if (build_layout(&candidate, n_tiles_per_block, tile_w, mid) <= vtcm_size) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        build_layout(result, n_tiles_per_block, tile_w, lo);
        return lo;
    };

    auto select_layout = [&](uint32_t n_tiles_per_block,
                             struct htp_conv2d_kernel_params * result) {
        uint32_t max_m_tiles = 0;
        for (uint32_t tile_w = HTP_CONV2D_TILE_W; tile_w <= padded_ow; tile_w += HTP_CONV2D_TILE_W) {
            struct htp_conv2d_kernel_params candidate;
            const uint32_t rows = max_tile_rows(n_tiles_per_block, tile_w, &candidate);
            max_m_tiles = std::max(max_m_tiles, rows * (tile_w / HTP_CONV2D_TILE_W));
        }
        if (max_m_tiles == 0) {
            return false;
        }

        bool selected = false;
        for (uint32_t tile_w = HTP_CONV2D_TILE_W; tile_w <= padded_ow; tile_w += HTP_CONV2D_TILE_W) {
            struct htp_conv2d_kernel_params candidate;
            const uint32_t rows = max_tile_rows(n_tiles_per_block, tile_w, &candidate);
            const uint32_t m_tiles = rows * (tile_w / HTP_CONV2D_TILE_W);
            const bool enough_m_tiles =
                kh * kw == 1 ? (uint64_t) m_tiles * 5 >= (uint64_t) max_m_tiles * 4 :
                               (uint64_t) m_tiles * 20 >= (uint64_t) max_m_tiles * 19;
            if (enough_m_tiles) {
                *result = candidate;
                selected = true;
            }
        }
        return selected;
    };

    const uint32_t total_n_tiles = (oc + 31u) / 32u;
    bool found = select_layout(total_n_tiles, kparams);
    if (!found) {
        uint64_t best_resident_tiles = 0;
        uint32_t best_n_tiles = 0;
        for (uint32_t n_tiles = 1; n_tiles < total_n_tiles; ++n_tiles) {
            struct htp_conv2d_kernel_params candidate;
            if (!select_layout(n_tiles, &candidate)) {
                continue;
            }
            const uint64_t resident_tiles = (uint64_t) n_tiles * candidate.m_tiles;
            if (resident_tiles > best_resident_tiles ||
                (resident_tiles == best_resident_tiles && n_tiles > best_n_tiles)) {
                *kparams = candidate;
                best_resident_tiles = resident_tiles;
                best_n_tiles = n_tiles;
                found = true;
            }
        }
    }
    if (found) {
        if (op->op == GGML_OP_CONV_2D_UPSCALE) {
            kparams->flags |= HTP_CONV2D_UPSCALE2;
            if (op->src[3] != nullptr) {
                kparams->flags |= HTP_CONV2D_DUP_UP_RESIDUAL;
                kparams->residual_factor_t = ggml_get_op_params_i32(op, 7);
            }
        }
        if (op->op == GGML_OP_CONV_2D_BIAS || op->src[2] != nullptr) {
            kparams->flags |= HTP_CONV2D_BIAS;
        }
        HEX_VERBOSE("ggml-hex: conv2d tile %ux%u (%u M tiles, %u/%u N tiles), VTCM %u/%zu bytes\n",
                    kparams->tile_w, kparams->tile_h, kparams->m_tiles,
                    kparams->n_tiles_per_block, kparams->n_tiles,
                    kparams->vtcm_size, vtcm_size);
    }
    return found;
}

static bool supports_conv2d(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op, struct htp_conv2d_kernel_params * params) {
    const struct ggml_tensor * weight = op->src[0];
    const struct ggml_tensor * src = op->src[1];
    const struct ggml_tensor * bias = op->src[2];
    const struct ggml_tensor * residual = op->src[3];
    if (htp_session_n_hmx(sess) == 0 || !weight || !src || weight->type != GGML_TYPE_F16 ||
        (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) ||
        (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) ||
        (src->type == GGML_TYPE_F16 && op->type != GGML_TYPE_F16)) {
        return false;
    }
    if (!ggml_is_contiguous(weight) || !ggml_is_contiguous(src) || !ggml_is_contiguous(op) ||
        src->ne[3] != op->ne[3] || (residual && src->ne[3] != 1)) {
        return false;
    }
    if ((op->op == GGML_OP_CONV_2D_BIAS && !bias) ||
        (bias && (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
                  ggml_nelements(bias) != weight->ne[3]))) {
        return false;
    }

    const int32_t s0 = ggml_get_op_params_i32(op, 0);
    const int32_t s1 = ggml_get_op_params_i32(op, 1);
    const int32_t p0 = ggml_get_op_params_i32(op, 2);
    const int32_t p1 = ggml_get_op_params_i32(op, 3);
    const int32_t d0 = ggml_get_op_params_i32(op, 4);
    const int32_t d1 = ggml_get_op_params_i32(op, 5);
    const int32_t upscale = op->op == GGML_OP_CONV_2D_UPSCALE
                                ? ggml_get_op_params_i32(op, 6)
                                : 1;
    const int32_t residual_factor_t = residual ? ggml_get_op_params_i32(op, 7) : 0;
    const int64_t kw = weight->ne[0];
    const int64_t kh = weight->ne[1];
    const int64_t ic = weight->ne[2];
    const int64_t oc = weight->ne[3];
    if (!((kw == 1 && kh == 1) || (kw == 3 && kh == 3)) ||
        s0 != 1 || s1 != 1 || d0 != 1 || d1 != 1 ||
        (op->op == GGML_OP_CONV_2D_UPSCALE && upscale != 2) ||
        p0 != (kw - 1) / 2 || p1 != (kh - 1) / 2 ||
        ic % 32 != 0 ||
        src->ne[2] != ic || op->ne[2] != oc ||
        op->ne[0] != src->ne[0] * upscale ||
        op->ne[1] != src->ne[1] * upscale) {
        return false;
    }
    if (residual &&
        (op->op != GGML_OP_CONV_2D_UPSCALE || upscale != 2 ||
         src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
         residual->type != GGML_TYPE_F32 || !ggml_is_contiguous(residual) ||
         residual->ne[0] != src->ne[0] || residual->ne[1] != src->ne[1] ||
         residual->ne[2] != 1 || residual->ne[3] % oc != 0 ||
         !((residual_factor_t == 2 &&
            (residual->ne[3] == oc || residual->ne[3] == 2 * oc)) ||
           (residual_factor_t == 1 && residual->ne[3] == 2 * oc)))) {
        return false;
    }

    return conv2d_layout(sess, op, params);
}

static bool supports_conv3d_causal(
        const struct ggml_hexagon_session * sess,
        const struct ggml_tensor * op, struct htp_conv2d_kernel_params * params) {
    const struct ggml_tensor * weight = op->src[0];
    const struct ggml_tensor * src = op->src[1];
    const struct ggml_tensor * bias = op->src[2];
    if (htp_session_n_hmx(sess) == 0 || !weight || !src || weight->type != GGML_TYPE_F16 ||
        (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) ||
        (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) ||
        (src->type == GGML_TYPE_F16 && op->type != GGML_TYPE_F16) ||
        !ggml_is_contiguous(weight) || !ggml_is_contiguous(src) ||
        !ggml_is_contiguous(op)) {
        return false;
    }

    const int64_t kw = weight->ne[0];
    const int64_t kh = weight->ne[1];
    const int64_t kd = weight->ne[2];
    const int64_t ic = ggml_get_op_params_i32(op, 6);
    if (ic <= 0 || weight->ne[3] % ic != 0) {
        return false;
    }
    const int64_t oc = weight->ne[3] / ic;
    const int32_t s0 = ggml_get_op_params_i32(op, 0);
    const int32_t s1 = ggml_get_op_params_i32(op, 1);
    const int32_t p0 = ggml_get_op_params_i32(op, 2);
    const int32_t p1 = ggml_get_op_params_i32(op, 3);
    const int32_t d0 = ggml_get_op_params_i32(op, 4);
    const int32_t d1 = ggml_get_op_params_i32(op, 5);
    if (!((kw == 1 && kh == 1) || (kw == 3 && kh == 3)) ||
        (kd != 1 && kd != 3) || s0 != 1 || s1 != 1 || d0 != 1 || d1 != 1 ||
        p0 != (kw - 1) / 2 || p1 != (kh - 1) / 2 || (ic & 1) != 0 ||
        src->ne[2] != 1 || src->ne[3] != ic ||
        op->ne[0] != src->ne[0] || op->ne[1] != src->ne[1] ||
        op->ne[2] != 1 || op->ne[3] != oc ||
        (bias && (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
                  ggml_nelements(bias) != oc))) {
        return false;
    }

    return conv2d_layout(sess, op, params) && ic % params->activation_group_channels == 0;
}

static bool supports_rms_norm_mul_silu(const struct ggml_tensor * op) {
    const struct ggml_tensor * src = op->src[0];
    const struct ggml_tensor * weight = op->src[1];
    return src && weight && src->type == GGML_TYPE_F32 &&
           weight->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           ggml_get_op_params_i32(op, 0) == 3 &&
           ggml_is_contiguous(src) && ggml_is_contiguous(weight) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(src, op) &&
           ggml_nelements(weight) == src->ne[3];
}

static bool supports_group_norm(const struct ggml_tensor * op) {
    const struct ggml_tensor * src = op->src[0];
    if (!src || !((src->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ||
                  (src->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16)) ||
        !ggml_is_contiguous(src) || !ggml_is_contiguous(op) ||
        !ggml_are_same_shape(src, op)) {
        return false;
    }
    const int32_t groups = ggml_get_op_params_i32(op, 0);
    return groups > 0 && src->ne[2] % groups == 0 &&
           (src->type == GGML_TYPE_F16 ||
            (src->ne[0] * src->ne[1]) % 32 == 0);
}

static bool supports_group_norm_affine_silu(const struct ggml_tensor * op) {
    const struct ggml_tensor * src    = op->src[0];
    const struct ggml_tensor * weight = op->src[1];
    const struct ggml_tensor * bias   = op->src[2];
    if (!src || !weight || !bias ||
        !((src->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ||
          (src->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16)) ||
        weight->type != GGML_TYPE_F32 ||
        bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(src) ||
        !ggml_is_contiguous(op) || !ggml_is_contiguous(weight) ||
        !ggml_is_contiguous(bias) || !ggml_are_same_shape(src, op) ||
        ggml_nelements(weight) != src->ne[2] ||
        ggml_nelements(bias) != src->ne[2]) {
        return false;
    }

    const int32_t groups = ggml_get_op_params_i32(op, 0);
    return groups > 0 && src->ne[2] % groups == 0 &&
           (src->type == GGML_TYPE_F16 ||
            (src->ne[0] * src->ne[1]) % 32 == 0);
}

bool htp_kparams_solve_vae(const ggml_hexagon_session * sess, const ggml_tensor * op, void * kparams) {
    switch (op->op) {
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_2D_BIAS:
        case GGML_OP_CONV_2D_UPSCALE:
        case GGML_OP_CONV_3D_CAUSAL: {
            htp_conv2d_kernel_params temp{};
            auto * params = kparams ? static_cast<htp_conv2d_kernel_params *>(kparams) : &temp;
            return op->op == GGML_OP_CONV_3D_CAUSAL ? supports_conv3d_causal(sess, op, params) :
                                                       supports_conv2d(sess, op, params);
        }
        case GGML_OP_GROUP_NORM:
        case GGML_OP_GROUP_NORM_AFFINE_SILU: {
            const bool affine = op->op == GGML_OP_GROUP_NORM_AFFINE_SILU;
            if (kparams) static_cast<htp_group_norm_kernel_params *>(kparams)->flags =
                affine ? HTP_GROUP_NORM_AFFINE | HTP_GROUP_NORM_SILU : 0;
            return affine ? supports_group_norm_affine_silu(op) : supports_group_norm(op);
        }
        case GGML_OP_RMS_NORM_MUL_SILU:
            return supports_rms_norm_mul_silu(op);
        default:
            return false;
    }
}
