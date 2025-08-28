#include "rope_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <vector>

namespace {

// Use double inside for stability, then cast to T on store.
template <typename T>
inline void rope_impl(T *out,
                      const T *in,
                      const int64_t *pos_ids,
                      size_t seqlen, size_t nhead, size_t d,
                      float theta_f) {
    // d must be even
    ASSERT(d % 2 == 0, "RoPE: head dimension must be even.");
    const size_t dh = d / 2;

    const double theta = static_cast<double>(theta_f);

    // inv_freq[i] = 1.0 / pow(theta, i/(d/2))   (use double for stability)
    std::vector<double> inv_freq(dh);
    for (size_t i = 0; i < dh; ++i) {
        inv_freq[i] = 1.0 / std::pow(theta, static_cast<double>(i) / static_cast<double>(dh));
    }

    for (size_t t = 0; t < seqlen; ++t) {
        const double p = static_cast<double>(pos_ids[t]);
        for (size_t h = 0; h < nhead; ++h) {
            const T *src = in + (t * nhead + h) * d;
            T *dst = out + (t * nhead + h) * d;

            // split [a | b], each length dh
            for (size_t i = 0; i < dh; ++i) {
                // read as float, compute in double
                const double a = static_cast<double>(llaisys::utils::cast<float>(src[i]));
                const double b = static_cast<double>(llaisys::utils::cast<float>(src[i + dh]));

                const double phi = p * inv_freq[i];
                const double c = std::cos(phi);
                const double s = std::sin(phi);

                const double ap = a * c - b * s;
                const double bp = b * c + a * s;

                dst[i] = llaisys::utils::cast<T>(static_cast<float>(ap));
                dst[i + dh] = llaisys::utils::cast<T>(static_cast<float>(bp));
            }
        }
    }
}

} // anonymous namespace

namespace llaisys::ops::cpu {

void rope(std::byte *out,
          const std::byte *in,
          const int64_t *pos,
          llaisysDataType_t type,
          size_t seqlen, size_t nhead, size_t d,
          float theta) {
    ASSERT(d % 2 == 0 && d >= 2, "RoPE: last dimension must be even and >= 2");

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_impl(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         pos, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_impl(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         pos, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_impl(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         pos, seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
