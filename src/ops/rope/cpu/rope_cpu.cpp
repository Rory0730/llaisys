#include "rope_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <vector>

// Compute in double, cast back to T
template <typename T>
static inline void rope_impl(T *out,
                             const T *in,
                             const int64_t *pos_ids,
                             double theta, // pass as double
                             size_t seqlen,
                             size_t nhead,
                             size_t d) {
    // d must be even; we rotate first and second halves
    const size_t dh = d / 2;
    ASSERT(d % 2 == 0 && dh > 0, "RoPE: last dimension must be even and >= 2");

    // Use phi = p / pow(theta, 2*i/d) (no negative exponent, better cross-platform)
    const double d_as_double = static_cast<double>(d);

    // Precompute inv = pow(theta, 2*i/d) as double
    std::vector<double> inv(dh);
    for (size_t i = 0; i < dh; ++i) {
        const double expo = (2.0 * static_cast<double>(i)) / d_as_double;
        inv[i] = std::pow(theta, expo);
    }

    for (size_t t = 0; t < seqlen; ++t) {
        const double p = static_cast<double>(pos_ids[t]);
        for (size_t h = 0; h < nhead; ++h) {
            const T *src = in + (t * nhead + h) * d;
            T *dst = out + (t * nhead + h) * d;

            for (size_t i = 0; i < dh; ++i) {
                // read as float, compute as double (bf16/f16 go via cast<float>)
                const double a = static_cast<double>(llaisys::utils::cast<float>(src[i]));
                const double b = static_cast<double>(llaisys::utils::cast<float>(src[i + dh]));

                const double phi = p / inv[i]; // == p * theta^{-(2i/d)}
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

namespace llaisys::ops::cpu {

void rope(std::byte *out,
          const std::byte *in,
          const std::byte *pos_ids,
          float theta_f,
          llaisysDataType_t type,
          size_t seqlen,
          size_t nhead,
          size_t d) {
    ASSERT(d % 2 == 0 && d >= 2, "RoPE: last dimension must be even and >= 2");

    const int64_t *pos = reinterpret_cast<const int64_t *>(pos_ids);
    const double theta = static_cast<double>(theta_f);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_impl(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         pos, theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_impl(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         pos, theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_impl(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         pos, theta, seqlen, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
