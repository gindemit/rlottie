#include "vdrawhelper.h"

#if defined(V_SIMD_NEON)

#include <arm_neon.h>
#include <cstring>

static inline uint8x16_t byte_mul_neon(uint8x16_t color, uint8_t alpha)
{
    const uint8x8_t alphaVector = vdup_n_u8(alpha);
    const uint16x8_t low = vmull_u8(vget_low_u8(color), alphaVector);
    const uint16x8_t high = vmull_u8(vget_high_u8(color), alphaVector);
    return vcombine_u8(vshrn_n_u16(low, 8), vshrn_n_u16(high, 8));
}

static inline uint8x16_t byte_mul_neon(uint8x16_t color,
                                       uint8x16_t alpha)
{
    const uint16x8_t low =
        vmull_u8(vget_low_u8(color), vget_low_u8(alpha));
    const uint16x8_t high =
        vmull_u8(vget_high_u8(color), vget_high_u8(alpha));
    return vcombine_u8(vshrn_n_u16(low, 8), vshrn_n_u16(high, 8));
}

static inline uint8x16_t interpolate_neon(uint8x16_t src, uint8_t srcAlpha,
                                         uint8x16_t dest, uint8_t destAlpha)
{
    const uint8x8_t srcAlphaVector = vdup_n_u8(srcAlpha);
    const uint8x8_t destAlphaVector = vdup_n_u8(destAlpha);
    const uint16x8_t low =
        vmlal_u8(vmull_u8(vget_low_u8(src), srcAlphaVector),
                 vget_low_u8(dest), destAlphaVector);
    const uint16x8_t high =
        vmlal_u8(vmull_u8(vget_high_u8(src), srcAlphaVector),
                 vget_high_u8(dest), destAlphaVector);
    return vcombine_u8(vshrn_n_u16(low, 8), vshrn_n_u16(high, 8));
}

static inline uint8x16_t alpha_bytes_neon(uint32x4_t color)
{
    const uint32x4_t alpha = vshrq_n_u32(color, 24);
    return vreinterpretq_u8_u32(vmulq_n_u32(alpha, 0x01010101u));
}

void memfill32(uint32_t *dest, uint32_t value, int length)
{
    const uint32x4_t values = vdupq_n_u32(value);
    while (length >= 4) {
        vst1q_u32(dest, values);
        dest += 4;
        length -= 4;
    }
    while (length-- > 0) *dest++ = value;
}

static inline void color_blend_neon(uint32_t *dest, int length,
                                    uint32_t color, uint8_t destAlpha)
{
    const uint32x4_t colors = vdupq_n_u32(color);
    while (length >= 4) {
        const uint8x16_t destBytes =
            vreinterpretq_u8_u32(vld1q_u32(dest));
        const uint32x4_t result = vaddq_u32(
            colors,
            vreinterpretq_u32_u8(byte_mul_neon(destBytes, destAlpha)));
        vst1q_u32(dest, result);
        dest += 4;
        length -= 4;
    }
    while (length-- > 0) {
        *dest = color + BYTE_MUL(*dest, destAlpha);
        ++dest;
    }
}

static void color_Source(uint32_t *dest, int length, uint32_t color,
                         uint32_t constAlpha)
{
    if (constAlpha == 255) {
        memfill32(dest, color, length);
        return;
    }

    color = BYTE_MUL(color, constAlpha);
    color_blend_neon(dest, length, color, uint8_t(255 - constAlpha));
}

static void color_SourceOver(uint32_t *dest, int length, uint32_t color,
                             uint32_t constAlpha)
{
    if (constAlpha != 255) color = BYTE_MUL(color, constAlpha);
    color_blend_neon(dest, length, color, uint8_t(255 - vAlpha(color)));
}

static void src_Source(uint32_t *dest, int length, const uint32_t *src,
                       uint32_t constAlpha)
{
    if (constAlpha == 255) {
        std::memcpy(dest, src, size_t(length) * sizeof(uint32_t));
        return;
    }

    const uint8_t srcAlpha = uint8_t(constAlpha);
    const uint8_t destAlpha = uint8_t(255 - constAlpha);
    while (length >= 4) {
        const uint8x16_t srcBytes = vreinterpretq_u8_u32(vld1q_u32(src));
        const uint8x16_t destBytes = vreinterpretq_u8_u32(vld1q_u32(dest));
        vst1q_u32(dest, vreinterpretq_u32_u8(interpolate_neon(
                            srcBytes, srcAlpha, destBytes, destAlpha)));
        src += 4;
        dest += 4;
        length -= 4;
    }
    while (length-- > 0) {
        *dest = interpolate_pixel(*src, constAlpha, *dest, 255 - constAlpha);
        ++src;
        ++dest;
    }
}

static inline uint32x4_t source_over_full_neon(uint32x4_t source,
                                               uint32x4_t dest)
{
    const uint8x16_t inverseAlpha =
        vmvnq_u8(alpha_bytes_neon(source));
    uint32x4_t result = vaddq_u32(
        source,
        vreinterpretq_u32_u8(byte_mul_neon(
            vreinterpretq_u8_u32(dest), inverseAlpha)));

    // The scalar full-opacity path deliberately leaves a destination pixel
    // untouched for a completely transparent source pixel. An alpha-zero
    // source with nonzero RGB is not equivalent and must still be blended.
    const uint32x4_t transparent = vceqq_u32(source, vdupq_n_u32(0));
    return vbslq_u32(transparent, dest, result);
}

static inline uint32x4_t source_over_partial_neon(uint32x4_t source,
                                                  uint32x4_t dest,
                                                  uint8_t constAlpha)
{
    const uint8x16_t scaledBytes =
        byte_mul_neon(vreinterpretq_u8_u32(source), constAlpha);
    const uint32x4_t scaledSource = vreinterpretq_u32_u8(scaledBytes);
    const uint8x16_t inverseAlpha =
        vmvnq_u8(alpha_bytes_neon(scaledSource));
    return vaddq_u32(
        scaledSource,
        vreinterpretq_u32_u8(byte_mul_neon(
            vreinterpretq_u8_u32(dest), inverseAlpha)));
}

static void src_SourceOverFull(uint32_t *dest, int length,
                               const uint32_t *src)
{
    // Two independent vectors give out-of-order AArch64 cores enough work to
    // overlap the multiply/shift chains without increasing the tail cost.
    while (length >= 8) {
        const uint32x4_t source0 = vld1q_u32(src);
        const uint32x4_t source1 = vld1q_u32(src + 4);
        const uint32x4_t dest0 = vld1q_u32(dest);
        const uint32x4_t dest1 = vld1q_u32(dest + 4);
        vst1q_u32(dest, source_over_full_neon(source0, dest0));
        vst1q_u32(dest + 4, source_over_full_neon(source1, dest1));
        src += 8;
        dest += 8;
        length -= 8;
    }
    while (length >= 4) {
        const uint32x4_t source = vld1q_u32(src);
        const uint32x4_t destPixels = vld1q_u32(dest);
        vst1q_u32(dest, source_over_full_neon(source, destPixels));
        src += 4;
        dest += 4;
        length -= 4;
    }
    while (length-- > 0) {
        const uint32_t source = *src++;
        if (source >= 0xff000000u) {
            *dest = source;
        } else if (source != 0) {
            *dest = source + BYTE_MUL(*dest, vAlpha(~source));
        }
        ++dest;
    }
}

static void src_SourceOverPartial(uint32_t *dest, int length,
                                  const uint32_t *src, uint8_t constAlpha)
{
    while (length >= 8) {
        const uint32x4_t source0 = vld1q_u32(src);
        const uint32x4_t source1 = vld1q_u32(src + 4);
        const uint32x4_t dest0 = vld1q_u32(dest);
        const uint32x4_t dest1 = vld1q_u32(dest + 4);
        vst1q_u32(dest,
                  source_over_partial_neon(source0, dest0, constAlpha));
        vst1q_u32(dest + 4,
                  source_over_partial_neon(source1, dest1, constAlpha));
        src += 8;
        dest += 8;
        length -= 8;
    }
    while (length >= 4) {
        const uint32x4_t source = vld1q_u32(src);
        const uint32x4_t destPixels = vld1q_u32(dest);
        vst1q_u32(dest,
                  source_over_partial_neon(source, destPixels, constAlpha));
        src += 4;
        dest += 4;
        length -= 4;
    }
    while (length-- > 0) {
        const uint32_t source = BYTE_MUL(*src, constAlpha);
        ++src;
        *dest = source + BYTE_MUL(*dest, vAlpha(~source));
        ++dest;
    }
}

static void src_SourceOver(uint32_t *dest, int length, const uint32_t *src,
                           uint32_t constAlpha)
{
    if (constAlpha == 255) {
        src_SourceOverFull(dest, length, src);
    } else {
        src_SourceOverPartial(dest, length, src, uint8_t(constAlpha));
    }
}

void RenderFuncTable::neon()
{
    updateColor(BlendMode::Src, color_Source);
    updateColor(BlendMode::SrcOver, color_SourceOver);
    updateSrc(BlendMode::Src, src_Source);
    updateSrc(BlendMode::SrcOver, src_SourceOver);
}

#endif
