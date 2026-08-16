#include "vdrawhelper.h"

#if defined(V_SIMD_SSE2)

#include <cstring>
#include <emmintrin.h> /* for SSE2 intrinsics */
#include <xmmintrin.h> /* for _mm_shuffle_pi16 and _MM_SHUFFLE */

// Each 32bits components of alphaChannel must be in the form 0x00AA00AA
inline static __m128i v4_byte_mul_sse2(__m128i c, __m128i a)
{
    const __m128i ag_mask = _mm_set1_epi32(0xFF00FF00);
    const __m128i rb_mask = _mm_set1_epi32(0x00FF00FF);

    /* for AG */
    __m128i v_ag = _mm_and_si128(ag_mask, c);
    v_ag = _mm_srli_epi32(v_ag, 8);
    v_ag = _mm_mullo_epi16(a, v_ag);
    v_ag = _mm_and_si128(ag_mask, v_ag);

    /* for RB */
    __m128i v_rb = _mm_and_si128(rb_mask, c);
    v_rb = _mm_mullo_epi16(a, v_rb);
    v_rb = _mm_srli_epi32(v_rb, 8);
    v_rb = _mm_and_si128(rb_mask, v_rb);

    /* combine */
    return _mm_add_epi32(v_ag, v_rb);
}

static inline __m128i v4_interpolate_color_sse2(__m128i a, __m128i c0,
                                                __m128i c1)
{
    const __m128i zero = _mm_setzero_si128();
    const __m128i inverse_a = _mm_sub_epi16(_mm_set1_epi16(255), a);
    const __m128i c0_l = _mm_unpacklo_epi8(c0, zero);
    const __m128i c0_h = _mm_unpackhi_epi8(c0, zero);
    const __m128i c1_l = _mm_unpacklo_epi8(c1, zero);
    const __m128i c1_h = _mm_unpackhi_epi8(c1, zero);

    const __m128i result_l = _mm_srli_epi16(
        _mm_add_epi16(_mm_mullo_epi16(c0_l, a),
                      _mm_mullo_epi16(c1_l, inverse_a)),
        8);
    const __m128i result_h = _mm_srli_epi16(
        _mm_add_epi16(_mm_mullo_epi16(c0_h, a),
                      _mm_mullo_epi16(c1_h, inverse_a)),
        8);
    return _mm_packus_epi16(result_l, result_h);
}

// Load src and dest vector
#define V4_FETCH_SRC_DEST                           \
    __m128i v_src = _mm_loadu_si128((__m128i*)src); \
    __m128i v_dest = _mm_load_si128((__m128i*)dest);

#define V4_FETCH_SRC __m128i v_src = _mm_loadu_si128((__m128i*)src);

#define V4_STORE_DEST _mm_store_si128((__m128i*)dest, v_src);

#define V4_SRC_DEST_LEN_INC \
    dest += 4;              \
    src += 4;               \
    length -= 4;

// Multiply src color with const_alpha
#define V4_ALPHA_MULTIPLY v_src = v4_byte_mul_sse2(v_src, v_alpha);


// dest = src + dest * sia
#define V4_COMP_OP_SRC \
    v_src = v4_interpolate_color_sse2(v_alpha, v_src, v_dest);

#define LOOP_ALIGNED_U1_A4(DEST, LENGTH, UOP, A4OP) \
    {                                               \
        while (((uintptr_t)DEST & 0xF) && LENGTH)   \
            UOP                                     \
                                                    \
                while (LENGTH)                      \
            {                                       \
                switch (LENGTH) {                   \
                case 3:                             \
                case 2:                             \
                case 1:                             \
                    UOP break;                      \
                default:                            \
                    A4OP break;                     \
                }                                   \
            }                                       \
    }

void memfill32(uint32_t* dest, uint32_t value, int length)
{
    __m128i vector_data = _mm_set_epi32(value, value, value, value);

    // run till memory alligned to 16byte memory
    while (length && ((uintptr_t)dest & 0xf)) {
        *dest++ = value;
        length--;
    }

    while (length >= 32) {
        _mm_store_si128((__m128i*)(dest), vector_data);
        _mm_store_si128((__m128i*)(dest + 4), vector_data);
        _mm_store_si128((__m128i*)(dest + 8), vector_data);
        _mm_store_si128((__m128i*)(dest + 12), vector_data);
        _mm_store_si128((__m128i*)(dest + 16), vector_data);
        _mm_store_si128((__m128i*)(dest + 20), vector_data);
        _mm_store_si128((__m128i*)(dest + 24), vector_data);
        _mm_store_si128((__m128i*)(dest + 28), vector_data);

        dest += 32;
        length -= 32;
    }

    if (length >= 16) {
        _mm_store_si128((__m128i*)(dest), vector_data);
        _mm_store_si128((__m128i*)(dest + 4), vector_data);
        _mm_store_si128((__m128i*)(dest + 8), vector_data);
        _mm_store_si128((__m128i*)(dest + 12), vector_data);

        dest += 16;
        length -= 16;
    }

    if (length >= 8) {
        _mm_store_si128((__m128i*)(dest), vector_data);
        _mm_store_si128((__m128i*)(dest + 4), vector_data);

        dest += 8;
        length -= 8;
    }

    if (length >= 4) {
        _mm_store_si128((__m128i*)(dest), vector_data);

        dest += 4;
        length -= 4;
    }

    while (length) {
        *dest++ = value;
        length--;
    }
}

// dest = color + (dest * alpha)
inline static void copy_helper_sse2(uint32_t* dest, int length,
                                         uint32_t color, uint32_t alpha)
{
    const __m128i v_color = _mm_set1_epi32(color);
    const __m128i v_a = _mm_set1_epi16(alpha);

    LOOP_ALIGNED_U1_A4(dest, length,
                       { /* UOP */
                         *dest = color + BYTE_MUL(*dest, alpha);
                         dest++;
                         length--;
                       },
                       { /* A4OP */
                         __m128i v_dest = _mm_load_si128((__m128i*)dest);

                         v_dest = v4_byte_mul_sse2(v_dest, v_a);
                         v_dest = _mm_add_epi32(v_dest, v_color);

                         _mm_store_si128((__m128i*)dest, v_dest);

                         dest += 4;
                         length -= 4;
                       })
}

static void color_Source(uint32_t* dest, int length, uint32_t color,
                                 uint32_t const_alpha)
{
    if (const_alpha == 255) {
        memfill32(dest, color, length);
    } else {
        int ialpha;

        ialpha = 255 - const_alpha;
        color = BYTE_MUL(color, const_alpha);
        copy_helper_sse2(dest, length, color, ialpha);
    }
}

static void color_SourceOver(uint32_t* dest, int length,
                                      uint32_t color,
                                     uint32_t const_alpha)
{
    int ialpha;

    if (const_alpha != 255) color = BYTE_MUL(color, const_alpha);
    ialpha = 255 - vAlpha(color);
    copy_helper_sse2(dest, length, color, ialpha);
}

static void src_Source(uint32_t* dest, int length, const uint32_t* src,
                           uint32_t const_alpha)
{
    int ialpha;
    if (const_alpha == 255) {
        memcpy(dest, src, length * sizeof(uint32_t));
    } else {
        ialpha = 255 - const_alpha;
        __m128i v_alpha = _mm_set1_epi16(const_alpha);

        LOOP_ALIGNED_U1_A4(dest, length,
                           { /* UOP */
                             *dest = interpolate_pixel(*src, const_alpha,
                                                       *dest, ialpha);
                             dest++;
                             src++;
                             length--;
                           },
                           {/* A4OP */
                            V4_FETCH_SRC_DEST V4_COMP_OP_SRC V4_STORE_DEST
                                                             V4_SRC_DEST_LEN_INC})
    }
}

static inline __m128i v4_byte_mul_alpha_sse2(__m128i color,
                                             __m128i alpha)
{
    const __m128i rb_mask = _mm_set1_epi32(0x00ff00ff);
    const __m128i ag_mask = _mm_set1_epi32(0xff00ff00);
    const __m128i alpha_pair =
        _mm_or_si128(alpha, _mm_slli_epi32(alpha, 16));
    const __m128i ag = _mm_and_si128(
        _mm_mullo_epi16(_mm_and_si128(_mm_srli_epi32(color, 8), rb_mask),
                        alpha_pair),
        ag_mask);
    const __m128i rb = _mm_and_si128(
        _mm_srli_epi32(
            _mm_mullo_epi16(_mm_and_si128(color, rb_mask), alpha_pair), 8),
        rb_mask);
    return _mm_add_epi32(ag, rb);
}

static void src_SourceOver(uint32_t* dest, int length, const uint32_t* src,
                           uint32_t const_alpha)
{
    const __m128i zero = _mm_setzero_si128();
    const __m128i all_255 = _mm_set1_epi32(255);
    const __m128i const_alpha_vector = _mm_set1_epi32(const_alpha);
    while (length >= 4) {
        const __m128i source = _mm_loadu_si128((const __m128i*)src);
        const __m128i destination =
            _mm_loadu_si128((const __m128i*)dest);
        __m128i scaled_source = source;

        if (const_alpha != 255)
            scaled_source = v4_byte_mul_alpha_sse2(
                source, const_alpha_vector);

        const __m128i inverse_alpha = _mm_sub_epi32(
            all_255, _mm_srli_epi32(scaled_source, 24));
        __m128i result = _mm_add_epi32(
            scaled_source,
            v4_byte_mul_alpha_sse2(destination, inverse_alpha));

        if (const_alpha == 255) {
            // The scalar full-opacity path leaves the destination unchanged
            // when the source pixel is completely transparent.
            const __m128i transparent = _mm_cmpeq_epi32(source, zero);
            result = _mm_or_si128(_mm_and_si128(transparent, destination),
                                  _mm_andnot_si128(transparent, result));
        }

        _mm_storeu_si128((__m128i*)dest, result);
        src += 4;
        dest += 4;
        length -= 4;
    }

    if (const_alpha == 255) {
        while (length-- > 0) {
            const uint32_t source = *src++;
            if (source >= 0xff000000u) {
                *dest = source;
            } else if (source != 0) {
                *dest = source + BYTE_MUL(*dest, vAlpha(~source));
            }
            ++dest;
        }
    } else {
        while (length-- > 0) {
            const uint32_t sourcePixel = *src++;
            const uint32_t source = BYTE_MUL(sourcePixel, const_alpha);
            *dest = source + BYTE_MUL(*dest, vAlpha(~source));
            ++dest;
        }
    }
}

void RenderFuncTable::sse()
{
    updateColor(BlendMode::Src , color_Source);
    updateColor(BlendMode::SrcOver , color_SourceOver);

    updateSrc(BlendMode::Src, src_Source);
    updateSrc(BlendMode::SrcOver, src_SourceOver);
}

#endif
