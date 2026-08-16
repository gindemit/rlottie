#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vdrawhelper.h"

#if !defined(V_SIMD_SSE2) && !defined(V_SIMD_NEON)
void memfill32(uint32_t *dest, uint32_t value, int length)
{
    while (length-- > 0) *dest++ = value;
}
#endif

namespace {

constexpr std::array<uint32_t, 5> kAlphas{{0, 1, 127, 254, 255}};

uint32_t nextRandom(uint32_t &state)
{
    state = state * 1664525u + 1013904223u;
    return state;
}

void referenceColor(BlendMode mode, uint32_t *dest, int length,
                    uint32_t color, uint32_t alpha)
{
    if (mode == BlendMode::Src) {
        if (alpha == 255) {
            while (length-- > 0) *dest++ = color;
            return;
        }
        color = BYTE_MUL(color, alpha);
        const uint32_t inverseAlpha = 255 - alpha;
        while (length-- > 0) {
            *dest = color + BYTE_MUL(*dest, inverseAlpha);
            ++dest;
        }
        return;
    }

    if (alpha != 255) color = BYTE_MUL(color, alpha);
    const uint32_t inverseAlpha = 255 - vAlpha(color);
    while (length-- > 0) {
        *dest = color + BYTE_MUL(*dest, inverseAlpha);
        ++dest;
    }
}

void referenceSrc(BlendMode mode, uint32_t *dest, int length,
                  const uint32_t *src, uint32_t alpha)
{
    if (mode == BlendMode::Src) {
        if (alpha == 255) {
            std::memcpy(dest, src, size_t(length) * sizeof(uint32_t));
            return;
        }
        const uint32_t inverseAlpha = 255 - alpha;
        while (length-- > 0) {
            *dest = interpolate_pixel(*src, alpha, *dest, inverseAlpha);
            ++src;
            ++dest;
        }
        return;
    }

    if (alpha == 255) {
        while (length-- > 0) {
            const uint32_t source = *src++;
            if (source >= 0xff000000u)
                *dest = source;
            else if (source != 0)
                *dest = source + BYTE_MUL(*dest, vAlpha(~source));
            ++dest;
        }
        return;
    }

    while (length-- > 0) {
        const uint32_t sourcePixel = *src++;
        const uint32_t source = BYTE_MUL(sourcePixel, alpha);
        *dest = source + BYTE_MUL(*dest, vAlpha(~source));
        ++dest;
    }
}

TEST(VDrawHelper, SimdColorKernelsMatchScalarReference)
{
    RenderFuncTable table;
    uint32_t state = 0x5eed1234u;

    for (const BlendMode mode : {BlendMode::Src, BlendMode::SrcOver}) {
        for (const uint32_t alpha : kAlphas) {
            for (int offset = 0; offset < 4; ++offset) {
                for (int length = 0; length <= 65; ++length) {
                    std::vector<uint32_t> actual(size_t(length + offset + 4));
                    for (uint32_t &pixel : actual) pixel = nextRandom(state);
                    std::vector<uint32_t> expected = actual;
                    const uint32_t color = nextRandom(state);

                    referenceColor(mode, expected.data() + offset, length,
                                   color, alpha);
                    table.color(mode)(actual.data() + offset, length, color,
                                      alpha);
                    EXPECT_EQ(expected, actual)
                        << "mode=" << int(mode) << " alpha=" << alpha
                        << " offset=" << offset << " length=" << length;
                }
            }
        }
    }
}

TEST(VDrawHelper, SimdSourceKernelsMatchScalarReference)
{
    RenderFuncTable table;
    uint32_t state = 0xc001d00du;

    for (const BlendMode mode : {BlendMode::Src, BlendMode::SrcOver}) {
        for (const uint32_t alpha : kAlphas) {
            for (int destOffset = 0; destOffset < 4; ++destOffset) {
                for (int srcOffset = 0; srcOffset < 4; ++srcOffset) {
                    for (int length = 0; length <= 65; ++length) {
                        const size_t destSize = size_t(length + destOffset + 4);
                        const size_t srcSize = size_t(length + srcOffset + 4);
                        std::vector<uint32_t> actual(destSize);
                        std::vector<uint32_t> source(srcSize);
                        for (uint32_t &pixel : actual) pixel = nextRandom(state);
                        for (uint32_t &pixel : source) pixel = nextRandom(state);
                        std::vector<uint32_t> expected = actual;

                        referenceSrc(mode, expected.data() + destOffset, length,
                                     source.data() + srcOffset, alpha);
                        table.src(mode)(actual.data() + destOffset, length,
                                        source.data() + srcOffset, alpha);
                        EXPECT_EQ(expected, actual)
                            << "mode=" << int(mode) << " alpha=" << alpha
                            << " destOffset=" << destOffset
                            << " srcOffset=" << srcOffset
                            << " length=" << length;
                    }
                }
            }
        }
    }
}

}  // namespace
