#include <gtest/gtest.h>

#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vdrawhelper.h"

void fetch_linear_gradient(uint32_t *buffer, const Operator *op,
                           const VSpanData *data, int y, int x, int length);

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

constexpr int kGradientFixedBits = 8;
constexpr int kGradientFixedSize = 1 << kGradientFixedBits;

int referenceGradientClamp(const VGradientData *gradient, int position)
{
    if (gradient->mSpread == VGradient::Spread::Repeat) {
        position %= VGradient::colorTableSize;
        if (position < 0) position += VGradient::colorTableSize;
    } else if (gradient->mSpread == VGradient::Spread::Reflect) {
        const int limit = VGradient::colorTableSize * 2;
        position %= limit;
        if (position < 0) position += limit;
        if (position >= VGradient::colorTableSize)
            position = limit - 1 - position;
    } else {
        if (position < 0)
            position = 0;
        else if (position >= VGradient::colorTableSize)
            position = VGradient::colorTableSize - 1;
    }
    return position;
}

uint32_t referenceGradientPixelFixed(const VGradientData *gradient,
                                     int fixedPosition)
{
    const int position =
        (fixedPosition + (kGradientFixedSize / 2)) >> kGradientFixedBits;
    return gradient->mColorTable[referenceGradientClamp(gradient, position)];
}

uint32_t referenceGradientPixel(const VGradientData *gradient, float position)
{
    const int tablePosition = int(position *
                                  (VGradient::colorTableSize - 1) + 0.5f);
    return gradient->mColorTable[
        referenceGradientClamp(gradient, tablePosition)];
}

// This intentionally mirrors the implementation before the spread-specialized
// fixed-point loop was introduced. Keep its operation order unchanged so the
// differential test detects even one-pixel rounding differences.
void referenceFetchLinearGradient(uint32_t *buffer, const Operator *op,
                                  const VSpanData *data, int y, int x,
                                  int length)
{
    float t, increment;
    const VGradientData *gradient = &data->mGradient;

    bool affine = true;
    float rx = 0, ry = 0;
    if (op->linear.l == 0) {
        t = increment = 0;
    } else {
        rx = data->m21 * (y + float(0.5)) +
             data->m11 * (x + float(0.5)) + data->dx;
        ry = data->m22 * (y + float(0.5)) +
             data->m12 * (x + float(0.5)) + data->dy;
        t = op->linear.dx * rx + op->linear.dy * ry + op->linear.off;
        increment = op->linear.dx * data->m11 +
                    op->linear.dy * data->m12;
        affine = !data->m13 && !data->m23;

        if (affine) {
            t *= (VGradient::colorTableSize - 1);
            increment *= (VGradient::colorTableSize - 1);
        }
    }

    uint32_t *end = buffer + length;
    if (affine) {
        if (increment > float(-1e-5) && increment < float(1e-5)) {
            memfill32(buffer,
                      referenceGradientPixelFixed(
                          gradient, int(t * kGradientFixedSize)),
                      length);
        } else if (t + increment * length <
                       float(INT_MAX >> (kGradientFixedBits + 1)) &&
                   t + increment * length >
                       float(INT_MIN >> (kGradientFixedBits + 1))) {
            int fixedPosition = int(t * kGradientFixedSize);
            const int fixedIncrement = int(increment * kGradientFixedSize);
            while (buffer < end) {
                *buffer++ =
                    referenceGradientPixelFixed(gradient, fixedPosition);
                fixedPosition += fixedIncrement;
            }
        } else {
            while (buffer < end) {
                *buffer++ = referenceGradientPixel(
                    gradient, t / VGradient::colorTableSize);
                t += increment;
            }
        }
    } else {
        float rw = data->m23 * (y + float(0.5)) +
                   data->m13 * (x + float(0.5)) + data->m33;
        while (buffer < end) {
            const float xt = rx / rw;
            const float yt = ry / rw;
            t = op->linear.dx * xt + op->linear.dy * yt + op->linear.off;

            *buffer++ = referenceGradientPixel(gradient, t);
            rx += data->m11;
            ry += data->m12;
            rw += data->m13;
            if (!rw) rw += data->m13;
        }
    }
}

void expectGradientMatchesReference(const Operator &op, const VSpanData &data,
                                    int y, int x, int length,
                                    const char *caseName)
{
    constexpr uint32_t sentinel = 0xd34db33fu;
    std::vector<uint32_t> expected(size_t(length + 8), sentinel);
    std::vector<uint32_t> actual = expected;

    referenceFetchLinearGradient(expected.data() + 3, &op, &data, y, x,
                                 length);
    fetch_linear_gradient(actual.data() + 3, &op, &data, y, x, length);

    EXPECT_EQ(expected, actual)
        << "case=" << caseName << " spread=" << int(data.mGradient.mSpread)
        << " x=" << x << " y=" << y << " length=" << length;
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

TEST(VDrawHelper, SourceOverPreservesSpecialPixelsAndExactInPlaceInput)
{
    RenderFuncTable table;
    const std::array<uint32_t, 17> pixels{{
        0x00000000u, 0x00000001u, 0x00ffffffu, 0x01000000u,
        0x7f010203u, 0x80abcdefu, 0xfe112233u, 0xff000000u,
        0xffffffffu, 0xff123456u, 0x00000000u, 0x40010203u,
        0x80102030u, 0xfeffffffu, 0xff654321u, 0x00000001u,
        0x00000000u,
    }};

    for (const uint32_t alpha : kAlphas) {
        std::vector<uint32_t> expected(pixels.begin(), pixels.end());
        std::vector<uint32_t> actual = expected;
        referenceSrc(BlendMode::SrcOver, expected.data(), int(expected.size()),
                     expected.data(), alpha);
        table.src(BlendMode::SrcOver)(actual.data(), int(actual.size()),
                                      actual.data(), alpha);
        EXPECT_EQ(expected, actual) << "alpha=" << alpha;
    }
}

TEST(VDrawHelper, LinearGradientAffineMatchesPreOptimizationReference)
{
    std::array<uint32_t, VGradient::colorTableSize> colorTable{};
    for (size_t i = 0; i < colorTable.size(); ++i) {
        colorTable[i] = uint32_t(i * 2654435761u) ^
                        uint32_t((i * 37u) << 16) ^ 0xa5000000u;
    }

    struct AffineCase {
        const char *name;
        int x;
        int y;
        float start;
        float increment;
    };
    const std::array<AffineCase, 9> cases{{
        {"constant-negative-boundary", -2049, -3, -2048.5f, 0.0f},
        {"negative-repeat-seam", -1025, 2, -1024.5f, 1.0f},
        {"negative-near-zero", -3, -2, -1.5f, 0.25f},
        {"zero-crossing-negative-step", 0, 1, 1.5f, -0.5f},
        {"repeat-positive-seam", 1023, -1, 1023.5f, 1.0f},
        {"reflect-positive-seam", 2047, 3, 2047.5f, 1.0f},
        {"reflect-negative-step", 2049, 0, 2048.5f, -1.0f},
        {"multi-period-negative-step", 4097, -4, 3073.25f, -2.25f},
        {"sub-fixed-increment", -17, 5, 511.498f, 0.0039f},
    }};
    const std::array<int, 13> lengths{{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 33, 65,
    }};

    for (const VGradient::Spread spread : {
             VGradient::Spread::Pad,
             VGradient::Spread::Repeat,
             VGradient::Spread::Reflect,
         }) {
        for (const AffineCase &testCase : cases) {
            VSpanData data{};
            data.mGradient.mSpread = spread;
            data.mGradient.mColorTable = colorTable.data();
            data.m11 = testCase.increment /
                       float(VGradient::colorTableSize - 1);
            data.m12 = 0.0f;
            data.m13 = 0.0f;
            data.m21 = 0.375f /
                       float(VGradient::colorTableSize - 1);
            data.m22 = 1.0f;
            data.m23 = 0.0f;
            data.m33 = 1.0f;
            data.dx = testCase.start /
                          float(VGradient::colorTableSize - 1) -
                      data.m11 * (testCase.x + 0.5f) -
                      data.m21 * (testCase.y + 0.5f);
            data.dy = 0.0f;

            Operator op{};
            op.linear.l = 1.0f;
            op.linear.dx = 1.0f;
            op.linear.dy = 0.0f;
            op.linear.off = 0.0f;

            for (const int length : lengths) {
                expectGradientMatchesReference(op, data, testCase.y,
                                               testCase.x, length,
                                               testCase.name);
            }
        }

        // A degenerate linear gradient follows a separate constant-affine path.
        VSpanData data{};
        data.mGradient.mSpread = spread;
        data.mGradient.mColorTable = colorTable.data();
        data.m13 = data.m23 = 0.0f;
        Operator op{};
        op.linear.l = 0.0f;
        for (const int length : lengths) {
            expectGradientMatchesReference(op, data, -11, -23, length,
                                           "degenerate-constant");
        }
    }
}

TEST(VDrawHelper, LinearGradientFloatFallbackMatchesPreOptimizationReference)
{
    std::array<uint32_t, VGradient::colorTableSize> colorTable{};
    for (size_t i = 0; i < colorTable.size(); ++i)
        colorTable[i] = 0xff000000u | uint32_t(i * 0x00010101u);

    struct FallbackCase {
        const char *name;
        float start;
        float increment;
    };
    const std::array<FallbackCase, 2> cases{{
        {"positive-fixed-limit", 1024.0f, 8388608.0f},
        {"negative-fixed-limit", -1024.0f, -8388608.0f},
    }};
    const std::array<int, 11> lengths{{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 67,
    }};

    for (const VGradient::Spread spread : {
             VGradient::Spread::Pad,
             VGradient::Spread::Repeat,
             VGradient::Spread::Reflect,
         }) {
        for (const FallbackCase &testCase : cases) {
            constexpr int x = 0;
            constexpr int y = 4;
            VSpanData data{};
            data.mGradient.mSpread = spread;
            data.mGradient.mColorTable = colorTable.data();
            data.m11 = testCase.increment /
                       float(VGradient::colorTableSize - 1);
            data.m12 = 0.0f;
            data.m13 = data.m23 = 0.0f;
            data.m21 = 0.0f;
            data.m22 = data.m33 = 1.0f;
            data.dx = testCase.start /
                          float(VGradient::colorTableSize - 1) -
                      data.m11 * (x + 0.5f);
            data.dy = 0.0f;

            Operator op{};
            op.linear.l = 1.0f;
            op.linear.dx = 1.0f;
            op.linear.dy = 0.0f;
            op.linear.off = 0.0f;
            for (const int length : lengths) {
                expectGradientMatchesReference(op, data, y, x, length,
                                               testCase.name);
            }
        }
    }
}

TEST(VDrawHelper, LinearGradientPerspectiveMatchesPreOptimizationReference)
{
    std::array<uint32_t, VGradient::colorTableSize> colorTable{};
    for (size_t i = 0; i < colorTable.size(); ++i)
        colorTable[i] = uint32_t(0x10203040u + i * 0x0003070bu);

    const std::array<int, 13> lengths{{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 31, 70,
    }};
    for (const VGradient::Spread spread : {
             VGradient::Spread::Pad,
             VGradient::Spread::Repeat,
             VGradient::Spread::Reflect,
         }) {
        for (int variant = 0; variant < 2; ++variant) {
            const int x = variant == 0 ? -19 : 1024;
            const int y = variant == 0 ? -7 : 5;
            VSpanData data{};
            data.mGradient.mSpread = spread;
            data.mGradient.mColorTable = colorTable.data();
            data.m11 = variant == 0 ? 1.25f : -0.75f;
            data.m12 = variant == 0 ? -0.5f : 0.625f;
            data.m13 = variant == 0 ? 0.015625f : -0.000125f;
            data.m21 = variant == 0 ? 0.375f : -0.25f;
            data.m22 = variant == 0 ? 0.875f : 1.125f;
            data.m23 = variant == 0 ? 0.03125f : 0.00025f;
            data.m33 = variant == 0 ? 3.0f : 1.75f;
            data.dx = variant == 0 ? -1024.25f : 2048.5f;
            data.dy = variant == 0 ? 512.75f : -1023.5f;

            Operator op{};
            op.linear.l = 1.0f;
            op.linear.dx = variant == 0 ? 0.75f : -1.125f;
            op.linear.dy = variant == 0 ? -0.25f : 0.375f;
            op.linear.off = variant == 0 ? 1023.5f : -2048.5f;

            for (const int length : lengths) {
                expectGradientMatchesReference(
                    op, data, y, x, length,
                    variant == 0 ? "perspective-negative-coordinate"
                                 : "perspective-positive-coordinate");
            }
        }
    }
}

}  // namespace
