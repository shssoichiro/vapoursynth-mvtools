#include <cstdint>
#include <stdexcept>
#include <unordered_map>

#include "Luma.h"


enum InstructionSets {
    Scalar,
    SSE2,
};


template <unsigned width, unsigned height, typename PixelType>
unsigned int luma_c(const uint8_t *pSrc8, intptr_t nSrcPitch) {
    unsigned int meanLuma = 0;
    for (unsigned j = 0; j < height; j++) {
        for (unsigned i = 0; i < width; i++) {
            const PixelType *pSrc = (const PixelType *)pSrc8;
            meanLuma += pSrc[i];
        }
        pSrc8 += nSrcPitch;
    }
    return meanLuma;
}


#if defined(MVTOOLS_X86) || defined(MVTOOLS_ARM)

#if defined(MVTOOLS_ARM)
#include "sse2neon.h"
#else
#include <emmintrin.h>
#endif


#define zeroes _mm_setzero_si128()


template <unsigned width, unsigned height>
unsigned int luma_sse2(const uint8_t *pSrc, intptr_t nSrcPitch) {
    __m128i sum = zeroes;

    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x += 16) {
            __m128i src;
            if (width == 4)
                src = _mm_cvtsi32_si128(*(const int *)pSrc);
            else if (width == 8)
                src = _mm_loadl_epi64((const __m128i *)pSrc);
            else
                src = _mm_loadu_si128((const __m128i *)&pSrc[x]);

            sum = _mm_add_epi64(sum, _mm_sad_epu8(src, zeroes));
        }

        pSrc += nSrcPitch;
    }

    if (width >= 16)
        sum = _mm_add_epi64(sum, _mm_srli_si128(sum, 8));

    return (unsigned)_mm_cvtsi128_si32(sum);
}

template <unsigned width, unsigned height>
unsigned int luma_sse2_16b(const uint8_t *pSrc, intptr_t nSrcPitch) {
    const uint16_t *pSrc16 = (const uint16_t *)pSrc;

    if (width == 4 && height == 4) {
        // Special case: 4x4 - process all at once
        __m128i row0 = _mm_loadl_epi64((const __m128i *)pSrc16);
        pSrc16 = (const uint16_t *)((const uint8_t *)pSrc16 + nSrcPitch);
        __m128i row1 = _mm_loadl_epi64((const __m128i *)pSrc16);
        pSrc16 = (const uint16_t *)((const uint8_t *)pSrc16 + nSrcPitch);
        __m128i row2 = _mm_loadl_epi64((const __m128i *)pSrc16);
        pSrc16 = (const uint16_t *)((const uint8_t *)pSrc16 + nSrcPitch);
        __m128i row3 = _mm_loadl_epi64((const __m128i *)pSrc16);

        // Unpack each row to 32-bit (zero-extend)
        row0 = _mm_unpacklo_epi16(row0, zeroes);
        row1 = _mm_unpacklo_epi16(row1, zeroes);
        row2 = _mm_unpacklo_epi16(row2, zeroes);
        row3 = _mm_unpacklo_epi16(row3, zeroes);

        // Add all rows
        __m128i sum = _mm_add_epi32(_mm_add_epi32(row0, row1),
                                    _mm_add_epi32(row2, row3));

        // Horizontal sum
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
        return (unsigned)_mm_cvtsi128_si32(sum);
    }
    else if (width <= 8) {
        // For width 8 or less, use single accumulator with unpacking
        __m128i sum = zeroes;

        for (unsigned y = 0; y < height; y++) {
            __m128i src = _mm_loadu_si128((const __m128i *)pSrc16);

            // Unpack to 32-bit and accumulate
            __m128i lo = _mm_unpacklo_epi16(src, zeroes);
            __m128i hi = _mm_unpackhi_epi16(src, zeroes);
            sum = _mm_add_epi32(sum, _mm_add_epi32(lo, hi));

            pSrc16 = (const uint16_t *)((const uint8_t *)pSrc16 + nSrcPitch);
        }

        // Horizontal sum
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
        return (unsigned)_mm_cvtsi128_si32(sum);
    }
    else {
        // For larger widths, use multiple accumulators to reduce dependency chains
        __m128i sum0 = zeroes;
        __m128i sum1 = zeroes;
        __m128i sum2 = zeroes;
        __m128i sum3 = zeroes;

        for (unsigned y = 0; y < height; y++) {
            const uint16_t *row = pSrc16;

            // Process width in chunks of 32 pixels (4 x 8)
            for (unsigned x = 0; x < width; x += 32) {
                __m128i src0 = _mm_loadu_si128((const __m128i *)&row[x]);
                __m128i src1 = _mm_loadu_si128((const __m128i *)&row[x + 8]);

                // Unpack and accumulate to different accumulators
                __m128i lo0 = _mm_unpacklo_epi16(src0, zeroes);
                __m128i hi0 = _mm_unpackhi_epi16(src0, zeroes);
                sum0 = _mm_add_epi32(sum0, lo0);
                sum1 = _mm_add_epi32(sum1, hi0);

                __m128i lo1 = _mm_unpacklo_epi16(src1, zeroes);
                __m128i hi1 = _mm_unpackhi_epi16(src1, zeroes);
                sum2 = _mm_add_epi32(sum2, lo1);
                sum3 = _mm_add_epi32(sum3, hi1);

                if (x + 16 < width) {
                    __m128i src2 = _mm_loadu_si128((const __m128i *)&row[x + 16]);
                    __m128i src3 = _mm_loadu_si128((const __m128i *)&row[x + 24]);

                    __m128i lo2 = _mm_unpacklo_epi16(src2, zeroes);
                    __m128i hi2 = _mm_unpackhi_epi16(src2, zeroes);
                    sum0 = _mm_add_epi32(sum0, lo2);
                    sum1 = _mm_add_epi32(sum1, hi2);

                    __m128i lo3 = _mm_unpacklo_epi16(src3, zeroes);
                    __m128i hi3 = _mm_unpackhi_epi16(src3, zeroes);
                    sum2 = _mm_add_epi32(sum2, lo3);
                    sum3 = _mm_add_epi32(sum3, hi3);
                }
            }

            pSrc16 = (const uint16_t *)((const uint8_t *)pSrc16 + nSrcPitch);
        }

        // Combine all accumulators
        sum0 = _mm_add_epi32(sum0, sum1);
        sum2 = _mm_add_epi32(sum2, sum3);
        sum0 = _mm_add_epi32(sum0, sum2);

        // Horizontal sum
        sum0 = _mm_add_epi32(sum0, _mm_srli_si128(sum0, 8));
        sum0 = _mm_add_epi32(sum0, _mm_srli_si128(sum0, 4));
        return (unsigned)_mm_cvtsi128_si32(sum0);
    }
}


#undef zeroes


#endif // MVTOOLS_X86


// opt can fit in four bits, if the width and height need more than eight bits each.
#define KEY(width, height, bits, opt) (unsigned)(width) << 24 | (height) << 16 | (bits) << 8 | (opt)

#if defined(MVTOOLS_X86) || defined(MVTOOLS_ARM)
#define LUMA_SSE2(width, height) \
    { KEY(width, height, 8, SSE2), luma_sse2<width, height> }, \
    { KEY(width, height, 16, SSE2), luma_sse2_16b<width, height> },
#else
#define LUMA_SSE2(width, height)
#endif

#define LUMA(width, height) \
    { KEY(width, height, 8, Scalar), luma_c<width, height, uint8_t> }, \
    { KEY(width, height, 16, Scalar), luma_c<width, height, uint16_t> },

static const std::unordered_map<uint32_t, LUMAFunction> luma_functions = {
    LUMA(4, 4)
    LUMA(8, 4)
    LUMA(8, 8)
    LUMA(16, 2)
    LUMA(16, 8)
    LUMA(16, 16)
    LUMA(32, 16)
    LUMA(32, 32)
    LUMA(64, 32)
    LUMA(64, 64)
    LUMA(128, 64)
    LUMA(128, 128)
    LUMA_SSE2(4, 4)
    LUMA_SSE2(8, 4)
    LUMA_SSE2(8, 8)
    LUMA_SSE2(16, 2)
    LUMA_SSE2(16, 8)
    LUMA_SSE2(16, 16)
    LUMA_SSE2(32, 16)
    LUMA_SSE2(32, 32)
    LUMA_SSE2(64, 32)
    LUMA_SSE2(64, 64)
    LUMA_SSE2(128, 64)
    LUMA_SSE2(128, 128)
};

LUMAFunction selectLumaFunction(unsigned width, unsigned height, unsigned bits, int opt) {
    LUMAFunction luma = luma_functions.at(KEY(width, height, bits, Scalar));

#if defined(MVTOOLS_X86) || defined(MVTOOLS_ARM)
    if (opt) {
        try {
            luma = luma_functions.at(KEY(width, height, bits, SSE2));
        } catch (std::out_of_range &) { }
    }
#endif

    return luma;
}

#undef LUMA
#undef LUMA_SSE2
#undef KEY
