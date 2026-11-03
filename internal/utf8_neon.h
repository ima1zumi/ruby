#ifndef INTERNAL_UTF8_NEON_H
#define INTERNAL_UTF8_NEON_H

/* Experimental C/ARM64 adaptation of simdutf's lookup4 validator.
 * Reference: simdutf 8.1.0, generic/utf8_validation/utf8_lookup4_algorithm.h
 * and arm64/implementation.cpp; provenance in experiment/reference/.
 * Copyright 2021 The simdutf authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "internal/simd.h"
#include "ruby/internal/attr/noinline.h"
#include <stddef.h>
#include <string.h>

/* vqtbl1q and vmaxvq require AArch64, unlike some other NEON intrinsics.
 * The opt-out is for exercising the ordinary fallback on the development host. */
#if defined(HAVE_SIMD_NEON) && (defined(__aarch64__) || defined(_M_ARM64)) && !defined(RUBY_UTF8_NEON_DISABLE)
#define RUBY_UTF8_NEON 1

static inline uint8x16_t
utf8_neon_errors(uint8x16_t in, uint8x16_t prev)
{
    enum { SHORT = 1, LONG = 2, OVER3 = 4, LARGE = 8,
           SURROGATE = 16, OVER2 = 32, LARGE_1000 = 64,
           OVER4 = 64, TWO = 128, CARRY = SHORT | LONG | TWO };
    const uint8x16_t high1 = {LONG,LONG,LONG,LONG,LONG,LONG,LONG,LONG,
        TWO,TWO,TWO,TWO, SHORT|OVER2, SHORT, SHORT|OVER3|SURROGATE,
        SHORT|LARGE|LARGE_1000|OVER4};
    const uint8x16_t low1 = {CARRY|OVER3|OVER2|OVER4, CARRY|OVER2, CARRY,CARRY,
        CARRY|LARGE, CARRY|LARGE|LARGE_1000, CARRY|LARGE|LARGE_1000,
        CARRY|LARGE|LARGE_1000, CARRY|LARGE|LARGE_1000,
        CARRY|LARGE|LARGE_1000, CARRY|LARGE|LARGE_1000,
        CARRY|LARGE|LARGE_1000, CARRY|LARGE|LARGE_1000,
        CARRY|LARGE|LARGE_1000|SURROGATE, CARRY|LARGE|LARGE_1000,
        CARRY|LARGE|LARGE_1000};
    const uint8x16_t high2 = {SHORT,SHORT,SHORT,SHORT,SHORT,SHORT,SHORT,SHORT,
        LONG|OVER2|TWO|OVER3|LARGE_1000|OVER4, LONG|OVER2|TWO|OVER3|LARGE,
        LONG|OVER2|TWO|SURROGATE|LARGE, LONG|OVER2|TWO|SURROGATE|LARGE,
        SHORT,SHORT,SHORT,SHORT};
    uint8x16_t p1 = vextq_u8(prev, in, 15);
    uint8x16_t p2 = vextq_u8(prev, in, 14);
    uint8x16_t p3 = vextq_u8(prev, in, 13);
    uint8x16_t special = vandq_u8(vandq_u8(
        vqtbl1q_u8(high1, vshrq_n_u8(p1, 4)),
        vqtbl1q_u8(low1, vandq_u8(p1, vdupq_n_u8(15)))),
        vqtbl1q_u8(high2, vshrq_n_u8(in, 4)));
    /* XOR (not OR) is the ARM64 reference's third/fourth-byte rule. */
    uint8x16_t must23 = veorq_u8(vcgeq_u8(p2, vdupq_n_u8(0xe0)),
                                vcgeq_u8(p3, vdupq_n_u8(0xf0)));
    return veorq_u8(special, vandq_u8(must23, vdupq_n_u8(0x80)));
}

/* p starts at a character boundary; no reads before p or at/after e.
 * The caller has already established that the input contains non-ASCII.
 * Full blocks carry the preceding 3 bytes through prev. A zero-padded
 * final block, including at exact multiples of 16, detects truncated EOF. */
RBIMPL_ATTR_NOINLINE()
static int
utf8_neon_valid(const unsigned char *p, const unsigned char *e)
{
    uint8x16_t prev = vdupq_n_u8(0);
    while ((size_t)(e - p) >= 16) {
        uint8x16_t in = vld1q_u8(p);
        if (vmaxvq_u8(utf8_neon_errors(in, prev))) return 0;
        prev = in;
        p += 16;
    }
    unsigned char tail[16] = {0};
    size_t remaining = (size_t)(e - p);
    if (remaining) memcpy(tail, p, remaining);
    return vmaxvq_u8(utf8_neon_errors(vld1q_u8(tail), prev)) == 0;
}
#endif
#endif
