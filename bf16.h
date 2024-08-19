// google brain half-precision bfloat16
typedef struct { uint16_t bits; } degima_bf16_t;
static degima_bf16_t degima_fp32_to_bf16(float);
static float         degima_bf16_to_fp32(degima_bf16_t);  // consider just doing << 16
/**
static void          degima_bf16_to_fp32_row(const degima_bf16_t *, float *, int64_t);
static void          degima_fp32_to_bf16_row_ref(const float *, degima_bf16_t *, int64_t);
static void          degima_fp32_to_bf16_row(const float *, degima_bf16_t *, int64_t);
*/

#define FP32_to_BF16(x) degima_fp32_to_bf16(x)
#define BF16_to_FP32(x) degima_bf16_to_fp32(x)

/**
 * Converts brain16 to float32.
 *
 * The bfloat16 floating point format has the following structure:
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───┐
 *     0b0000000000000000 brain16
 *
 * Since bf16 has the same number of exponent bits as a 32bit float,
 * encoding and decoding numbers becomes relatively straightforward.
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───────────────────┐
 *     0b00000000000000000000000000000000 IEEE binary32
 *
 * For comparison, the standard fp16 format has fewer exponent bits.
 *
 *       ┌sign
 *       │
 *       │  ┌exponent
 *       │  │
 *       │  │    ┌mantissa
 *       │  │    │
 *       │┌─┴─┐┌─┴──────┐
 *     0b0000000000000000 IEEE binary16
 *
 * @see IEEE 754-2008
 *   https://iremi.univ-reunion.fr/IMG/pdf/ieee-754-2008.pdf
 */
static inline float degima_bf16_to_fp32(degima_bf16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    return u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
static inline degima_bf16_t degima_fp32_to_bf16(float s) {
    degima_bf16_t h;
    union {
        float f;
        uint32_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        return h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    return h;
}

