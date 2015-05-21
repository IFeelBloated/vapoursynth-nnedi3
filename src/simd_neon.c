#include <stdint.h>
#include <arm_neon.h>


static const uint32x4_t sign_bits_f = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
static const uint32x4_t sign_bits_f_zero_l = { 0, 0x7fffffff, 0x7fffffff, 0x7fffffff };
static const float32x4_t ones_f = { 1.0f, 1.0f, 1.0f, 1.0f };


// http://stackoverflow.com/questions/6759897/
static inline __attribute__((always_inline)) float32x4_t reciprocal(float32x4_t x) {
    float32x4_t recip = vrecpeq_f32(x);
    recip = vmulq_f32(vrecpsq_f32(x, recip), recip);
    recip = vmulq_f32(vrecpsq_f32(x, recip), recip);
    return recip;
}


void word2float48_neon(const uint8_t *t8, const int pitch, float *p) {
    const uint16_t *t = (const uint16_t *)t8;

    uint16x4_t zero = { 0, 0, 0, 0 };

    for (int i = 0; i < 4; i++) {
        uint16x4x2_t u1 = vzip_u16(vld1_u16(t), zero);
        uint16x4x2_t u2 = vzip_u16(vld1_u16(t + 4), zero);
        uint16x4x2_t u3 = vzip_u16(vld1_u16(t + 8), zero);

        float32x4_t f1 = vcvtq_f32_u32(vreinterpretq_u32_u16(vcombine_u16(u1.val[0], u1.val[1])));
        float32x4_t f2 = vcvtq_f32_u32(vreinterpretq_u32_u16(vcombine_u16(u2.val[0], u2.val[1])));
        float32x4_t f3 = vcvtq_f32_u32(vreinterpretq_u32_u16(vcombine_u16(u3.val[0], u3.val[1])));

        vst1q_f32(p, f1);
        vst1q_f32(p + 4, f2);
        vst1q_f32(p + 8, f3);

        t += pitch; // it was already halved
        p += 12;
    }
}


void computeNetwork0_neon(const float *input, const float *weights, uint8_t *d) {
    float32x4_t m0 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float32x4_t m1 = m0;
    float32x4_t m2 = m0;
    float32x4_t m3 = m0;

    float32x4_t m4, m5, m6, m7;

    for (int i = 0; i < 192/4; i += 4) {
        m4 = vld1q_f32(input + i);
        m5 = m4;
        m6 = m4;
        m7 = m4;

        m4 = vmulq_f32(m4, vld1q_f32(weights + i * 4));
        m5 = vmulq_f32(m5, vld1q_f32(weights + i * 4 + 4));
        m6 = vmulq_f32(m6, vld1q_f32(weights + i * 4 + 8));
        m7 = vmulq_f32(m7, vld1q_f32(weights + i * 4 + 12));

        m0 = vaddq_f32(m0, m4);
        m1 = vaddq_f32(m1, m5);
        m2 = vaddq_f32(m2, m6);
        m3 = vaddq_f32(m3, m7);
    }

    float32x2_t sum0 = vpadd_f32(vget_low_f32(m0), vget_high_f32(m0));
    float32x2_t sum1 = vpadd_f32(vget_low_f32(m1), vget_high_f32(m1));
    float32x2_t sum2 = vpadd_f32(vget_low_f32(m2), vget_high_f32(m2));
    float32x2_t sum3 = vpadd_f32(vget_low_f32(m3), vget_high_f32(m3));
    sum0 = vpadd_f32(sum0, sum1);
    sum1 = vpadd_f32(sum2, sum3);
    m0 = vcombine_f32(sum0, sum1);

    m0 = vaddq_f32(m0, vld1q_f32(weights + 768/4));

    m1 = m0;
    m0 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(m0), sign_bits_f_zero_l));
    m0 = vaddq_f32(m0, ones_f);
    m0 = vmulq_f32(reciprocal(m0), m1);

    m1 = vdupq_lane_f32(vget_low_f32(m0), 0);
    m2 = vdupq_lane_f32(vget_low_f32(m0), 1);
    m3 = vdupq_lane_f32(vget_high_f32(m0), 0);
    m4 = vdupq_lane_f32(vget_high_f32(m0), 1);

    m1 = vmulq_f32(m1, vld1q_f32(weights + 784/4));
    m2 = vmulq_f32(m2, vld1q_f32(weights + (784+16)/4));
    m3 = vmulq_f32(m3, vld1q_f32(weights + (784+32)/4));
    m4 = vmulq_f32(m4, vld1q_f32(weights + (784+48)/4));

    m1 = vaddq_f32(m1, m2);
    m3 = vaddq_f32(m3, m4);
    m1 = vaddq_f32(m1, m3);
    m1 = vaddq_f32(m1, vld1q_f32(weights + (784+64)/4));

    m7 = m1;
    m1 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(m1), sign_bits_f));
    m1 = vaddq_f32(m1, ones_f);
    m7 = vmulq_f32(reciprocal(m1), m7);

    m3 = m0;

    m0 = vdupq_lane_f32(vget_low_f32(m0), 0);
    m1 = vdupq_lane_f32(vget_low_f32(m3), 1);
    m2 = vdupq_lane_f32(vget_high_f32(m3), 0);
    m3 = vdupq_lane_f32(vget_high_f32(m3), 1);

    m0 = vmulq_f32(m0, vld1q_f32(weights + 864/4));
    m1 = vmulq_f32(m1, vld1q_f32(weights + (864+16)/4));
    m2 = vmulq_f32(m2, vld1q_f32(weights + (864+32)/4));
    m3 = vmulq_f32(m3, vld1q_f32(weights + (864+48)/4));

    m4 = vdupq_lane_f32(vget_low_f32(m7), 0);
    m5 = vdupq_lane_f32(vget_low_f32(m7), 1);
    m6 = vdupq_lane_f32(vget_high_f32(m7), 0);
    m7 = vdupq_lane_f32(vget_high_f32(m7), 1);

    m4 = vmulq_f32(m4, vld1q_f32(weights + (864+64)/4));
    m5 = vmulq_f32(m5, vld1q_f32(weights + (864+80)/4));
    m6 = vmulq_f32(m6, vld1q_f32(weights + (864+96)/4));
    m7 = vmulq_f32(m7, vld1q_f32(weights + (864+112)/4));

    m0 = vaddq_f32(m0, m1);
    m2 = vaddq_f32(m2, m3);
    m4 = vaddq_f32(m4, m5);
    m6 = vaddq_f32(m6, m7);

    m0 = vaddq_f32(m0, m2);
    m4 = vaddq_f32(m4, m6);
    m0 = vaddq_f32(m0, m4);

    m0 = vaddq_f32(m0, vld1q_f32(weights + (864+128)/4));

    float32x2_t maximum = vmax_f32(vget_low_f32(m0), vget_high_f32(m0));
    d[0] = (vget_lane_f32(maximum, 1) <= vget_lane_f32(maximum, 0));
}


void dotProd_neon(const float *data, const float *weights, float *vals, const int n, const int len, const float *istd) {
    for (int i = 0; i < n; i += 4) {
        float32x4_t accum0 = { 0.0f, 0.0f, 0.0f, 0.0f };
        float32x4_t accum1 = accum0;
        float32x4_t accum2 = accum0;
        float32x4_t accum3 = accum0;

        for (int j = 0; j < len; j += 4) {
            float32x4_t d0 = vld1q_f32(data + j);
            float32x4_t d1 = d0;
            float32x4_t d2 = d0;
            float32x4_t d3 = d0;

            float32x4_t w0 = vld1q_f32(weights);
            float32x4_t w1 = vld1q_f32(weights + 4);
            float32x4_t w2 = vld1q_f32(weights + 8);
            float32x4_t w3 = vld1q_f32(weights + 12);

            accum0 = vaddq_f32(accum0, vmulq_f32(d0, w0));
            accum1 = vaddq_f32(accum1, vmulq_f32(d1, w1));
            accum2 = vaddq_f32(accum2, vmulq_f32(d2, w2));
            accum3 = vaddq_f32(accum3, vmulq_f32(d3, w3));

            weights += 16;
        }

        float32x2_t sum0 = vpadd_f32(vget_low_f32(accum0), vget_high_f32(accum0));
        float32x2_t sum1 = vpadd_f32(vget_low_f32(accum1), vget_high_f32(accum1));
        float32x2_t sum2 = vpadd_f32(vget_low_f32(accum2), vget_high_f32(accum2));
        float32x2_t sum3 = vpadd_f32(vget_low_f32(accum3), vget_high_f32(accum3));
        sum0 = vpadd_f32(sum0, sum1);
        sum1 = vpadd_f32(sum2, sum3);
        float32x4_t sum = vcombine_f32(sum0, sum1);
        
        vst1q_f32(vals + i, sum);
    }

    // XXX this should be in the loop above
    // the sum shouldn't be stored only to be loaded again here
    for (int i = 0; i < n; i += 4) {
        float32x4_t val = vld1q_f32(vals + i);
        float32x4_t weight = vld1q_f32(weights + i);
        val = vmulq_n_f32(val, istd[0]);
        val = vaddq_f32(val, weight);
        vst1q_f32(vals + i, val);
    }
}
