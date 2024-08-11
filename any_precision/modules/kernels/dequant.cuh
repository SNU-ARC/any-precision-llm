#pragma once

#include <stdint.h>

/* macros */

#define num_rows 4
#define DIV_ROUND_UP(x,y) (((x)+(y)-1)/(y))

template <int, bool>
__device__ __forceinline__ void dequant(const uint32_t q[], uint32_t q_w[]);


template <>
__device__ __forceinline__ void dequant<2, false>(const uint32_t q[2], uint32_t q_w[8]) {
    constexpr uint32_t mask0 = 0x88888888;
    constexpr uint32_t mask1 = 0x44444444;
    constexpr uint32_t mask2 = 0x22222222;
    constexpr uint32_t mask3 = 0x11111111;

    q_w[0] = (((q[0]&mask0)) | ((q[1]&mask0) >> 1)) >> 2;
    q_w[1] = (((q[0]&mask1)) | ((q[1]&mask1) >> 1)) >> 1;
    q_w[2] = (q[0]&mask2) | ((q[1]&mask2) >> 1);
    q_w[3] = ((q[0]&mask3) << 1) | (q[1]&mask3);

    constexpr uint32_t mask = 0x03030303;
    q_w[4] = q_w[0] & mask;
    q_w[5] = q_w[1] & mask;
    q_w[6] = q_w[2] & mask;
    q_w[7] = q_w[3] & mask;

    q_w[0] = (q_w[0] >> 4) & mask;
    q_w[1] = (q_w[1] >> 4) & mask;
    q_w[2] = (q_w[2] >> 4) & mask;
    q_w[3] = (q_w[3] >> 4) & mask;
}


template <>
__device__ __forceinline__ void dequant<3, true>(const uint32_t q[3], uint32_t q_w[4]) {
    constexpr uint32_t mask0 = 0x88888888;
    constexpr uint32_t mask1 = 0x44444444;
    constexpr uint32_t mask2 = 0x22222222;
    constexpr uint32_t mask3 = 0x11111111;

    // fast transpose
    q_w[0] = (((q[0]&mask0)) | ((q[1]&mask0) >> 1) | ((q[2]&mask0)>>2))>>1;
    q_w[1] = ((q[0]&mask1)) | ((q[1]&mask1) >> 1) | ((q[2]&mask1)>>2);
    q_w[2] = ((q[0]&mask2) << 1) | ((q[1]&mask2)) | ((q[2]&mask2)>>1);
    q_w[3] = ((q[0]&mask3) << 2) | ((q[1]&mask3) << 1) | ((q[2]&mask3));

    // table lookup merge
    #pragma unroll
    for (int i = 0; i < 4; i++)
        q_w[i] = (q_w[i] & 0x0f0f0f0f) | ((q_w[i] & 0xf0f0f0f0) >> 1);
}

template <>
__device__ __forceinline__ void dequant<3, false>(const uint32_t q[3], uint32_t q_w[8]) {
    constexpr uint32_t mask0 = 0x88888888;
    constexpr uint32_t mask1 = 0x44444444;
    constexpr uint32_t mask2 = 0x22222222;
    constexpr uint32_t mask3 = 0x11111111;

    q_w[0] = (((q[0]&mask0)) | ((q[1]&mask0) >> 1) | ((q[2]&mask0)>>2))>>1;
    q_w[1] = ((q[0]&mask1)) | ((q[1]&mask1) >> 1) | ((q[2]&mask1)>>2);
    q_w[2] = ((q[0]&mask2) << 1) | ((q[1]&mask2)) | ((q[2]&mask2)>>1);
    q_w[3] = ((q[0]&mask3) << 2) | ((q[1]&mask3) << 1) | ((q[2]&mask3));

    constexpr uint32_t mask = 0x0f0f0f0f;
    q_w[4] = q_w[0] & mask;
    q_w[5] = q_w[1] & mask;
    q_w[6] = q_w[2] & mask;
    q_w[7] = q_w[3] & mask;

    q_w[0] = (q_w[0] >> 4) & mask;
    q_w[1] = (q_w[1] >> 4) & mask;
    q_w[2] = (q_w[2] >> 4) & mask;
    q_w[3] = (q_w[3] >> 4) & mask;
}

template <>
__device__ __forceinline__ void dequant<4, true>(const uint32_t q[4], uint32_t q_w[4]) {
    constexpr uint32_t mask0 = 0x88888888;
    constexpr uint32_t mask1 = 0x44444444;
    constexpr uint32_t mask2 = 0x22222222;
    constexpr uint32_t mask3 = 0x11111111;

    q_w[0] = ((q[0]&mask0)) | ((q[1]&mask0) >> 1) | ((q[2]&mask0)>>2) | ((q[3]&mask0) >> 3);
    q_w[1] = ((q[0]&mask1) << 1) | (q[1]&mask1) | ((q[2]&mask1)>>1) | ((q[3]&mask1) >> 2);
    q_w[2] = ((q[0]&mask2) << 2) | ((q[1]&mask2) << 1) | (q[2]&mask2) | ((q[3]&mask2) >> 1);
    q_w[3] = ((q[0]&mask3) << 3) | ((q[1]&mask3) << 2) | ((q[2]&mask3) << 1) | (q[3]&mask3);
}

template <>
__device__ __forceinline__ void dequant<4, false>(const uint32_t q[4], uint32_t q_w[8]) {
    constexpr uint32_t mask0 = 0x88888888;
    constexpr uint32_t mask1 = 0x44444444;
    constexpr uint32_t mask2 = 0x22222222;
    constexpr uint32_t mask3 = 0x11111111;

    q_w[0] = ((q[0]&mask0)) | ((q[1]&mask0) >> 1) | ((q[2]&mask0)>>2) | ((q[3]&mask0) >> 3);
    q_w[1] = ((q[0]&mask1) << 1) | (q[1]&mask1) | ((q[2]&mask1)>>1) | ((q[3]&mask1) >> 2);
    q_w[2] = ((q[0]&mask2) << 2) | ((q[1]&mask2) << 1) | (q[2]&mask2) | ((q[3]&mask2) >> 1);
    q_w[3] = ((q[0]&mask3) << 3) | ((q[1]&mask3) << 2) | ((q[2]&mask3) << 1) | (q[3]&mask3);

    constexpr uint32_t mask = 0x0f0f0f0f;
    q_w[4] = q_w[0] & mask;
    q_w[5] = q_w[1] & mask;
    q_w[6] = q_w[2] & mask;
    q_w[7] = q_w[3] & mask;

    q_w[0] = (q_w[0] >> 4) & mask;
    q_w[1] = (q_w[1] >> 4) & mask;
    q_w[2] = (q_w[2] >> 4) & mask;
    q_w[3] = (q_w[3] >> 4) & mask;
}

template <>
__device__ __forceinline__ void dequant<8, false>(const uint32_t q[8], uint32_t q_w[8]) {
    constexpr uint32_t mask0 = 0x80808080;
    constexpr uint32_t mask1 = 0x40404040;
    constexpr uint32_t mask2 = 0x20202020;
    constexpr uint32_t mask3 = 0x10101010;
    constexpr uint32_t mask4 = 0x08080808;
    constexpr uint32_t mask5 = 0x04040404;
    constexpr uint32_t mask6 = 0x02020202;
    constexpr uint32_t mask7 = 0x01010101;

    q_w[0] = ((q[0]&mask0)>>0) | ((q[1]&mask0)>>1) | ((q[2]&mask0)>>2) | ((q[3]&mask0)>>3) | ((q[4]&mask0)>>4) | ((q[5]&mask0)>>5) | ((q[6]&mask0)>>6) | ((q[7]&mask0)>>7);
    q_w[1] = ((q[0]&mask1)<<1) | ((q[1]&mask1)>>0) | ((q[2]&mask1)>>1) | ((q[3]&mask1)>>2) | ((q[4]&mask1)>>3) | ((q[5]&mask1)>>4) | ((q[6]&mask1)>>5) | ((q[7]&mask1)>>6);
    q_w[2] = ((q[0]&mask2)<<2) | ((q[1]&mask2)<<1) | ((q[2]&mask2)>>0) | ((q[3]&mask2)>>1) | ((q[4]&mask2)>>2) | ((q[5]&mask2)>>3) | ((q[6]&mask2)>>4) | ((q[7]&mask2)>>5);
    q_w[3] = ((q[0]&mask3)<<3) | ((q[1]&mask3)<<2) | ((q[2]&mask3)<<1) | ((q[3]&mask3)>>0) | ((q[4]&mask3)>>1) | ((q[5]&mask3)>>2) | ((q[6]&mask3)>>3) | ((q[7]&mask3)>>4);
    q_w[4] = ((q[0]&mask4)<<4) | ((q[1]&mask4)<<3) | ((q[2]&mask4)<<2) | ((q[3]&mask4)<<1) | ((q[4]&mask4)>>0) | ((q[5]&mask4)>>1) | ((q[6]&mask4)>>2) | ((q[7]&mask4)>>3);
    q_w[5] = ((q[0]&mask5)<<5) | ((q[1]&mask5)<<4) | ((q[2]&mask5)<<3) | ((q[3]&mask5)<<2) | ((q[4]&mask5)<<1) | ((q[5]&mask5)>>0) | ((q[6]&mask5)>>1) | ((q[7]&mask5)>>2);
    q_w[6] = ((q[0]&mask6)<<6) | ((q[1]&mask6)<<5) | ((q[2]&mask6)<<4) | ((q[3]&mask6)<<3) | ((q[4]&mask6)<<2) | ((q[5]&mask6)<<1) | ((q[6]&mask6)>>0) | ((q[7]&mask6)>>1);
    q_w[7] = ((q[0]&mask7)<<7) | ((q[1]&mask7)<<6) | ((q[2]&mask7)<<5) | ((q[3]&mask7)<<4) | ((q[4]&mask7)<<3) | ((q[5]&mask7)<<2) | ((q[6]&mask7)<<1) | ((q[7]&mask7)>>0);
}

template <>
__device__ __forceinline__ void dequant<7, false>(const uint32_t q[7], uint32_t q_w[8]) {
    constexpr uint32_t mask0 = 0x80808080;
    constexpr uint32_t mask1 = 0x40404040;
    constexpr uint32_t mask2 = 0x20202020;
    constexpr uint32_t mask3 = 0x10101010;
    constexpr uint32_t mask4 = 0x08080808;
    constexpr uint32_t mask5 = 0x04040404;
    constexpr uint32_t mask6 = 0x02020202;
    constexpr uint32_t mask7 = 0x01010101;

    q_w[0] = ((q[0]&mask0)>>1) | ((q[1]&mask0)>>2) | ((q[2]&mask0)>>3) | ((q[3]&mask0)>>4) | ((q[4]&mask0)>>5) | ((q[5]&mask0)>>6) | ((q[6]&mask0)>>7);
    q_w[1] = ((q[0]&mask1)>>0) | ((q[1]&mask1)>>1) | ((q[2]&mask1)>>2) | ((q[3]&mask1)>>3) | ((q[4]&mask1)>>4) | ((q[5]&mask1)>>5) | ((q[6]&mask1)>>6);
    q_w[2] = ((q[0]&mask2)<<1) | ((q[1]&mask2)>>0) | ((q[2]&mask2)>>1) | ((q[3]&mask2)>>2) | ((q[4]&mask2)>>3) | ((q[5]&mask2)>>4) | ((q[6]&mask2)>>5);
    q_w[3] = ((q[0]&mask3)<<2) | ((q[1]&mask3)<<1) | ((q[2]&mask3)>>0) | ((q[3]&mask3)>>1) | ((q[4]&mask3)>>2) | ((q[5]&mask3)>>3) | ((q[6]&mask3)>>4);
    q_w[4] = ((q[0]&mask4)<<3) | ((q[1]&mask4)<<2) | ((q[2]&mask4)<<1) | ((q[3]&mask4)>>0) | ((q[4]&mask4)>>1) | ((q[5]&mask4)>>2) | ((q[6]&mask4)>>3);
    q_w[5] = ((q[0]&mask5)<<4) | ((q[1]&mask5)<<3) | ((q[2]&mask5)<<2) | ((q[3]&mask5)<<1) | ((q[4]&mask5)>>0) | ((q[5]&mask5)>>1) | ((q[6]&mask5)>>2);
    q_w[6] = ((q[0]&mask6)<<5) | ((q[1]&mask6)<<4) | ((q[2]&mask6)<<3) | ((q[3]&mask6)<<2) | ((q[4]&mask6)<<1) | ((q[5]&mask6)>>0) | ((q[6]&mask6)>>1);
    q_w[7] = ((q[0]&mask7)<<6) | ((q[1]&mask7)<<5) | ((q[2]&mask7)<<4) | ((q[3]&mask7)<<3) | ((q[4]&mask7)<<2) | ((q[5]&mask7)<<1) | ((q[6]&mask7)>>0);
}

template <>
__device__ __forceinline__ void dequant<6, false>(const uint32_t q[6], uint32_t q_w[8]) {
    constexpr uint32_t mask0 = 0x80808080;
    constexpr uint32_t mask1 = 0x40404040;
    constexpr uint32_t mask2 = 0x20202020;
    constexpr uint32_t mask3 = 0x10101010;
    constexpr uint32_t mask4 = 0x08080808;
    constexpr uint32_t mask5 = 0x04040404;
    constexpr uint32_t mask6 = 0x02020202;
    constexpr uint32_t mask7 = 0x01010101;

    q_w[0] = ((q[0]&mask0)>>2) | ((q[1]&mask0)>>3) | ((q[2]&mask0)>>4) | ((q[3]&mask0)>>5) | ((q[4]&mask0)>>6) | ((q[5]&mask0)>>7);
    q_w[1] = ((q[0]&mask1)>>1) | ((q[1]&mask1)>>2) | ((q[2]&mask1)>>3) | ((q[3]&mask1)>>4) | ((q[4]&mask1)>>5) | ((q[5]&mask1)>>6);
    q_w[2] = ((q[0]&mask2)>>0) | ((q[1]&mask2)>>1) | ((q[2]&mask2)>>2) | ((q[3]&mask2)>>3) | ((q[4]&mask2)>>4) | ((q[5]&mask2)>>5);
    q_w[3] = ((q[0]&mask3)<<1) | ((q[1]&mask3)>>0) | ((q[2]&mask3)>>1) | ((q[3]&mask3)>>2) | ((q[4]&mask3)>>3) | ((q[5]&mask3)>>4);
    q_w[4] = ((q[0]&mask4)<<2) | ((q[1]&mask4)<<1) | ((q[2]&mask4)>>0) | ((q[3]&mask4)>>1) | ((q[4]&mask4)>>2) | ((q[5]&mask4)>>3);
    q_w[5] = ((q[0]&mask5)<<3) | ((q[1]&mask5)<<2) | ((q[2]&mask5)<<1) | ((q[3]&mask5)>>0) | ((q[4]&mask5)>>1) | ((q[5]&mask5)>>2);
    q_w[6] = ((q[0]&mask6)<<4) | ((q[1]&mask6)<<3) | ((q[2]&mask6)<<2) | ((q[3]&mask6)<<1) | ((q[4]&mask6)>>0) | ((q[5]&mask6)>>1);
    q_w[7] = ((q[0]&mask7)<<5) | ((q[1]&mask7)<<4) | ((q[2]&mask7)<<3) | ((q[3]&mask7)<<2) | ((q[4]&mask7)<<1) | ((q[5]&mask7)<<0);
}

template <>
__device__ __forceinline__ void dequant<5, false>(const uint32_t q[5], uint32_t q_w[8]) {
    constexpr uint32_t mask0 = 0x80808080;
    constexpr uint32_t mask1 = 0x40404040;
    constexpr uint32_t mask2 = 0x20202020;
    constexpr uint32_t mask3 = 0x10101010;
    constexpr uint32_t mask4 = 0x08080808;
    constexpr uint32_t mask5 = 0x04040404;
    constexpr uint32_t mask6 = 0x02020202;
    constexpr uint32_t mask7 = 0x01010101;

    q_w[0] = ((q[0]&mask0)>>3) | ((q[1]&mask0)>>4) | ((q[2]&mask0)>>5) | ((q[3]&mask0)>>6) | ((q[4]&mask0)>>7);
    q_w[1] = ((q[0]&mask1)>>2) | ((q[1]&mask1)>>3) | ((q[2]&mask1)>>4) | ((q[3]&mask1)>>5) | ((q[4]&mask1)>>6);
    q_w[2] = ((q[0]&mask2)>>1) | ((q[1]&mask2)>>2) | ((q[2]&mask2)>>3) | ((q[3]&mask2)>>4) | ((q[4]&mask2)>>5);
    q_w[3] = ((q[0]&mask3)>>0) | ((q[1]&mask3)>>1) | ((q[2]&mask3)>>2) | ((q[3]&mask3)>>3) | ((q[4]&mask3)>>4);
    q_w[4] = ((q[0]&mask4)<<1) | ((q[1]&mask4)>>0) | ((q[2]&mask4)>>1) | ((q[3]&mask4)>>2) | ((q[4]&mask4)>>3);
    q_w[5] = ((q[0]&mask5)<<2) | ((q[1]&mask5)<<1) | ((q[2]&mask5)>>0) | ((q[3]&mask5)>>1) | ((q[4]&mask5)>>2);
    q_w[6] = ((q[0]&mask6)<<3) | ((q[1]&mask6)<<2) | ((q[2]&mask6)<<1) | ((q[3]&mask6)>>0) | ((q[4]&mask6)>>1);
    q_w[7] = ((q[0]&mask7)<<4) | ((q[1]&mask7)<<3) | ((q[2]&mask7)<<2) | ((q[3]&mask7)<<1) | ((q[4]&mask7)>>0);
}

template <int bits>
__global__ void dequant_kbit_store(
    const uint32_t * W,
    const uint32_t N, const uint32_t K,
    const __half * C, __half * O
) {
    static_assert(bits >= 2 && bits <= 8);
    constexpr int num_centroids = 1 << bits, warp_size = 32;

    const uint32_t row_idx = blockIdx.x * num_rows + threadIdx.y;
    const int centroid_idx = threadIdx.y * num_centroids;

    __shared__ __half shC[num_rows * num_centroids];

    if constexpr (bits < 6) {
        if (threadIdx.x < num_centroids)
            shC[centroid_idx + threadIdx.x] = C[num_centroids * row_idx + threadIdx.x];
    } else if constexpr (bits == 6) {
        ((half2 *)shC)[centroid_idx / 2 + threadIdx.x] = ((half2 *)C)[num_centroids * row_idx / 2 + threadIdx.x];
    } else if constexpr (bits == 7) {
        ((float2 *)shC)[centroid_idx / 4 + threadIdx.x] = ((float2 *)C)[num_centroids * row_idx / 4 + threadIdx.x];
    } else if constexpr (bits == 8) {
        ((float4 *)shC)[centroid_idx / 8 + threadIdx.x] = ((float4 *)C)[num_centroids * row_idx / 8 + threadIdx.x];
    }
    __syncthreads();

    int eff_warp_size = warp_size;
    uint32_t q[bits], q_w[8];
    half2 dq_w[16];

    const uint32_t maxi = DIV_ROUND_UP(K, 32 * warp_size);
    for (int i = 0; i < maxi; i++) {
        if (i == K / (32 * warp_size)) {
            eff_warp_size = (K % (32 * warp_size)) / 32;
            if (threadIdx.x >= eff_warp_size) break;
        }

        // W: (support_bits, N, K//32)
        // load quantized weight
        #pragma unroll
        for (int j = 0; j < bits; j++) {
            const int k = (j * N + row_idx) * (K / 32) + i * warp_size + threadIdx.x;
            q[j] = W[k];
        }

        // dequantize
        dequant<bits, false>(q, q_w);

        // lookup
        #pragma unroll
        for (int j = 3; j >= 0; j--) {
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                const __half x = shC[centroid_idx | (q_w[k*2+0] & 0xff)];
                const __half y = shC[centroid_idx | (q_w[k*2+1] & 0xff)];
                dq_w[j * 4 + k] = make_half2(x, y);
            }
            #pragma unroll
            for (int k = 0; k < 8; k++)
                q_w[k] >>= 8;
        }

        #pragma unroll
        for (int j = 0; j < 4; j++)
            ((float4 *)O)[(row_idx*K + 8*eff_warp_size*j + i*warp_size*32 + 8*threadIdx.x)/8] = ((float4 *)dq_w)[j];
    }
}
