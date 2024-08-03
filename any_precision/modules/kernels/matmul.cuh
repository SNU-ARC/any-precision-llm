#pragma once

#include <cuda_fp16.h>
#include "dequant.cuh"

/* warp-wide sum with tree-reduction */
__device__ __forceinline__ __half warp_reduce_sum(
    __half sum
) {
    #pragma unroll
    for (int i = 4; i >= 0; i--)
        sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, 1<<i));
    return sum;
}

template <int maxm, int bits, bool use_ksplit>
__global__ void matmul_kbit_32(
    const __half * I, const uint32_t * W,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const __half * C, __half * O
) {
    static_assert(maxm >= 1 && bits >= 2 && bits <= 8);
    static_assert(!use_ksplit || maxm == 1);
    constexpr bool use_half2_centroid = (bits == 3 || (bits == 4 && maxm > 1));
    constexpr int multi_row = (maxm == 1 ? 1 : 4);

    constexpr int num_centroids = 1 << bits, warp_size = 32;
    constexpr int shC_siz = (use_half2_centroid ? num_centroids * num_centroids * 2 : num_centroids);
    constexpr int q_w_siz = (use_half2_centroid ? 4 : 8);

    const uint32_t row_idx_base = blockIdx.x * num_rows * multi_row + threadIdx.y;
    const int centroid_idx_base = threadIdx.y * (use_half2_centroid ? num_centroids * num_centroids : num_centroids);

    __shared__ __half shC[num_rows * multi_row * shC_siz];

    if (!use_ksplit || threadIdx.z == 0) {
        #pragma unroll
        for (int h = 0; h < multi_row; h++) {
            const uint32_t row_idx = row_idx_base + h * num_rows;
            const int centroid_idx = centroid_idx_base + h * num_rows * (use_half2_centroid ? num_centroids * num_centroids : num_centroids);
            if constexpr (use_half2_centroid) {
                const int xx = threadIdx.x % num_centroids, yy = threadIdx.x / num_centroids;
                const __half fragCX = C[row_idx * num_centroids | xx];
                #pragma unroll
                for (int i = 0; i < shC_siz / warp_size / 2; i++) {
                    const int yidx = yy | (i * warp_size / num_centroids);
                    const __half fragCY = C[row_idx * num_centroids | yidx];
                    ((__half2 * )shC)[centroid_idx | (yidx * num_centroids) | xx] = make_half2(fragCY, fragCX);
                }
            } else if constexpr (bits < 6) {
                if (threadIdx.x < num_centroids)
                    shC[centroid_idx + threadIdx.x] = C[num_centroids * row_idx + threadIdx.x];
            } else if constexpr (bits == 6) {
                ((__half2 *)shC)[centroid_idx / 2 + threadIdx.x] = ((__half2 *)C)[num_centroids * row_idx / 2 + threadIdx.x];
            } else if constexpr (bits == 7) {
                ((float2 *)shC)[centroid_idx / 4 + threadIdx.x] = ((float2 *)C)[num_centroids * row_idx / 4 + threadIdx.x];
            } else if constexpr (bits == 8) {
                ((float4 *)shC)[centroid_idx / 8 + threadIdx.x] = ((float4 *)C)[num_centroids * row_idx / 8 + threadIdx.x];
            }
        }
    }
    __syncthreads();

    int eff_warp_size = warp_size;
    __half partial_sum[maxm * multi_row] = {__float2half(0.0), };
    uint32_t q[bits], q_w[q_w_siz];
    __half2 dq_w[16];

    int mini = (use_ksplit ? threadIdx.z * 4 : 0);
    int maxi = DIV_ROUND_UP(K, 32 * warp_size);
    if (use_ksplit && maxi > mini + 4) maxi = mini + 4;
    for (int i = mini; i < maxi; i++) {
        if (i == K / (32 * warp_size)) {
            eff_warp_size = (K % (32 * warp_size)) / 32;
            if (threadIdx.x >= eff_warp_size) break;
        }

        #pragma unroll
        for (int h = 0; h < multi_row; h++) {
            const uint32_t row_idx = row_idx_base + h * num_rows;
            const int centroid_idx = centroid_idx_base + h * num_rows * (use_half2_centroid ? num_centroids * num_centroids : num_centroids);

            // load quantized weight
            #pragma unroll
            for (int j = 0; j < bits; j++) {
                const int k = (j * N + row_idx) * (K / 32) + i * 32 + threadIdx.x;
                q[j] = W[k];
            }

            // dequantize
            dequant<bits, use_half2_centroid>(q, q_w);

            // lookup
            #pragma unroll
            for (int j = 3; j >= 0; j--) {
                if constexpr (use_half2_centroid) {
                    #pragma unroll
                    for (int k = 0; k < 2; k++) {
                        const __half2 x = ((__half2 *)shC)[centroid_idx | (q_w[k*2+0] & 0xff)];
                        const __half2 y = ((__half2 *)shC)[centroid_idx | (q_w[k*2+1] & 0xff)];
                        dq_w[j * 4 + k + 0] = make_half2(x.x, y.x);
                        dq_w[j * 4 + k + 2] = make_half2(x.y, y.y);
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        const __half x = shC[centroid_idx | (q_w[k*2+0] & 0xff)];
                        const __half y = shC[centroid_idx | (q_w[k*2+1] & 0xff)];
                        dq_w[j * 4 + k] = make_half2(x, y);
                    }
                }
                #pragma unroll
                for (int k = 0; k < q_w_siz; k++)
                    q_w[k] >>= 8;
            }

            // accumulate
            #pragma unroll
            for (int l = 0; l < maxm; l++) {
                __half2 sum = make_half2(__float2half(0.0), __float2half(0.0));
                #pragma unroll
                for (int j = 3; j >= 0; j--) {
                    const int idx = (l*K/8 + eff_warp_size*j) + i*warp_size*4 + threadIdx.x;
                    float4 in_buf = ((float4 *)I)[idx];
                    __half2 * in_half = (__half2 *)&in_buf;
                    #pragma unroll
                    for (int k = 0; k < 4; k++)
                        sum = __hfma2(dq_w[j * 4 + k], in_half[k], sum);
                }
                partial_sum[l + h * maxm] = __hadd(partial_sum[l + h * maxm], __hadd(sum.x, sum.y));
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < maxm * multi_row; i++)
        partial_sum[i] = warp_reduce_sum(partial_sum[i]);

    if constexpr (use_ksplit) {
        __shared__ __half shO[maxm * multi_row * num_rows];
        if (threadIdx.x == 0 && threadIdx.z == 0)
            #pragma unroll
            for (int j = 0; j < multi_row; j++)
                shO[j + threadIdx.y * multi_row] = __float2half(0.0);
        __syncthreads();
        if (threadIdx.x == 0)
            #pragma unroll
            for (int j = 0; j < multi_row; j++)
                atomicAdd(shO + j + threadIdx.y * multi_row, partial_sum[j]);
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.z == 0)
            #pragma unroll
            for (int j = 0; j < multi_row; j++)
                partial_sum[j] = shO[j + threadIdx.y * multi_row];
    }

    if (threadIdx.x == 0 && (!use_ksplit || threadIdx.z == 0)) {
        #pragma unroll
        for (int i = 0; i < maxm; i++) {
            #pragma unroll
            for (int j = 0; j < multi_row; j++) {
                const uint32_t row_idx = row_idx_base + j * num_rows;
                O[i * N + row_idx] = partial_sum[i + j * maxm];
            }
        }
    }
}
