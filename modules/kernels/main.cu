#include <assert.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include "matmul.cuh"

void cudaError(cudaError_t errCode, const char * filename, int linenum) {
    if(errCode != cudaSuccess) {
        printf("Error : %s (%s : %d)\n", cudaGetErrorString(errCode), filename, linenum);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (cudaError(err, __FILE__, __LINE__))

void cuBLASError(cublasStatus_t errCode, const char * filename, int linenum) {
    if(errCode != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS Error : %d (%s : %d)\n", errCode, filename, linenum);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_CUBLAS_ERROR(err) (cuBLASError(err, __FILE__, __LINE__))

typedef void (* matmul_func) (
    const __half *, const uint32_t *,
    const uint32_t, const uint32_t, const uint32_t,
    const __half *, __half *
);

typedef void (* dequant_func) (
    const uint32_t *,
    const uint32_t, const uint32_t,
    const __half *, __half *
);

template <int s, int e>
struct get_func {
    void operator()(matmul_func func[][9][2], dequant_func dfunc[]) const {
        if constexpr (s <= e) {
            func[s][1][0] = matmul_kbit_32<1, s, false>;
            func[s][1][1] = matmul_kbit_32<1, s, true>;
            func[s][2][0] = matmul_kbit_32<2, s, false>;
            func[s][3][0] = matmul_kbit_32<3, s, false>;
            func[s][4][0] = matmul_kbit_32<4, s, false>;
            func[s][5][0] = matmul_kbit_32<5, s, false>;
            func[s][6][0] = matmul_kbit_32<6, s, false>;
            func[s][7][0] = matmul_kbit_32<7, s, false>;
            func[s][8][0] = matmul_kbit_32<8, s, false>;
            dfunc[s] = dequant_kbit_store<s>;
            get_func<s+1, e>()(func, dfunc);
        }
    }
};

__host__ void cublas_matmul(
    const int M, const int K, const int N,
    const __half * d_A, const __half * d_B, __half * d_C 
) {
    // create cuBLAS handle
    cublasHandle_t handle;
    HANDLE_CUBLAS_ERROR(cublasCreate(&handle));

    // run cuBLAS GEMM: C = A @ B
    const __half alpha = __float2half(1.0), beta = __float2half(0.0);
    HANDLE_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        N, M, K,
                        &alpha,
                        d_B, CUDA_R_16F, K,
                        d_A, CUDA_R_16F, K,
                        &beta,
                        d_C, CUDA_R_16F, N,
                        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // destory cuBLAS handle
    HANDLE_CUBLAS_ERROR(cublasDestroy(handle));

    // sync device to wait end of execution
    cudaDeviceSynchronize();
}

bool initialized = false;
bool is_orin = false;
matmul_func matmul_functions[9][9][2] = {NULL, };
dequant_func dequant_functions[9] = {NULL, };

torch::Tensor matmul_kbit(
    torch::Tensor in,
    torch::Tensor qweight,
    torch::Tensor lut,
    int w_bits
) {
    // TODO assert with size or dtype
    const int M = in.size(0);
    const int K = in.size(1);
    const int N = qweight.size(0);
    assert(M >= 1 && w_bits >= 3 && w_bits <= 8);

    if (!initialized) {
        int device;
        HANDLE_ERROR(cudaGetDevice(&device));
        cudaDeviceProp prop;
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));
        is_orin = strcmp(prop.name, "Orin") == 0;

        get_func<3, 8>()(matmul_functions, dequant_functions);

        initialized = true;
    }

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(in.device());
    at::Tensor out = torch::empty({M, N}, options);

    if (M <= 8) {
        const int multi_row = (M == 1 ? 1 : 4);
        const int use_ksplit = !is_orin && M == 1 && K > 4096 && w_bits >= 7;
        const int num_ksplit = (use_ksplit ? DIV_ROUND_UP(K, 4096) : 1);
        dim3 grid(N/(num_rows*multi_row)), block(32, num_rows, num_ksplit);

        matmul_functions[w_bits][M][use_ksplit]<<<grid, block>>>(
            (__half *)in.data_ptr<at::Half>(),
            (uint32_t *)qweight.data_ptr<int>(),
            M, N, K,
            (__half *)lut.data_ptr<at::Half>(),
            (__half *)out.data_ptr<at::Half>()
        );
    } else {
        at::Tensor dweight = torch::empty({N, K}, options);

        dim3 grid(N/num_rows), block(32, num_rows);
        dequant_functions[w_bits]<<<grid, block>>>(
            (uint32_t *)qweight.data_ptr<int>(),
            N, K,
            (__half *)lut.data_ptr<at::Half>(),
            (__half *)dweight.data_ptr<at::Half>()
        );

        cublas_matmul(
            M, K, N,
            (__half *)in.data_ptr<at::Half>(),
            (__half *)dweight.data_ptr<at::Half>(),
            (__half *)out.data_ptr<at::Half>()
        );
    }

    return out;
}

PYBIND11_MODULE(any_precision_ext, m) {
    m.def("matmul_kbit", &matmul_kbit, "kbit quantized matmul_functions");
}
