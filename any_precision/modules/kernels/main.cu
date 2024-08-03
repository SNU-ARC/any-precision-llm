#include <assert.h>
#include <torch/extension.h>
#include "matmul.cuh"

void cudaError(cudaError_t errCode, const char * filename, int linenum) {
    if(errCode != cudaSuccess) {
        printf("Error : %s (%s : %d)\n", cudaGetErrorString(errCode), filename, linenum);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (cudaError(err, __FILE__, __LINE__))

typedef void (* matmul_func) (
    const __half *, const uint32_t *,
    const uint32_t, const uint32_t, const uint32_t,
    const __half *, __half *
);

template <int s, int e>
struct get_matmul_func {
    void operator()(matmul_func func[][9][2]) const {
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
            get_matmul_func<s+1, e>()(func);
        }
    }
};

typedef void (* dequant_func) (
    const uint32_t *,
    const uint32_t, const uint32_t,
    const __half *, __half *
);

template <int s, int e>
struct get_dequant_func {
    void operator()(dequant_func func[]) const {
        if constexpr (s <= e) {
            func[s] = dequant_kbit_store<s>;
            get_dequant_func<s+1, e>()(func);
        }
    }
};

bool dequant_initalized = false;
bool matmul_initialized = false;
bool is_orin = false;
matmul_func matmul_functions[9][9][2] = {NULL, };
dequant_func dequant_functions[9] = {NULL, };

torch::Tensor dequant_kbit(
    torch::Tensor qweight,
    torch::Tensor lut,
    int w_bits
) {
    assert(qweight.ndimension() == 3 && qweight.dtype() == torch::kInt && lut.dtype() == torch::kHalf);
    assert(qweight.device() == lut.device() && qweight.is_cuda());
    assert(w_bits >= 2 && w_bits <= 8);
    const int N = qweight.size(1);
    const int K = qweight.size(2) * 32;

    if (!dequant_initalized) {
        get_dequant_func<2, 8>()(dequant_functions);
        dequant_initalized = true;
    }

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(qweight.device());
    at::Tensor weight = torch::empty({N, K}, options);

    dim3 grid(N/num_rows), block(32, num_rows);
    dequant_functions[w_bits]<<<grid, block>>>(
        (uint32_t *)qweight.data_ptr<int>(),
        N, K,
        (__half *)lut.data_ptr<at::Half>(),
        (__half *)weight.data_ptr<at::Half>()
    );

    return weight;
}

torch::Tensor matmul_kbit(
    torch::Tensor in,
    torch::Tensor qweight,
    torch::Tensor lut,
    int w_bits
) {
    const int N = qweight.size(1);
    const int K = qweight.size(2) * 32;
    int64_t in_ndim = in.ndimension();
    const int M = in.numel() / K;

    // TODO assert with size or dtype
    assert(M >= 1 && M <= 8 && w_bits >= 2 && w_bits <= 8);
    assert(in.device() == qweight.device() && in.device() == lut.device() && in.is_cuda());
    assert(qweight.ndimension() == 3 && qweight.dtype() == torch::kInt && lut.dtype() == torch::kHalf);
    assert(in.dtype() == torch::kHalf);

    if (!matmul_initialized) {
        int device;
        HANDLE_ERROR(cudaGetDevice(&device));
        cudaDeviceProp prop;
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));
        is_orin = strcmp(prop.name, "Orin") == 0;

        get_matmul_func<2, 8>()(matmul_functions);
        matmul_initialized = true;
    }

    auto sizes = in.sizes().vec();
    sizes.at(in_ndim - 1) = N;
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(in.device());
    at::Tensor out = torch::empty(sizes, options);

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

    return out;
}

PYBIND11_MODULE(any_precision_ext, m) {
    m.def("matmul_kbit", &matmul_kbit, "kbit quantized matmul_function");
    m.def("dequant_kbit", &dequant_kbit, "kbit dequantize function");
}
