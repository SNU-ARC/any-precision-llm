#include <torch/extension.h>

#define OFFSET(N,K) ((N)*(K)/(32))
#define CHANGE_ENDIANNESS(x) x-x%4+(4-1-x%4)
void permute_bitmaps(uint32_t *bitmaps, int N, int K, int w_bits=8) {
	const int num_w_per_thread = 32; // uint32_t

	for(int i=0; i<w_bits; i++){
		uint32_t *bitmap_ = (uint32_t *)malloc(N*K/8);
		memcpy(bitmap_, &bitmaps[i*OFFSET(N,K)], N*K/8);
		int warp_size = 32;
		for (int j=0; j<K/(num_w_per_thread*warp_size); j++){
			for (int t_id=0; t_id<warp_size; t_id++){
				// 4 = 32/(sizeof(float4) / sizeof(half))
				for (int k=0; k<4; k++) {
					int orig_idx = (j*warp_size*4+t_id*4) + k;
					int new_idx = (j*warp_size*4+t_id+k*warp_size);

					for (int l=0; l<N; l++) {
						uint8_t* origpos = reinterpret_cast<uint8_t*>(&bitmaps[i*OFFSET(N,K)]);
						origpos[l*(K/8) + CHANGE_ENDIANNESS(orig_idx)]
						// reinterpret_cast<uint8_t*>(bitmaps[i*OFFSET(N,K)])[l*(K/8) + CHANGE_ENDIANNESS(orig_idx)]
							= reinterpret_cast<uint8_t*>(bitmap_)[l*(K/8) + CHANGE_ENDIANNESS(new_idx)];
					}
				}
			}
		}


		int num_remains = K%(num_w_per_thread*warp_size);
		if (num_remains != 0) {
			assert(num_remains % num_w_per_thread == 0);
			int last_warp_size = num_remains/num_w_per_thread;
			for (int t_id=0; t_id<last_warp_size; t_id++){
				// 4 = 32/(sizeof(float4) / sizeof(half))
				for (int k=0; k<4; k++) {
					int orig_idx = (K/(num_w_per_thread*warp_size))*warp_size*4+t_id*4 + k;
					int new_idx = (K/(num_w_per_thread*warp_size))*warp_size*4+t_id+k*last_warp_size;

					for (int l=0; l<N; l++) {
						// reinterpret_cast<uint8_t*>(bitmaps[i*OFFSET(N,K)])[l*(K/8) + CHANGE_ENDIANNESS(orig_idx)]
						// 	= reinterpret_cast<uint8_t*>(bitmap_)[l*(K/8) + CHANGE_ENDIANNESS(new_idx)];

							uint8_t* origpos = reinterpret_cast<uint8_t*>(&bitmaps[i*OFFSET(N,K)]);
                        origpos[l*(K/8) + CHANGE_ENDIANNESS(orig_idx)]
                            = reinterpret_cast<uint8_t*>(bitmap_)[l*(K/8) + CHANGE_ENDIANNESS(new_idx)];
					}
				}
			}
		}

		free(bitmap_);
	}
}

void preprocess_bitmaps(torch::Tensor bitmaps, int N, int K, int w_bits)
{
    uint32_t *p_bitmaps = reinterpret_cast<uint32_t*>(bitmaps.data_ptr<uint8_t>());

    permute_bitmaps(p_bitmaps, N, K, w_bits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("preprocess_bitmaps", &preprocess_bitmaps, "permute bitmap.");
}

