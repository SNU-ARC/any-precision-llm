import numpy as np
import numba
from tqdm import tqdm
import os
import torch
import logging
from multiprocessing import Pool

import utils

_bytes_per_thread = 4


#@numba.njit
def _permute_bitmaps(bitmaps):
    w_bits, N, total_bytes = bitmaps.shape
    assert total_bytes % 4 == 0, "Number of bytes must be a multiple of 4"

    threads_per_warp = 32
    bytes_per_warp = threads_per_warp * _bytes_per_thread

    # Calculate the number of full warps and the starting index of remaining bytes
    full_warps_bytes = (total_bytes // bytes_per_warp) * bytes_per_warp
    remaining_bytes_start_idx = full_warps_bytes

    # Create an array of byte indices for full warps
    full_warp_byte_indices = np.arange(full_warps_bytes)
    # Calculate new indices for full warp bytes
    new_full_warp_byte_indices = _calculate_new_indices(full_warp_byte_indices, threads_per_warp)

    remaining_bytes = total_bytes - full_warps_bytes
    # Handle remaining bytes
    if remaining_bytes:
        remaining_byte_indices = np.arange(remaining_bytes)
        # Adjust the calculation for remaining bytes, which might not fill a complete warp
        adjusted_threads_per_warp = remaining_byte_indices.size // _bytes_per_thread
        new_remaining_byte_indices = _calculate_new_indices(remaining_byte_indices,
                                                            adjusted_threads_per_warp,
                                                            offset=remaining_bytes_start_idx)

        # Combine indices - the choice to not use np.concatenate is for numba compatibility
        new_byte_indices = np.empty(total_bytes, dtype=np.int64)
        new_byte_indices[:full_warps_bytes] = new_full_warp_byte_indices
        new_byte_indices[full_warps_bytes:] = new_remaining_byte_indices
    else:
        new_byte_indices = new_full_warp_byte_indices

    permuted_bitmaps = bitmaps[:, :, np.argsort(new_byte_indices)]

    return permuted_bitmaps


#@numba.njit
def _calculate_new_indices(byte_indices, threads_per_warp, offset=0):
    """
    Calculate new byte indices for a given array of byte indices.
    """
    bytes_per_warp = threads_per_warp * _bytes_per_thread

    warp_idx, byte_offsets_within_warp = np.divmod(byte_indices, bytes_per_warp)

    warp_offsets = warp_idx * bytes_per_warp
    thread_indices = byte_indices % threads_per_warp

    # Change endianness within each thread and calculate new byte positions
    byte_offsets_within_thread = byte_offsets_within_warp // threads_per_warp
    byte_offsets_within_thread ^= 3  # Change endianness
    new_byte_indices = warp_offsets + thread_indices * _bytes_per_thread + byte_offsets_within_thread + offset

    return new_byte_indices


#@numba.njit
def _permute_bitmaps_int32(bitmaps):
    """Return a permuted version of the input bitmaps, reshaped to int32."""
    w_bits, N, total_bytes = bitmaps.shape
    bitmaps = _permute_bitmaps(bitmaps)
    return bitmaps.reshape(-1, 4).view(np.int32).reshape(w_bits, N, total_bytes // 4)


def pack(model, lut_path, output_model_path, seed_precision, parent_precision, model_type=None,
         cpu_count=None):
    if cpu_count is not None:
        cpu_count = os.cpu_count()
    model = utils.load_model(model)
    if model_type is None:
        model_type = utils.guess_model_type(model)
    state_dict = model.state_dict()
    model_weights = utils.get_model_weights(model, model_type)

    num_layers = len(model_weights)

    module_names = utils.get_module_names(model_type)
    layers_name = utils.get_layers_name(model_type)
    module_real_names = utils.get_sequential(model_type)

    pool = Pool()

    for layeridx in tqdm(range(num_layers)):
        weightpath = os.path.join(lut_path, f'weights', f'l{layeridx}.pt')
        layer_lut = torch.load(weightpath)

        for i, name in enumerate(module_names):
            N, group_count, group_size = layer_lut[name].shape
            K = group_count * group_size

            qweight_flattened = layer_lut[name].flatten()

            bitarray = np.empty((parent_precision, len(qweight_flattened) // 8), dtype=np.uint8)
            mask = 1 << (parent_precision - 1)
            for bit in range(parent_precision):
                curbitpack = np.packbits((qweight_flattened & mask).astype(bool))
                bitarray[bit] = curbitpack
                mask >>= 1

            # permute bitmap @ Section 5.3
            bitarray = bitarray.reshape((8, N, K // 8))
            weighttensor = _permute_bitmaps_int32(bitarray)
            weighttensor = torch.from_numpy(weighttensor)

            param_name = f'{layers_name}.{layeridx}.{module_real_names[i]}'
            del state_dict[param_name + '.weight']
            state_dict[param_name + '.qweight'] = weighttensor

        for bit in range(seed_precision, parent_precision + 1):
            layer_lut_path = os.path.join(lut_path, f'lut_{bit}', f'l{layeridx}.pt')
            layer_lut = torch.load(layer_lut_path)

            # shapes are (out_features, 1, in_features)

            for i, name in enumerate(module_names):
                N, group_count, group_size = layer_lut[name].shape
                K = group_count * group_size
                if group_count != 1:
                    raise NotImplementedError("Group count != 1 not implemented yet. Kernel adaptation required.")
                curLUT = np.empty((N, K), dtype=np.float16)

                for j in range(N):
                    curLUT[j] = layer_lut[name][j][0]  # the 0 here assumes group_count == 1, so one group per row

                param_name = f'{layers_name}.{layeridx}.{module_real_names[i]}'
                state_dict[param_name + '.lut' + str(bit)] = torch.tensor(curLUT)

    if not output_model_path.endswith('.pt'):
        output_model_path += '.pt'
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(state_dict, output_model_path)
    logging.info(f"Model saved to {output_model_path}")


if __name__ == '__main__':
    pack('facebook/opt-1.3b',
         '../cache/parent/(opt-1.3b)-w8_orig3-c4_s100_blk512',
         '../models/test.pt', 3, 8)
