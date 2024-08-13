import numpy as np
from tqdm import tqdm
import os
import torch
import logging
from multiprocessing import Pool
import numba

_bytes_per_thread = 4


@numba.njit(cache=True)
def _permute_bitmaps(bitmaps):
    _, _, total_bytes = bitmaps.shape
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


@numba.njit(cache=True)
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


@numba.njit(cache=True)
def _permute_bitmaps_int32(bitmaps):
    """Return a permuted version of the input bitmaps, reshaped to int32."""
    w_bits, N, total_bytes = bitmaps.shape
    bitmaps = _permute_bitmaps(bitmaps)
    return bitmaps.reshape(-1, 4).view(np.int32).reshape(w_bits, N, total_bytes // 4)


def _process_layer_data(args):
    layer_idx, lut_path, model_name, layers_name, module_names, parent_precision, seed_precision, dns = args
    layer_data = {}
    layer_dns = {}

    weightpath = os.path.join(lut_path, 'weights', f'l{layer_idx}.pt')
    layer_weights = torch.load(weightpath)

    if dns:
        sparsepath = os.path.join(lut_path, 'sparse', f'l{layer_idx}.pt')
        layer_sparse = torch.load(sparsepath)

    for i, name in enumerate(module_names):
        N, group_count, group_size = layer_weights[name].shape
        K = group_count * group_size

        qweight_flattened = layer_weights[name].flatten()
        bitarray = np.empty((parent_precision, len(qweight_flattened) // 8), dtype=np.uint8)
        mask = 1 << (parent_precision - 1)  # MSB first
        for bit in range(parent_precision):
            curbitpack = np.packbits((qweight_flattened & mask).astype(bool))
            bitarray[bit] = curbitpack
            mask >>= 1

        bitarray = bitarray.reshape((parent_precision, N, K // 8))
        weighttensor = _permute_bitmaps_int32(bitarray)

        param_name = f'{model_name}.{layers_name}.{layer_idx}.{name}'
        layer_data[param_name + '.qweight'] = weighttensor
        
        if dns: 
            sp_mat = layer_sparse[name].to_dense().to_sparse(layout=torch.sparse_csr)
            layer_data[param_name + ".rows"] = sp_mat.crow_indices().to(torch.int32).cpu().data.numpy()
            layer_data[param_name + ".cols"] = sp_mat.col_indices().to(torch.int32).cpu().data.numpy()
            layer_data[param_name + ".vals"] = sp_mat.values().to(torch.float16).cpu().data.numpy()
            layer_dns[param_name] = len(sp_mat.values())

        for bit in range(seed_precision, parent_precision + 1):
            layer_lut_path = os.path.join(lut_path, f'lut_{bit}', f'l{layer_idx}.pt')
            layer_lut = torch.load(layer_lut_path)

            curLUT = np.empty((N, 2 ** bit), dtype=np.float16)
            for r_idx in range(N):
                curLUT[r_idx] = layer_lut[name][r_idx][0]  # the 0 here assumes group_count == 1

            layer_data[param_name + '.lut' + str(bit)] = curLUT

    return layer_idx, layer_data, layer_dns


def pack(
        analyzer,
        lut_path,
        output_model_path,
        seed_precision,
        parent_precision,
        group_count=1,
        dns=False,
        cpu_count=None
):

    if group_count != 1:
        raise NotImplementedError("Group counts other than 1 are not supported yet for packing")

    # if dns:
    #     raise NotImplementedError("D&S packing is not supported yet")

    if cpu_count is None:
        cpu_count = os.cpu_count()

    # Limit cpu_count to 8 as larger values use too much memory, without much speedup
    _max_cpu_count = 8
    if cpu_count > _max_cpu_count:
        logging.warning(f"cpu_count will be limited to 8 to avoid excessive memory usage. "
                        f"Original value: {cpu_count}")
        cpu_count = _max_cpu_count

    tokenizer = analyzer.tokenizer

    num_layers = analyzer.num_layers

    model_name = analyzer.model_name
    layers_name = analyzer.layers_name
    module_names = analyzer.module_names
    config = analyzer.config  # original model config
    arch_config = analyzer.get_arch_config()

    state_dict = analyzer.state_dict

    args_list = [(layer_idx, lut_path, model_name, layers_name, module_names, parent_precision, seed_precision, dns) for
                 layer_idx in range(num_layers)]

    # with Pool(cpu_count) as pool:
    #     for layer_idx, layer_data in tqdm(pool.imap(_process_layer_data, args_list), total=num_layers, desc="Packing"):
    #         for key, value in layer_data.items():
    #             state_dict[key] = torch.from_numpy(value)  # Update with modified weights

    sparse_info = {}
    for task in args_list:
        layer_idx, layer_data, layer_dns = _process_layer_data(task)
        for key, value in layer_data.items():
            state_dict[key] = torch.from_numpy(value)  # Update with modified weights
        if dns:
            for key, value in layer_dns.items():
                sparse_info[key] = value

    # add new config parameters
    anyprec_configs = {
        'seed_precision': seed_precision,
        'parent_precision': parent_precision,
        'group_count': group_count,
        'arch_config': arch_config,
        'sparse_numvals': sparse_info,
    }
    config.anyprec = anyprec_configs

    logging.info(f"Writing model to disk...")
    os.makedirs(output_model_path, exist_ok=True)
    torch.save(state_dict, os.path.join(output_model_path, 'pytorch_model.bin'))
    tokenizer.save_pretrained(output_model_path)
    config.save_pretrained(output_model_path)
    logging.info(f"Model saved to {output_model_path}")
