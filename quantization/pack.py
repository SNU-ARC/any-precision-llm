import numpy as np

_bytes_per_thread = 4


def permute_bitmaps(bitmaps):
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
    new_full_warp_byte_indices = calculate_new_indices(full_warp_byte_indices, threads_per_warp)

    # Handle remaining bytes
    if total_bytes > full_warps_bytes:
        remaining_byte_indices = np.arange(remaining_bytes_start_idx, total_bytes)
        # Adjust the calculation for remaining bytes, which might not fill a complete warp
        adjusted_threads_per_warp = remaining_byte_indices.size // _bytes_per_thread
        new_remaining_byte_indices = calculate_new_indices(remaining_byte_indices,
                                                           adjusted_threads_per_warp,
                                                           offset=remaining_bytes_start_idx)
        # Combine indices
        new_byte_indices = np.concatenate([new_full_warp_byte_indices, new_remaining_byte_indices])
    else:
        new_byte_indices = new_full_warp_byte_indices

    # Use advanced indexing to permute the bitmaps
    permuted_bitmaps = bitmaps[:, :, new_byte_indices]

    return permuted_bitmaps


def calculate_new_indices(byte_indices, threads_per_warp, offset=0):
    """
    Calculate new byte indices for a given array of byte indices.
    """
    bytes_per_warp = threads_per_warp * _bytes_per_thread
    warp_offsets = (byte_indices // bytes_per_warp) * bytes_per_warp
    thread_indices = (byte_indices % bytes_per_warp) // _bytes_per_thread

    # Change endianness within each thread and calculate new byte positions
    byte_offsets_within_thread = byte_indices % _bytes_per_thread
    byte_offsets_within_thread ^= 3  # Change endianness
    new_byte_indices = warp_offsets + thread_indices * _bytes_per_thread + byte_offsets_within_thread + offset

    return new_byte_indices
