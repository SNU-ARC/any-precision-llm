import os
import logging
import torch
import numpy as np
from tqdm import tqdm
import numba
from concurrent.futures import ThreadPoolExecutor
import flash1dkmeans


@numba.njit(cache=True)
def _upscale_group(orig_centroids,
                   orig_cluster_borders, weights,
                   weighted_X_prefix_sum, sample_weight_prefix_sum,
                   seed_bit, parent_bit):
    """WARNING: labels, weights and sample_weight should be sorted by weights in ascending order"""
    luts_by_bit = [orig_centroids]

    cluster_borders = orig_cluster_borders

    # Run the upscale
    for i in range(seed_bit, parent_bit):
        centroids, cluster_borders = _increment_group(luts_by_bit[-1], cluster_borders, weights,
                                                      weighted_X_prefix_sum,
                                                      sample_weight_prefix_sum, i)
        luts_by_bit.append(centroids)

    return luts_by_bit, cluster_borders


@numba.njit(cache=True)
def _increment_group(orig_centroids, cluster_borders, weights, weighted_X_prefix_sum,
                     sample_weight_prefix_sum,
                     seed_bit):
    """WARNING: labels, weights and sample_weight should be sorted by weights in ascending order"""
    new_centroids = np.empty(2 ** (seed_bit + 1), dtype=np.float32)
    new_cluster_borders = np.empty(2 ** (seed_bit + 1) + 1, dtype=np.int32)

    assert len(orig_centroids) == 2 ** seed_bit, "The number of centroids should be 2^seed_bit"
    assert len(cluster_borders) == 2 ** seed_bit + 1, \
        "The number of cluster start indices should be 2^seed_bit + 1"

    for c in range(2 ** seed_bit):
        start_idx = cluster_borders[c]
        stop_idx = cluster_borders[c + 1]

        if start_idx == stop_idx:
            # These are empty clusters, but we still need to save the centroids
            new_centroids[c * 2] = orig_centroids[c]
            new_centroids[c * 2 + 1] = orig_centroids[c]
            # new_cluster_borders still needs to be set
            new_cluster_borders[c * 2] = start_idx
            new_cluster_borders[c * 2 + 1] = start_idx
            continue

        cluster_centers, local_cluster_borders = flash1dkmeans.numba_kmeans_1d_two_cluster(
            sorted_X=weights,
            weights_prefix_sum=sample_weight_prefix_sum,
            weighted_X_prefix_sum=weighted_X_prefix_sum,
            start_idx=start_idx,
            stop_idx=stop_idx
        )

        # local_cluster_borders is [start_idx, division_point, stop_idx]

        # save the new centroids and labels
        new_centroids[c * 2] = cluster_centers[0]
        new_centroids[c * 2 + 1] = cluster_centers[1]
        new_cluster_borders[c * 2] = start_idx
        new_cluster_borders[c * 2 + 1] = local_cluster_borders[1]

    new_cluster_borders[-1] = cluster_borders[-1]  # the final border must be set manually

    return new_centroids, new_cluster_borders


@numba.njit(parallel=True, cache=True)
def _seed_and_upscale_layer(layer_gradients, layer_modules, seed_bit, parent_bit, group_count, random_state=None):
    # The shape of LUTs are different for each module and bit.
    # The logical thing to do would be to use a list of lists(for each bit-width) of numpy arrays(for each module).
    # However as numba doesn't like nested python lists, we will use a list of numpy arrays instead,
    # in such a way that we flatten the list of lists into a single list.
    lut_by_bit_by_module = []
    parent_weights_by_modules = []

    n_cluster = 2 ** seed_bit

    for m_idx in range(len(layer_modules)):
        module_gradient = layer_gradients[m_idx]
        module_weight = layer_modules[m_idx]

        row_count = module_weight.shape[0]
        group_size = module_weight.shape[1] // group_count

        assert group_size * group_count == module_weight.shape[1], \
            f"Group count {group_count} does not divide the number of columns {module_weight.shape[1]}"

        parent_weights = np.empty((row_count, group_count, group_size), dtype=np.float32)

        lut_by_bit = []
        for bit in range(seed_bit, parent_bit + 1):
            lut_by_bit.append(np.empty((row_count, group_count, 2 ** bit), dtype=np.float32))

        for r_idx in numba.prange(module_weight.shape[0]):
            for g_idx in range(group_count):
                start_col_idx = g_idx * group_size
                end_col_idx = (g_idx + 1) * group_size

                weights_np = module_weight[r_idx, start_col_idx:end_col_idx]

                weight_mask = weights_np != 0
                sample_weight = module_gradient[r_idx, start_col_idx:end_col_idx]
                sample_weight = sample_weight * weight_mask

                # ---------------- Preprocessing ----------------

                # Use fp64 to avoid precision loss in prefix sum subtraction
                X = weights_np
                sorted_indices = np.argsort(X)

                sorted_X = X[sorted_indices]
                sorted_weights = sample_weight[sorted_indices]

                sorted_X_fp64 = sorted_X.astype(np.float64)
                sorted_weights_fp64 = sorted_weights.astype(np.float64)
                sorted_weights_prefix_sum = np.cumsum(sorted_weights_fp64)

                if sorted_weights_prefix_sum[-1] == 0:
                    # If the sum of the sample weights is zero, we act as if the sample weights are all 1
                    sorted_weights_prefix_sum = np.arange(1, len(sorted_weights_fp64) + 1, dtype=np.float64)
                    sorted_weighted_X_prefix_sum = np.cumsum(sorted_X_fp64)
                    sorted_weighted_X_squared_prefix_sum = np.cumsum(sorted_X_fp64 ** 2)
                else:
                    # Else we proceed with the normal prefix sum calculations
                    sorted_weighted_X_fp64 = sorted_X_fp64 * sorted_weights_fp64
                    sorted_weighted_X_squared_fp64 = sorted_weighted_X_fp64 * sorted_X_fp64

                    sorted_weighted_X_prefix_sum = np.cumsum(sorted_weighted_X_fp64)
                    sorted_weighted_X_squared_prefix_sum = np.cumsum(sorted_weighted_X_squared_fp64)

                # ---------------- Seed ----------------

                # Generate the seed weights

                centroids, cluster_borders = flash1dkmeans.numba_kmeans_1d_k_cluster(
                    sorted_X=sorted_X,
                    n_clusters=n_cluster,
                    max_iter=50,
                    weights_prefix_sum=sorted_weights_prefix_sum,
                    weighted_X_prefix_sum=sorted_weighted_X_prefix_sum,
                    weighted_X_squared_prefix_sum=sorted_weighted_X_squared_prefix_sum,
                    start_idx=0,
                    stop_idx=len(sorted_X),
                    random_state=random_state,
                )

                centroids = centroids.astype(np.float32)

                # ---------------- Upscale ----------------

                # Upscale the seed weights
                lut_per_bit, parent_cluster_borders = _upscale_group(
                    centroids, cluster_borders, sorted_X,
                    sorted_weighted_X_prefix_sum, sorted_weights_prefix_sum,
                    seed_bit, parent_bit)

                # ---------------- Postprocessing ----------------

                # Save the LUTs
                for k, bit in enumerate(range(seed_bit, parent_bit + 1)):
                    lut_by_bit[k][r_idx][g_idx] = lut_per_bit[k]

                # Convert cluster_borders back to labels
                labels = np.empty(group_size, dtype=np.uint8)
                for i in range(2 ** parent_bit):
                    labels[parent_cluster_borders[i]:parent_cluster_borders[i + 1]] = i

                # Unsort the labels
                labels = labels[np.argsort(sorted_indices)]

                # Save the parent weights
                parent_weights[r_idx][g_idx] = labels

        parent_weights_by_modules.append(parent_weights)
        lut_by_bit_by_module.append(lut_by_bit)

    return lut_by_bit_by_module, parent_weights_by_modules


def _get_layer_loader(analyzer, gradients):
    def layer_loader(l):
        # Convert from torch.bf16 to np.fp32 for numba processing
        # Only converts one layer at a time to avoid excessive memory usage
        gradient_layer = [gradients[l][name].float().numpy() for name in analyzer.module_names]
        model_layer = [analyzer.get_layer_weights(l)[name].float().numpy() for name in analyzer.module_names]

        for i in range(len(gradient_layer)):
            weight_mask = model_layer[i] != 0
            gradient_layer[i] = gradient_layer[i] * weight_mask
            if np.sum(gradient_layer[i]) == 0:
                gradient_layer[i] = np.ones_like(gradient_layer[i])

        return gradient_layer, model_layer

    return layer_loader


def _save_results(parent_parameters_path, seed_precision, parent_precision, module_names,
                  luts_by_bit_by_module, parent_weights, l):
    # Note that it is important to cast the luts to fp16 before saving them,
    # as we previously cast them to fp32 to use with numba
    for i, bit in enumerate(range(seed_precision, parent_precision + 1)):
        output_lut_file_name = f"{parent_parameters_path}/lut_{bit}/l{l}.pt"
        os.makedirs(os.path.dirname(output_lut_file_name), exist_ok=True)
        lut_dict = {}
        for j in range(len(module_names)):
            lut_dict[module_names[j]] = luts_by_bit_by_module[j][i].astype(np.float16)
        torch.save(lut_dict, output_lut_file_name)

    parent_weight_dict = {module_names[j]: parent_weights[j].astype(np.uint8)
                          for j in range(len(module_names))}

    output_weights_layer_file_name = f"{parent_parameters_path}/weights/l{l}.pt"
    os.makedirs(os.path.dirname(output_weights_layer_file_name), exist_ok=True)
    torch.save(parent_weight_dict, output_weights_layer_file_name)


def _get_saver(parent_parameters_path, seed_precision, parent_precision, module_names):
    """Returns a function that saves the results for a given layer"""

    def save_results(luts_by_bit_by_module, parent_weights, l):
        return _save_results(parent_parameters_path, seed_precision, parent_precision, module_names,
                             luts_by_bit_by_module, parent_weights, l)

    return save_results


def _load_progress(parent_parameters_path, seed_precision, parent_precision, layer_count):
    # Check if the layer has already been processed
    todo_ran = []
    processed_ran = []
    for l in range(layer_count):
        if all([os.path.exists(f"{parent_parameters_path}/lut_{bit}/l{l}.pt")
                for bit in range(seed_precision, parent_precision + 1)]) and \
                os.path.exists(f"{parent_parameters_path}/weights/l{l}.pt"):
            processed_ran.append(l)
        else:
            todo_ran.append(l)
    return todo_ran, processed_ran


def seed_and_upscale(
        analyzer,
        gradients,
        output_folder,
        seed_precision,
        parent_precision,
        cpu_count=None,
        random_state=None,
        group_count=1,
):
    assert seed_precision <= parent_precision, "Parent precision should be equal or higher than seed precision"

    if cpu_count is None:
        cpu_count = os.cpu_count()
    # Determine IO and threading settings based on the number of cores
    if cpu_count >= 8:
        pipelined_io = True
        io_workers = 2 if cpu_count >= 64 else 1
        numba.set_num_threads(cpu_count - io_workers)
    else:
        pipelined_io = False
        io_workers = 0  # No separate IO workers needed for non-pipelined IO
        numba.set_num_threads(cpu_count)

    logging.info(f"Using {cpu_count} cores for parallelization")

    logging.info(f"Seeding & Upscaling from {seed_precision}-bit to {parent_precision}-bit")

    layers_to_process, completed_layers = _load_progress(output_folder, seed_precision, parent_precision,
                                                        analyzer.num_layers)

    if completed_layers:
        logging.info(f"The following layers will be skipped as they have already been processed:\n{completed_layers}")
        logging.info(f"To reprocess these layers, delete the corresponding files in {output_folder}")

    if not layers_to_process:
        logging.info("All layers have already been processed. Exiting...")
        return

    logging.info(f"Quantizing layers {layers_to_process}")

    layer_loader = _get_layer_loader(analyzer, gradients)
    layer_saver = _get_saver(output_folder, seed_precision, parent_precision, analyzer.module_names)

    if pipelined_io:
        with ThreadPoolExecutor(max_workers=io_workers) as io_executor:
            for l in tqdm(layers_to_process, desc="Quantizing layers..."):
                if l == layers_to_process[0]:
                    future_load = io_executor.submit(layer_loader, l)

                gradient_layer, model_layer = future_load.result()

                if l != layers_to_process[-1]:
                    future_load = io_executor.submit(layer_loader, l + 1)

                luts_by_bit_by_module, parent_weights = _seed_and_upscale_layer(
                    gradient_layer,
                    model_layer,
                    seed_precision,
                    parent_precision,
                    group_count,
                    random_state=random_state,
                )

                io_executor.submit(layer_saver, luts_by_bit_by_module, parent_weights, l)
            logging.info("Waiting for IO to finish...")
    else:
        for l in tqdm(layers_to_process, desc="Quantizing layers..."):
            gradient_layer, model_layer = layer_loader(l)

            luts_by_bit_by_module, parent_weights = _seed_and_upscale_layer(
                gradient_layer,
                model_layer,
                seed_precision,
                parent_precision,
                group_count,
                random_state=random_state
            )

            layer_saver(luts_by_bit_by_module, parent_weights, l)
