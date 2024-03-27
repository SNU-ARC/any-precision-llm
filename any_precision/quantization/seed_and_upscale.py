import os
import logging
import torch
import numpy as np
from tqdm import tqdm
import numba
from concurrent.futures import ThreadPoolExecutor
from .utils import query_prefix_sum
from .upscale import _upscale_group


@numba.njit(cache=True)
def rand_choice_prefix_sum(arr, prob_prefix_sum):
    total_prob = prob_prefix_sum[-1]
    selector = np.random.random_sample() * total_prob
    return arr[np.searchsorted(prob_prefix_sum, selector)]


@numba.njit(cache=True)
def calculate_inertia(X, centroids, weights_prefix_sum, weighted_X_prefix_sum, weighted_X_squared_prefix_sum,
                      count=None):
    """ Time complexity: O(n_clusters * log(n_samples))"""
    if count is None:
        count = len(X)
    # Assume sorted X
    centroids = np.sort(centroids)
    midpoints = (centroids[:-1] + centroids[1:]) / 2

    centroid_ranges = np.empty(len(centroids) + 1, dtype=np.int32)
    centroid_ranges[0] = 0
    centroid_ranges[-1] = len(X)
    centroid_ranges[1:-1] = np.searchsorted(X, midpoints)

    # inertia = sigma_i(w_i * abs(x_i - c)^2) = sigma_i(w_i * (x_i^2 - 2 * x_i * c + c^2))
    #         = sigma_i(w_i * x_i^2) - 2 * c * sigma_i(w_i * x_i) + c^2 * sigma_i(w_i)
    #         = sigma_i(weighted_X_squared) - 2 * c * sigma_i(weighted_X) + c^2 * sigma_i(weight)
    #  Note that the centroid c is the CLOSEST centroid to x_i, so the above calculation must be done for each cluster

    inertia = 0
    for i in range(len(centroids)):
        start = centroid_ranges[i]
        end = centroid_ranges[i + 1]

        if start >= count:
            break
        if end >= count:
            end = count

        if start == end:
            continue

        cluster_weighted_X_squared_sum = query_prefix_sum(weighted_X_squared_prefix_sum, start, end)
        cluster_weighted_X_sum = query_prefix_sum(weighted_X_prefix_sum, start, end)
        cluster_weight_sum = query_prefix_sum(weights_prefix_sum, start, end)

        inertia += cluster_weighted_X_squared_sum - 2 * centroids[i] * cluster_weighted_X_sum + \
                   centroids[i] ** 2 * cluster_weight_sum

    return inertia


@numba.njit(cache=True)
def rand_choice_centroids(X, centroids, weights_prefix_sum, weighted_X_prefix_sum, weighted_X_squared_prefix_sum, size):
    total_inertia = calculate_inertia(X, centroids, weights_prefix_sum, weighted_X_prefix_sum,
                                      weighted_X_squared_prefix_sum)
    selectors = np.random.random_sample(size) * total_inertia
    results = np.empty(size, dtype=centroids.dtype)

    for i in range(size):
        selector = selectors[i]
        left = 1
        right = len(X)
        while left < right:
            mid = (left + right) // 2
            inertia = calculate_inertia(X, centroids, weights_prefix_sum, weighted_X_prefix_sum,
                                        weighted_X_squared_prefix_sum,
                                        count=mid)
            if inertia < selector:
                left = mid + 1
            else:
                right = mid
        results[i] = X[left - 1]

    return results


@numba.njit(cache=True)
def kmeans_plusplus(X, n_clusters, weights_prefix_sum, weighted_X_prefix_sum, weighted_X_squared_prefix_sum):
    n_local_trials = 2 + int(np.log(n_clusters))

    centroids = np.empty(n_clusters, dtype=np.float64)

    # First centroid is chosen randomly according to sample_weight
    centroids[0] = rand_choice_prefix_sum(X, weights_prefix_sum)

    for c_id in range(1, n_clusters):
        # Choose the next centroid randomly according to the weighted distances
        # Sample n_local_trials candidates and choose the best one
        centroid_candidates = rand_choice_centroids(X, centroids[:c_id], weights_prefix_sum, weighted_X_prefix_sum,
                                                    weighted_X_squared_prefix_sum, n_local_trials)

        best_inertia = np.inf
        best_centroid = None
        for i in range(len(centroid_candidates)):
            centroids[c_id] = centroid_candidates[i]
            inertia = calculate_inertia(X, centroids[:c_id + 1], weights_prefix_sum, weighted_X_prefix_sum,
                                        weighted_X_squared_prefix_sum)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroid = centroid_candidates[i]
        centroids[c_id] = best_centroid

    return centroids


@numba.njit(cache=True)
def my_kmeans(X, weights_prefix_sum, weighted_X_prefix_sum,
              weighted_X_squared_prefix_sum,
              n_clusters, max_iter=50):
    """WARNING: All inputs must be sorted in ascending order of X"""
    cluster_borders = np.empty(n_clusters + 1, dtype=np.int32)
    cluster_borders[0] = 0
    cluster_borders[-1] = len(X)

    new_cluster_borders = np.empty(n_clusters + 1, dtype=np.int32)
    new_cluster_borders[0] = 0
    new_cluster_borders[-1] = len(X)

    centroids = kmeans_plusplus(X, n_clusters,
                                weights_prefix_sum, weighted_X_prefix_sum,
                                weighted_X_squared_prefix_sum)
    centroids.sort()

    for _ in range(max_iter):
        cluster_midpoints = (centroids[:-1] + centroids[1:]) / 2
        for i in range(n_clusters - 1):
            new_cluster_borders[i + 1] = np.searchsorted(X, cluster_midpoints[i])

        if np.array_equal(cluster_borders, new_cluster_borders):
            break

        cluster_borders[:] = new_cluster_borders
        for i in range(n_clusters):
            cluster_start = cluster_borders[i]
            cluster_end = cluster_borders[i + 1]

            if cluster_end < cluster_start:
                raise ValueError("Cluster end is less than cluster start")

            if cluster_start == cluster_end:
                continue

            cluster_weighted_X_sum = query_prefix_sum(weighted_X_prefix_sum, cluster_start, cluster_end)
            cluster_weight_sum = query_prefix_sum(weights_prefix_sum, cluster_start, cluster_end)

            if cluster_weight_sum == 0:
                # if the sum of the weights is zero, we set the centroid to the mean of the cluster
                centroids[i] = X[cluster_start:cluster_end].mean()
            else:
                centroids[i] = cluster_weighted_X_sum / cluster_weight_sum

    return centroids, cluster_borders


@numba.njit(parallel=True, cache=True)
def seed_and_upscale_layer(layer_gradients, layer_modules, seed_bit, parent_bit, group_count, random_state=None):
    # WARNING: random_state does NOT guarantee reproducibility on different machines

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
            set_np_seed_njit(random_state)  # this needs to be set per thread
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
                centroids, cluster_borders = my_kmeans(
                    sorted_X, sorted_weights_prefix_sum,
                    sorted_weighted_X_prefix_sum,
                    sorted_weighted_X_squared_prefix_sum, n_cluster
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


@numba.njit(cache=True)
def set_np_seed_njit(random_state):
    """Set the seed for numpy random number generator.
    Must be used in a numba.jit function."""
    if random_state is not None:
        np.random.seed(random_state)


def get_layer_loader(model_weights, gradients, module_names):
    def layer_loader(l):
        # Convert from torch.bf16 to np.fp32 for numba processing
        # Only converts one layer at a time to avoid excessive memory usage
        gradient_layer = [gradients[l][name].float().numpy() for name in module_names]
        model_layer = [model_weights[l][name].float().numpy() for name in module_names]
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


def get_saver(parent_parameters_path, seed_precision, parent_precision, module_names):
    """Returns a function that saves the results for a given layer"""

    def save_results(luts_by_bit_by_module, parent_weights, l):
        return _save_results(parent_parameters_path, seed_precision, parent_precision, module_names,
                             luts_by_bit_by_module, parent_weights, l)

    return save_results


def load_progress(parent_parameters_path, seed_precision, parent_precision, layer_count):
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

    layers_to_process, completed_layers = load_progress(output_folder, seed_precision, parent_precision,
                                                        analyzer.num_layers)

    if completed_layers:
        logging.info(f"The following layers will be skipped as they have already been processed:\n{completed_layers}")
        logging.info(f"To reprocess these layers, delete the corresponding files in {output_folder}")

    if not layers_to_process:
        logging.info("All layers have already been processed. Exiting...")
        return

    model_weights = analyzer.get_model_weights()

    logging.info(f"Quantizing layers {layers_to_process}")

    layer_loader = get_layer_loader(model_weights, gradients, analyzer.module_names)
    layer_saver = get_saver(output_folder, seed_precision, parent_precision, analyzer.module_names)

    if pipelined_io:
        with ThreadPoolExecutor(max_workers=io_workers) as io_executor:
            for l in tqdm(layers_to_process, desc="Quantizing layers..."):
                if l == layers_to_process[0]:
                    future_load = io_executor.submit(layer_loader, l)

                gradient_layer, model_layer = future_load.result()

                if l != layers_to_process[-1]:
                    future_load = io_executor.submit(layer_loader, l + 1)

                luts_by_bit_by_module, parent_weights = seed_and_upscale_layer(
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

            luts_by_bit_by_module, parent_weights = seed_and_upscale_layer(
                gradient_layer,
                model_layer,
                seed_precision,
                parent_precision,
                group_count,
                random_state=random_state
            )

            layer_saver(luts_by_bit_by_module, parent_weights, l)