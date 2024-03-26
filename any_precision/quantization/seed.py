import os
import logging
import torch
import numpy as np
from tqdm import tqdm
import numba
from concurrent.futures import ThreadPoolExecutor


@numba.njit(cache=True)
def query_prefix_sum(arr, start, stop):
    return arr[stop - 1] - arr[start - 1] if start > 0 else arr[stop - 1]


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
def my_kmeans(X, sample_weight, n_clusters, max_iter=50):
    sorted_indices = np.argsort(X)

    # Use fp64 to avoid precision loss in prefix sum subtraction
    sorted_X = X[sorted_indices].astype(np.float64)
    sorted_weights = sample_weight[sorted_indices].astype(np.float64)
    sorted_weighted_X = sorted_X * sorted_weights
    sorted_weighted_X_squared = sorted_weighted_X * sorted_X

    sorted_weights_prefix_sum = np.cumsum(sorted_weights)
    sorted_weighted_X_prefix_sum = np.cumsum(sorted_weighted_X)
    sorted_weighted_X_squared_prefix_sum = np.cumsum(sorted_weighted_X_squared)

    cluster_borders = np.empty(n_clusters + 1, dtype=np.int32)
    cluster_borders[0] = 0
    cluster_borders[-1] = len(X)

    new_cluster_borders = np.empty(n_clusters + 1, dtype=np.int32)
    new_cluster_borders[0] = 0
    new_cluster_borders[-1] = len(X)

    centroids = kmeans_plusplus(sorted_X, n_clusters,
                                sorted_weights_prefix_sum, sorted_weighted_X_prefix_sum,
                                sorted_weighted_X_squared_prefix_sum)
    centroids.sort()

    for _ in range(max_iter):
        cluster_midpoints = (centroids[:-1] + centroids[1:]) / 2
        for i in range(n_clusters - 1):
            new_cluster_borders[i + 1] = np.searchsorted(sorted_X, cluster_midpoints[i])

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

            cluster_weighted_X_sum = query_prefix_sum(sorted_weighted_X_prefix_sum, cluster_start, cluster_end)
            cluster_weight_sum = query_prefix_sum(sorted_weights_prefix_sum, cluster_start, cluster_end)

            if cluster_weight_sum == 0:
                # if the sum of the weights is zero, we set the centroid to the mean of the cluster
                centroids[i] = sorted_X[cluster_start:cluster_end].mean()
            else:
                centroids[i] = cluster_weighted_X_sum / cluster_weight_sum

    labels = np.empty(len(X), dtype=np.uint8)
    for i in range(n_clusters):
        cluster_start = cluster_borders[i]
        cluster_end = cluster_borders[i + 1]
        labels[sorted_indices[cluster_start:cluster_end]] = i

    return centroids, labels


@numba.njit(parallel=True, cache=True)
def seed_layer(layer_gradients, layer_modules, bit_width, group_size, random_state=None):
    # WARNING: random_state does NOT guarantee reproducibility on different machines
    n_cluster = 2 ** bit_width
    layer_lut_by_module = []
    layer_weight_by_module = []

    for module_gradient, module_weight in zip(layer_gradients, layer_modules):
        current_group_size = group_size if group_size > 0 else module_weight.shape[1]
        assert module_weight.shape[1] % current_group_size == 0, \
            f"Group size {current_group_size} does not divide {module_weight.shape[1]}"
        group_count = module_weight.shape[1] // current_group_size
        module_lut = np.empty((module_weight.shape[0], group_count, n_cluster), dtype=np.float32)
        q_module_weight = np.empty((module_weight.shape[0], group_count, current_group_size), dtype=np.uint8)
        for i in numba.prange(module_weight.shape[0]):
            set_np_seed_njit(random_state)  # this needs to be set per thread
            for j in range(group_count):
                start_col_idx = j * current_group_size
                end_col_idx = (j + 1) * current_group_size

                weights_np = module_weight[i, start_col_idx:end_col_idx]

                weight_mask = weights_np != 0
                sample_weight = module_gradient[i, start_col_idx:end_col_idx]
                sample_weight = sample_weight * weight_mask

                if np.sum(sample_weight) == 0:
                    sample_weight = np.ones_like(sample_weight)

                centroids, labels = my_kmeans(weights_np, sample_weight, n_cluster)
                module_lut[i, j, :] = centroids
                q_module_weight[i, j, :] = labels

        layer_lut_by_module.append(module_lut)
        layer_weight_by_module.append(q_module_weight)

    return layer_lut_by_module, layer_weight_by_module


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


def get_saver(lut_folder, weight_folder, module_names):
    def layer_saver(l, layer_lut_by_module, layer_weight_by_module):
        lut_per_layer, weight_per_layer = {}, {}

        for i, name in enumerate(module_names):
            lut_per_layer[name] = layer_lut_by_module[i].astype(np.float16)
            weight_per_layer[name] = layer_weight_by_module[i]

        # save parts
        torch.save(lut_per_layer, f"{lut_folder}/l{l}.pt")
        torch.save(weight_per_layer, f"{weight_folder}/l{l}.pt")

    return layer_saver


def load_progress(layer_count, lut_folder, weight_folder):
    processed_layers = []
    to_process = []
    for l in range(layer_count):
        if os.path.exists(f"{lut_folder}/l{l}.pt") and os.path.exists(f"{weight_folder}/l{l}.pt"):
            processed_layers.append(l)
        else:
            to_process.append(l)

    return processed_layers, to_process


def get_seed(
        analyzer,
        gradients,
        bit_width,
        output_folder,
        cpu_count=None,
        random_state=None,
        group_size=-1,
):
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

    numba.set_num_threads(cpu_count)

    logging.info(f"Using {cpu_count} cores for parallelization")

    lut_folder = f"{output_folder}/lut"
    if not os.path.exists(lut_folder):
        os.makedirs(lut_folder)

    weight_folder = f"{output_folder}/weights"
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

    layers_to_skip, layers_to_process = load_progress(len(analyzer.get_model_weights()), lut_folder, weight_folder)

    if layers_to_skip:
        logging.info(f"The following layers have been skipped as they have already been processed:\n{layers_to_skip}")
        logging.info(f"To reprocess these layers, "
                     f"delete the corresponding files in {lut_folder} and {weight_folder}")

    if not layers_to_process:
        logging.info("All layers have already been processed. Exiting...")
        return

    model_weights = analyzer.get_model_weights()

    logging.info(f"Quantizing layers {layers_to_process}")

    layer_loader = get_layer_loader(model_weights, gradients, analyzer.module_names)
    layer_saver = get_saver(lut_folder, weight_folder, analyzer.module_names)

    if pipelined_io:
        with ThreadPoolExecutor(max_workers=io_workers) as io_executor:
            for l in tqdm(layers_to_process, desc="Quantizing layers..."):
                if l == layers_to_process[0]:
                    future_load = io_executor.submit(layer_loader, l)

                gradient_layer, model_layer = future_load.result()

                if l != layers_to_process[-1]:
                    future_load = io_executor.submit(layer_loader, l + 1)

                layer_lut_by_module, layer_weight_by_module = seed_layer(
                    gradient_layer,
                    model_layer,
                    bit_width,
                    group_size,
                    random_state=random_state,
                )

                io_executor.submit(layer_saver, l, layer_lut_by_module, layer_weight_by_module)
            logging.info("Waiting for IO to finish...")
    else:
        for l in tqdm(layers_to_process, desc="Quantizing layers..."):
            gradient_layer, model_layer = layer_loader(l)

            layer_lut_by_module, layer_weight_by_module = seed_layer(
                gradient_layer,
                model_layer,
                bit_width,
                group_size,
                random_state=random_state
            )

            layer_saver(l, layer_lut_by_module, layer_weight_by_module)
