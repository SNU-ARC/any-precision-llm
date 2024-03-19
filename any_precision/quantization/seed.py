import os
import logging
import torch
import numpy as np
from tqdm import tqdm
import numba


@numba.njit(cache=True)
def build_segment_tree(arr):
    n = len(arr)
    height = int(np.ceil(np.log2(n)))
    offset = 2 ** height
    segment_tree = np.zeros(2 ** (height + 1), dtype=np.float32)
    segment_tree[offset:offset + n] = arr
    for i in range(offset - 1, 0, -1):
        segment_tree[i] = segment_tree[2 * i] + segment_tree[2 * i + 1]
    return segment_tree


@numba.njit(cache=True)
def query_segment_tree(segment_tree, start, end):
    offset = len(segment_tree) // 2
    start += offset
    end += offset
    result = 0
    while start <= end:
        if start % 2 == 1:
            result += segment_tree[start]
            start += 1
        if end % 2 == 0:
            result += segment_tree[end]
            end -= 1
        start = start // 2
        end = end // 2
    return result


@numba.njit(cache=True)
def rand_choice_nb(arr, prob, size):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :param size: The number of samples to draw.
    :return: A random sample from the given array with a given probability.
    """
    cumsum = np.cumsum(prob)
    selectors = cumsum[-1] * np.random.random_sample(size)

    return arr[np.searchsorted(cumsum, selectors, side="right")]


@numba.njit(cache=True)
def kmeans_plusplus(X, sample_weight, n_clusters):
    n_local_trials = 2 + int(np.log(n_clusters))

    centroids = np.empty(n_clusters, dtype=np.float32)

    # First centroid is chosen randomly according to sample_weight
    centroids[0] = rand_choice_nb(X, sample_weight, 1)[0]

    min_distances = np.abs(X - centroids[0])

    for c_id in range(1, n_clusters):
        # Square the distances to increase the probability of choosing distant points
        distances_squared = min_distances ** 2

        # Apply sample_weight to the squared distances
        weighted_distances = distances_squared * sample_weight

        # Choose the next centroid randomly according to the weighted distances
        # Sample n_local_trials candidates and choose the best one
        centroid_candidates = rand_choice_nb(X, weighted_distances, n_local_trials)
        best_error = np.inf
        best_candidate = None
        best_new_min_distances = None
        for i, centroid in enumerate(centroid_candidates):
            new_distances = np.abs(X - centroid)
            new_min_distances = np.minimum(min_distances, new_distances)
            error = np.sum(sample_weight * new_min_distances ** 2)
            if error < best_error:
                best_error = error
                best_candidate = centroid
                best_new_min_distances = new_min_distances

        centroids[c_id] = best_candidate
        min_distances = best_new_min_distances

    return centroids


@numba.njit(cache=True)
def my_kmeans(X, sample_weight, n_clusters, max_iter=50):
    sorted_indices = np.argsort(X)
    sorted_X = X[sorted_indices]

    # A segment tree is used as opposed to a prefix sum array to avoid numerical issues in floating point subtraction
    weights_segtree = build_segment_tree(sample_weight[sorted_indices])
    weighted_X_segtree = build_segment_tree((X * sample_weight)[sorted_indices])

    cluster_borders = np.empty(n_clusters + 1, dtype=np.int32)
    cluster_borders[0] = 0
    cluster_borders[-1] = len(X)

    new_cluster_borders = np.empty(n_clusters + 1, dtype=np.int32)
    new_cluster_borders[0] = 0
    new_cluster_borders[-1] = len(X)

    centroids = kmeans_plusplus(sorted_X, sample_weight, n_clusters)
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

            cluster_weighted_X_sum = query_segment_tree(weighted_X_segtree, cluster_start, cluster_end - 1)
            cluster_weight_sum = query_segment_tree(weights_segtree, cluster_start, cluster_end - 1)

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
def seed_layer(layer_gradients, layer_modules, bit_width):
    n_cluster = 2 ** bit_width
    layer_lut_by_module = []
    layer_weight_by_module = []

    for module_gradient, module_weight in zip(layer_gradients, layer_modules):
        module_lut = np.empty((module_weight.shape[0], 1, n_cluster), dtype=np.float32)
        q_module_weight = np.empty((module_weight.shape[0], 1, module_weight.shape[1]), dtype=np.uint8)
        for i in numba.prange(module_weight.shape[0]):
            weights_np = module_weight[i, :]

            weight_mask = weights_np != 0
            sample_weight = module_gradient[i, :]
            sample_weight = sample_weight * weight_mask

            if np.sum(sample_weight) == 0:
                sample_weight = np.ones_like(sample_weight)

            centroids, labels = my_kmeans(weights_np, sample_weight, n_cluster)
            module_lut[i, 0, :] = centroids
            q_module_weight[i, 0, :] = labels

        layer_lut_by_module.append(module_lut)
        layer_weight_by_module.append(q_module_weight)

    return layer_lut_by_module, layer_weight_by_module


def get_seed(analyzer, gradients, bit_width, output_folder, cpu_count=None):
    if cpu_count is None:
        cpu_count = os.cpu_count()

    numba.set_num_threads(cpu_count)

    logging.info(f"Using {cpu_count} cores for parallelization")

    lut_folder = f"{output_folder}/lut"
    if not os.path.exists(lut_folder):
        os.makedirs(lut_folder)

    weight_folder = f"{output_folder}/weights"
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

    model_weights = analyzer.get_model_weights()

    if isinstance(gradients, str):
        gradients = torch.load(gradients)

    logging.info(f"Quantizing layers {list(range(len(model_weights)))}")

    skipped_layers = []

    for l in tqdm(range(len(model_weights)), desc="Quantizing layers..."):
        if os.path.exists(f"{lut_folder}/l{l}.pt") and os.path.exists(f"{weight_folder}/l{l}.pt"):
            skipped_layers.append(l)
            continue
        if skipped_layers:
            logging.info(f"The following layers have been skipped: {skipped_layers}")
            logging.info(f"To reprocess these layers, "
                         f"delete the corresponding files in {lut_folder} and {weight_folder}")
            skipped_layers = []

        gradient_layer = [gradients[l][name].float().numpy() for name in analyzer.module_names]
        model_layer = [model_weights[l][name].float().numpy() for name in analyzer.module_names]

        layer_lut_by_module, layer_weight_by_module = seed_layer(
            gradient_layer,
            model_layer,
            bit_width,
        )

        lut_per_layer, weight_per_layer = {}, {}

        for i, name in enumerate(analyzer.module_names):
            lut_per_layer[name] = layer_lut_by_module[i]
            weight_per_layer[name] = layer_weight_by_module[i]

        # save parts
        torch.save(lut_per_layer, f"{lut_folder}/l{l}.pt")
        torch.save(weight_per_layer, f"{weight_folder}/l{l}.pt")

    if skipped_layers:
        logging.info(f"The following layers have been skipped: {skipped_layers}")
        logging.info(f"To reprocess these layers, "
                     f"delete the corresponding files in {lut_folder} and {weight_folder}")
