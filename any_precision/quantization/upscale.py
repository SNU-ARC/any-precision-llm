import os
import torch
from concurrent.futures import ThreadPoolExecutor
import argparse
import numpy as np
import numba
from tqdm import tqdm
import logging
from .utils import query_prefix_sum

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model', type=str,
    help='model weights to load', required=True
)

parser.add_argument(
    '--gradient', type=str,
    help='model gradients to load', required=True
)
parser.add_argument(
    '--orig_bit', type=int, default=3,
    help='bitwidth', choices=[2, 3, 4, 5, 6, 7, 8],
)
parser.add_argument(
    '--bit', type=int, default=4,
    help='bitwidth', choices=[2, 3, 4, 5, 6, 7, 8],
)

parser.add_argument(
    '--lut_src', type=str, required=True,
    help='LUT path'
)

parser.add_argument(
    '--lut_dest', type=str, required=True,
    help='LUT path'
)

parser.add_argument(
    '--cores', type=int, default=os.cpu_count(),
    help='number of cores to use for parallelization'
)


# TODO: Merge seed & upscale, reuse sort & prefix sum

@numba.njit(cache=True)
def _upscale_group(orig_centroids, orig_labels, weights, sample_weight, seed_bit, parent_bit):
    luts_by_bit = [orig_centroids]
    labels = orig_labels

    # sort by the weights
    sorted_indices = np.argsort(weights)
    weights = weights[sorted_indices]
    sample_weight = sample_weight[sorted_indices]
    labels = labels[sorted_indices]

    weights_fp64 = weights.astype(np.float64)
    sample_weight_fp64 = sample_weight.astype(np.float64)

    weighted_X_prefix_sum = np.cumsum(weights_fp64 * sample_weight_fp64)
    sample_weight_prefix_sum = np.cumsum(sample_weight_fp64)

    cluster_borders = np.searchsorted(labels, np.arange(2 ** seed_bit + 1, dtype=np.uint8))

    # Run the upscale
    for i in range(seed_bit, parent_bit):
        centroids, cluster_borders = _increment_group(luts_by_bit[-1], cluster_borders, weights,
                                                      weighted_X_prefix_sum,
                                                      sample_weight_prefix_sum, i)
        luts_by_bit.append(centroids)

    # Convert cluster_borders back to labels
    new_labels = np.empty_like(labels)
    for i in range(2 ** parent_bit):
        new_labels[cluster_borders[i]:cluster_borders[i + 1]] = i

    # Unsort the labels
    new_labels = new_labels[np.argsort(sorted_indices)]

    return luts_by_bit, new_labels


@numba.njit(cache=True)
def _increment_group(orig_centroids, cluster_borders, weights, weighted_X_prefix_sum,
                     sample_weight_prefix_sum,
                     seed_bit):
    """WARNING: labels, weights and sample_weight should be sorted by weights in ascending order"""
    new_centroids = np.empty(2 ** (seed_bit + 1), dtype=np.float32)
    new_cluster_borders = np.empty(2 ** (seed_bit + 1) + 1, dtype=np.int64)

    assert len(orig_centroids) == 2 ** seed_bit, "The number of centroids should be 2^seed_bit"
    assert len(cluster_borders) == 2 ** seed_bit + 1, \
        "The number of cluster start indices should be 2^seed_bit + 1"

    for c in range(2 ** seed_bit):
        start_idx = cluster_borders[c]
        end_idx = cluster_borders[c + 1]

        if start_idx == end_idx:
            # These are empty clusters, but we still need to save the centroids
            new_centroids[c * 2] = orig_centroids[c]
            new_centroids[c * 2 + 1] = orig_centroids[c]
            # new_cluster_borders still needs to be set
            new_cluster_borders[c * 2] = start_idx
            new_cluster_borders[c * 2 + 1] = start_idx
            continue

        cluster_centers, division_point = _faster_1d_two_cluster_kmeans(
            weights,
            weighted_X_prefix_sum,
            sample_weight_prefix_sum,
            start_idx, end_idx
        )

        # save the new centroids and labels
        new_centroids[c * 2] = cluster_centers[0]
        new_centroids[c * 2 + 1] = cluster_centers[1]
        new_cluster_borders[c * 2] = start_idx
        new_cluster_borders[c * 2 + 1] = division_point

    new_cluster_borders[-1] = cluster_borders[-1]  # the final border must be set manually

    return new_centroids, new_cluster_borders


@numba.njit(cache=True)
def _faster_1d_two_cluster_kmeans(X, weighted_X_prefix_sum, sample_weight_prefix_sum, start_idx,
                                  end_idx):
    """An optimized kmeans for 1D data with 2 clusters.
    Only  operates on the range [start_idx, end_idx) of the input arrays.
    
    This function uses np.float32 instead of np.float16 for the centroids so that numba can compile it.
    Please cast the result back to np.float16 before saving it.

    WARNING: X should be sorted in ascending order.
    """
    size = end_idx - start_idx
    if size < 0:
        raise ValueError("The end index should be greater than or equal to the start index")

    if size == 0:
        raise ValueError("This function should not be called with an empty range")

    centroids = np.empty(2, dtype=np.float32)

    if size == 1:
        centroids[0], centroids[1] = X[start_idx], X[start_idx]
        return centroids, start_idx + 1

    if size == 2:
        centroids[0], centroids[1] = X[start_idx], X[start_idx + 1]
        return centroids, start_idx + 1

    # Now we know that there are at least 3 elements

    # If the sum of the sample weight in the range is 0, we call an unweighted version of the function
    if query_prefix_sum(sample_weight_prefix_sum, start_idx, end_idx) == 0:
        return _faster_1d_two_cluster_kmeans_unweighted(X, start_idx, end_idx)

    # Check if there is only one nonzero sample weight
    total_weight = query_prefix_sum(sample_weight_prefix_sum, start_idx, end_idx)
    sample_weight_prefix_sum_within_range = sample_weight_prefix_sum[start_idx:end_idx]
    final_increase_idx = np.searchsorted(sample_weight_prefix_sum_within_range,
                                         sample_weight_prefix_sum_within_range[-1])
    final_increase_amount = (sample_weight_prefix_sum_within_range[final_increase_idx] -
                             sample_weight_prefix_sum_within_range[final_increase_idx - 1])
    if total_weight == final_increase_amount:
        # If there is only one nonzero sample weight, we need to return the corresponding weight as the centroid
        # and set all elements to the left cluster
        nonzero_weight_index = start_idx + final_increase_idx
        centroids[0], centroids[1] = X[nonzero_weight_index], X[nonzero_weight_index]
        return centroids, end_idx

    # Now we know that there are at least 3 elements and at least 2 nonzero weights

    # KMeans with 2 clusters on 1D data is equivalent to finding a division point.
    # The division point can be found by doing a binary search on the prefix sum.

    # We will do a search for the division point,
    # where we search for the optimum number of elements in the first cluster
    # We don't want empty clusters, so we set the floor and ceiling to 1 and len(X) - 1
    floor = start_idx
    ceiling = end_idx
    left_centroid = None
    right_centroid = None
    division_point = None

    while floor + 1 < ceiling:
        division_point = (floor + ceiling) // 2
        # If the left cluster has no weight, we need to move the floor up
        left_weight_sum = query_prefix_sum(sample_weight_prefix_sum, start_idx, division_point)
        if left_weight_sum == 0:
            floor = division_point
            continue
        right_weight_sum = query_prefix_sum(sample_weight_prefix_sum, division_point, end_idx)
        # If the right cluster has no weight, we need to move the ceiling down
        if right_weight_sum == 0:
            ceiling = division_point
            continue

        left_centroid = query_prefix_sum(weighted_X_prefix_sum, start_idx, division_point) / left_weight_sum
        right_centroid = query_prefix_sum(weighted_X_prefix_sum, division_point, end_idx) / right_weight_sum

        new_division_point_value = (left_centroid + right_centroid) / 2
        if X[division_point - 1] <= new_division_point_value:
            if new_division_point_value <= X[division_point]:
                # The new division point matches the previous one, so we can stop
                break
            else:
                floor = division_point
        else:
            ceiling = division_point

    # initialize variables in case the loop above does not run through
    if left_centroid is None:
        division_point = (floor + ceiling) // 2
        left_centroid = query_prefix_sum(weighted_X_prefix_sum, start_idx, division_point) / \
                        query_prefix_sum(sample_weight_prefix_sum, start_idx, division_point)
    if right_centroid is None:
        division_point = (floor + ceiling) // 2
        right_centroid = query_prefix_sum(weighted_X_prefix_sum, division_point, end_idx) / \
                         query_prefix_sum(sample_weight_prefix_sum, division_point, end_idx)

    # avoid using lists to allow numba.njit
    centroids[0] = left_centroid
    centroids[1] = right_centroid

    return centroids, division_point


@numba.njit(cache=True)
def _faster_1d_two_cluster_kmeans_unweighted(X, start_idx, end_idx):
    """Unweighted version of _faster_1d_two_cluster_kmeans.

    WARNING: X should have more than 3 elements and should be sorted in ascending order.
    """
    centroids = np.empty(2, dtype=np.float32)

    floor = start_idx
    ceiling = end_idx
    left_centroid = None
    right_centroid = None
    division_point = None

    X_prefix_sum = np.cumsum(X.astype(np.float64))

    while floor + 1 < ceiling:
        division_point = (floor + ceiling) // 2
        # If the left cluster has no weight, we need to move the floor up
        left_cluster_size = division_point - start_idx
        if left_cluster_size == 0:
            floor = division_point
            continue
        right_cluster_size = end_idx - division_point
        # If the right cluster has no weight, we need to move the ceiling down
        if right_cluster_size == 0:
            ceiling = division_point
            continue

        left_centroid = query_prefix_sum(X_prefix_sum, start_idx, division_point) / left_cluster_size
        right_centroid = query_prefix_sum(X_prefix_sum, division_point, end_idx) / right_cluster_size

        new_division_point_value = (left_centroid + right_centroid) / 2
        if X[division_point - 1] <= new_division_point_value:
            if new_division_point_value <= X[division_point]:
                # The new division point matches the previous one, so we can stop
                break
            else:
                floor = division_point
        else:
            ceiling = division_point

    # initialize variables in case the loop above does not run through
    if left_centroid is None:
        division_point = (floor + ceiling) // 2
        left_centroid = query_prefix_sum(X_prefix_sum, start_idx, division_point) / (division_point - start_idx)
    if right_centroid is None:
        division_point = (floor + ceiling) // 2
        right_centroid = query_prefix_sum(X_prefix_sum, division_point, end_idx) / (end_idx - division_point)

    # avoid using lists to allow numba.njit
    centroids[0] = left_centroid
    centroids[1] = right_centroid

    return centroids, division_point


@numba.njit(parallel=True, cache=True)
def _upscale_layer_njit(seed_lut, seed_weights, model_layer, gradient_layer, seed_bit, parent_bit):
    # The shape of LUTs are different for each module and bit.
    # The logical thing to do would be to use a list of lists(for each bit-width) of numpy arrays(for each module).
    # However as numba doesn't like nested python lists, we will use a list of numpy arrays instead,
    # in such a way that we flatten the list of lists into a single list.
    lut_by_modules_by_bit = []
    for bit in range(seed_bit, parent_bit + 1):
        for m_idx in range(len(seed_lut)):
            row_count, group_count, original_bin_count = seed_lut[m_idx].shape
            lut_by_modules_by_bit.append(np.empty((row_count, group_count, 2 ** bit), dtype=np.float32))

    # This is a list of numpy arrays, where each array is the parent weights for a module
    parent_weights_by_modules = [np.empty_like(seed_weights[m_idx]) for m_idx in range(len(seed_weights))]

    for m_idx in range(len(seed_lut)):
        g = gradient_layer[m_idx]

        module_weight = model_layer[m_idx]

        for r_idx in numba.prange(len(seed_weights[m_idx])):  # rows
            weights_np = module_weight[r_idx]
            sample_weight = g[r_idx]

            # reshape weights into groups
            weights_np = weights_np.reshape(seed_weights[m_idx][r_idx].shape)
            sample_weight = sample_weight.reshape(seed_weights[m_idx][r_idx].shape)

            for g_idx in range(len(seed_weights[m_idx][r_idx])):  # groups
                orig_labels = seed_weights[m_idx][r_idx][g_idx]
                weights_np_group = weights_np[g_idx]
                sample_weight_group = sample_weight[g_idx]

                weight_mask = weights_np_group != 0
                sample_weight_group = sample_weight_group * weight_mask

                if np.sum(sample_weight_group) == 0:
                    sample_weight_group = np.ones_like(sample_weight_group)

                orig_centroids = seed_lut[m_idx][r_idx][g_idx]
                lut_per_bit_per_group, parent_weights_per_group = _upscale_group(orig_centroids, orig_labels,
                                                                                 weights_np_group,
                                                                                 sample_weight_group,
                                                                                 seed_bit, parent_bit)

                for k, bit in enumerate(range(seed_bit, parent_bit + 1)):
                    lut_by_modules_by_bit[k * len(seed_lut) + m_idx][r_idx][g_idx] = lut_per_bit_per_group[k]

                parent_weights_by_modules[m_idx][r_idx][g_idx] = parent_weights_per_group

    return lut_by_modules_by_bit, parent_weights_by_modules


def _upscale_layer(seed_lut, seed_weights, model_layer, gradient_layer, seed_bit, parent_bit):
    if seed_bit == parent_bit:
        return seed_lut, seed_weights

    # Convert gradients and model weights to np.float32 for numba
    gradient_layer = [x.float().numpy() for x in gradient_layer]
    model_layer = [x.float().numpy() for x in model_layer]

    return _upscale_layer_njit(seed_lut, seed_weights, model_layer, gradient_layer, seed_bit, parent_bit)


def __load_values(seed_lut_path, seed_weights_path, module_names, l):
    seed_lut_layer = torch.load(f"{seed_lut_path}/l{l}.pt")
    seed_weights_layer = torch.load(f"{seed_weights_path}/l{l}.pt")

    seed_lut_list, orig_weight_list = [], []
    for name in module_names:
        seed_lut_list.append(seed_lut_layer[name].astype(np.float32))
        orig_weight_list.append(seed_weights_layer[name].astype(np.uint8))

    return seed_lut_list, orig_weight_list


def __save_results(parent_parameters_path, seed_precision, parent_precision, module_names,
                   luts_by_modules_by_bit, parent_weights, l):
    # Note that it is important to cast the luts to fp16 before saving them,
    # as we previously casted them to fp32 to use with numba
    for i, bit in enumerate(range(seed_precision, parent_precision + 1)):
        output_lut_file_name = f"{parent_parameters_path}/lut_{bit}/l{l}.pt"
        os.makedirs(os.path.dirname(output_lut_file_name), exist_ok=True)
        lut_dict = {module_names[j]: luts_by_modules_by_bit[i * len(module_names) + j].astype(np.float16)
                    for j in range(len(module_names))}
        torch.save(lut_dict, output_lut_file_name)

    parent_weight_dict = {module_names[j]: parent_weights[j].astype(np.uint8)
                          for j in range(len(module_names))}

    output_weights_layer_file_name = f"{parent_parameters_path}/weights/l{l}.pt"
    os.makedirs(os.path.dirname(output_weights_layer_file_name), exist_ok=True)
    torch.save(parent_weight_dict, output_weights_layer_file_name)


def _get_loader(seed_lut_path, seed_weights_path, module_names):
    """Returns a function that loads the values for a given layer"""

    def load_values(l):
        return __load_values(seed_lut_path, seed_weights_path, module_names, l)

    return load_values


def _get_saver(parent_parameters_path, seed_precision, parent_precision, module_names):
    """Returns a function that saves the results for a given layer"""

    def save_results(luts_by_modules_by_bit, parent_weights, l):
        return __save_results(parent_parameters_path, seed_precision, parent_precision, module_names,
                              luts_by_modules_by_bit, parent_weights, l)

    return save_results


def _load_progress(ran, parent_parameters_path, seed_precision, parent_precision):
    # Check if the layer has already been processed
    todo_ran = []
    processed_ran = []
    for l in ran:
        if all([os.path.exists(f"{parent_parameters_path}/lut_{bit}/l{l}.pt")
                for bit in range(seed_precision, parent_precision + 1)]) and \
                os.path.exists(f"{parent_parameters_path}/weights/l{l}.pt"):
            processed_ran.append(l)
        else:
            todo_ran.append(l)
    return todo_ran, processed_ran


def upscale(
        analyzer,
        seed_precision,
        parent_precision,
        seed_parameters_path,
        parent_parameters_path,
        gradients,
        cpu_count=None
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

    assert seed_precision <= parent_precision, "Parent precision should be equal or higher than seed precision"

    if seed_precision == parent_precision:
        # Nothing to do, just copy the seed to the parent
        logging.info(f"Parent precision is the same as seed precision. Symlinking the seed to the parent...")
        os.makedirs(parent_parameters_path, exist_ok=True)
        os.makedirs(f"{parent_parameters_path}/lut_{seed_precision}", exist_ok=True)
        os.makedirs(f"{parent_parameters_path}/weights", exist_ok=True)
        for l in range(analyzer.num_layers):
            # copy the seed LUTs using symlinks
            os.symlink(os.path.abspath(f"{seed_parameters_path}/lut/l{l}.pt"),
                       f"{parent_parameters_path}/lut_{seed_precision}/l{l}.pt")
            # copy the seed weights using symlinks
            os.symlink(os.path.abspath(f"{os.getcwd()}/{seed_parameters_path}/weights/l{l}.pt"),
                       f"{parent_parameters_path}/weights/l{l}.pt")
        return

    logging.info(f"Upscaling from {seed_precision} to {parent_precision}")

    logging.info("Loading original model weights...")

    model_weights = analyzer.get_model_weights()

    ran = list(range(len(model_weights)))

    module_names = analyzer.module_names

    # Format the weights and gradients
    logging.info("Formatting weights and gradients...")
    model_weights_by_layer = []
    gradients_by_layer = []
    for l in ran:
        model_layer_list, gradient_layer_list = [], []
        for name in module_names:
            model_layer_list.append(model_weights[l][name])
            gradient_layer_list.append(gradients[l][name])
        model_weights_by_layer.append(model_layer_list)
        gradients_by_layer.append(gradient_layer_list)

    ran, completed = _load_progress(ran, parent_parameters_path, seed_precision, parent_precision)

    if completed:
        logging.info(f"The following layers will be skipped as they have already been processed:\n{completed}")
        logging.info(f"To reprocess these layers, delete the corresponding files in {parent_parameters_path}")

    if not ran:
        logging.info("All layers have already been processed. Exiting...")
        return

    seed_weights_path = f"{seed_parameters_path}/weights"
    seed_lut_path = f"{seed_parameters_path}/lut"

    logging.info(f"Quantizing layers {ran}")

    load_values = _get_loader(seed_lut_path, seed_weights_path, module_names)

    save_results = _get_saver(parent_parameters_path, seed_precision, parent_precision, module_names)

    if pipelined_io:
        with ThreadPoolExecutor(max_workers=io_workers) as io_executor:
            for l in tqdm(ran, desc="Quantizing layers"):
                if l == ran[0]:
                    future_load = io_executor.submit(load_values, l)

                seed_lut_layer, seed_qweight_layer = future_load.result()

                if l != ran[-1]:
                    future_load = io_executor.submit(load_values, l + 1)

                luts_by_modules_by_bit, parent_weights = _upscale_layer(
                    seed_lut_layer,
                    seed_qweight_layer,
                    model_weights_by_layer[l],
                    gradients_by_layer[l],
                    seed_precision,
                    parent_precision
                )

                io_executor.submit(save_results, luts_by_modules_by_bit, parent_weights, l)
            logging.info("Waiting for IO to finish...")
    else:
        for l in tqdm(ran, desc="Quantizing layers"):
            seed_lut_layer, seed_qweight_layer = load_values(l)

            luts_by_modules_by_bit, parent_weights = _upscale_layer(
                seed_lut_layer,
                seed_qweight_layer,
                model_weights_by_layer[l],
                gradients_by_layer[l],
                seed_precision,
                parent_precision
            )

            save_results(luts_by_modules_by_bit, parent_weights, l)
