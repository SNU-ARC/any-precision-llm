import os
import torch
from concurrent.futures import ThreadPoolExecutor
import argparse
import numpy as np
import numba
from tqdm import tqdm

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


@numba.njit(cache=True)
def _upscale_group(orig_centroids, orig_labels, weights, sample_weight, seed_bit, parent_bit):
    luts_by_bit = [orig_centroids]
    labels = orig_labels
    for i in range(seed_bit, parent_bit):
        centroids, labels = _increment_group(luts_by_bit[-1], labels, weights, sample_weight, i)
        luts_by_bit.append(centroids)

    return luts_by_bit, labels


@numba.njit(cache=True)
def _increment_group(orig_centroids, orig_labels, weights, sample_weight, seed_bit):
    new_centroids = np.empty(2 ** (seed_bit + 1), dtype=np.float32)
    new_labels = np.empty_like(orig_labels)

    for c in range(2 ** seed_bit):
        indices = np.nonzero(orig_labels == c)
        assert len(indices) == 1, "There should be only one index array"
        indices = indices[0]

        selected_weights_np = weights[indices]
        selected_sample_weight = sample_weight[indices]

        # Added by Syphon to prevent KMeans from crashing
        if np.sum(selected_sample_weight) == 0:
            selected_sample_weight = np.ones_like(selected_sample_weight)

        cluster_centers, labels = _faster_1d_two_cluster_kmeans(selected_weights_np, selected_sample_weight)

        if len(cluster_centers) == 0:
            assert len(indices) == 0, "If there are no cluster centers, there should be no indices"
            # These are empty clusters, but we still need to save the centroids
            new_centroids[c * 2] = orig_centroids[c]
            new_centroids[c * 2 + 1] = orig_centroids[c]
            # There are no indices, so we don't need to update the labels
            continue

        assert max(labels) <= 1, "There should be only two labels"
        assert len(cluster_centers) == 2, "There should be two cluster centers"

        # save the new centroids and labels
        new_centroids[c * 2] = cluster_centers[0]
        new_centroids[c * 2 + 1] = cluster_centers[1]
        new_labels[indices] = c * 2 + labels

    return new_centroids, new_labels


@numba.njit(cache=True)
def _faster_1d_two_cluster_kmeans(X, sample_weight):
    """An optimized kmeans for 1D data with 2 clusters.
    This function uses np.float32 instead of np.float16 for the centroids so that numba can compile it.
    Please cast the result back to np.float16 before saving it.
    """
    assert len(X) == len(sample_weight), "X and sample_weight should have the same length"
    # if there are no elements, return arrays of length 0
    # the caller should check for and handle this case
    if len(X) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.uint8)

    centroids = np.empty(2, dtype=np.float32)
    labels = np.empty(len(X), dtype=np.uint8)

    if len(X) == 1:
        centroids[0], centroids[1] = X[0], X[0]
        labels[0] = 0
        return centroids, labels

    if len(X) == 2:
        if X[0] < X[1]:
            centroids[0], centroids[1] = X[0], X[1]
            labels[0], labels[1] = 0, 1
        else:
            centroids[0], centroids[1] = X[1], X[0]
            labels[0], labels[1] = 1, 0

        return centroids, labels

    nonzero_weight_count = np.sum(sample_weight != 0)
    if nonzero_weight_count <= 1:
        if nonzero_weight_count == 0:
            raise ValueError("This case should have been filtered out before calling this function")
        else:
            # If there is only one nonzero sample weight, we need to return the corresponding weight as the centroid
            # and set all labels to 0
            nonzero_weight_index = np.argmax(sample_weight)
            centroids[0], centroids[1] = X[nonzero_weight_index], X[nonzero_weight_index]
            labels[:] = 0
            return centroids, labels

    # Now we know that there are at least 3 elements and at least 2 nonzero weights

    # KMeans with 2 clusters on 1D data is equivalent to finding a division point.
    # The division point can be found by doing a binary search on the prefix sum.

    # First we sort the data
    sorted_indices = np.argsort(X)

    weighted_X = (X * sample_weight)[sorted_indices]
    # Use fp64 in prefix sum to avoid precision loss in subtraction
    weighted_X_prefix_sum = np.cumsum(weighted_X.astype(np.float64))
    sample_weight_prefix_sum = np.cumsum(sample_weight[sorted_indices].astype(np.float64))

    # We will do a search for the division point,
    # where we search for the optimum number of elements in the first cluster
    # We don't want empty clusters, so we set the floor and ceiling to 1 and len(X) - 1
    floor = 1
    ceiling = len(X)
    left_center = None
    right_center = None

    while floor + 1 < ceiling:
        left_cluster_size = (floor + ceiling) // 2
        # If the left cluster has no weight, we need to move the floor up
        left_weight_sum = sample_weight_prefix_sum[left_cluster_size - 1]
        if left_weight_sum == 0:
            floor = left_cluster_size
            continue
        right_weight_sum = sample_weight_prefix_sum[-1] - left_weight_sum
        # If the right cluster has no weight, we need to move the ceiling down
        if right_weight_sum == 0:
            ceiling = left_cluster_size
            continue

        left_center = weighted_X_prefix_sum[left_cluster_size - 1] / left_weight_sum
        right_center = (weighted_X_prefix_sum[-1] - weighted_X_prefix_sum[left_cluster_size - 1]) / right_weight_sum

        new_division_point = (left_center + right_center) / 2
        if X[sorted_indices[left_cluster_size - 1]] <= new_division_point:
            if new_division_point <= X[sorted_indices[left_cluster_size]]:
                break
            else:
                floor = left_cluster_size
        else:
            ceiling = left_cluster_size

    # initialize variables in case the loop above does not run through
    if left_center is None:
        left_cluster_size = (floor + ceiling) // 2
        left_center = weighted_X_prefix_sum[left_cluster_size - 1] / sample_weight_prefix_sum[left_cluster_size - 1]
    if right_center is None:
        left_cluster_size = (floor + ceiling) // 2
        right_center = (weighted_X_prefix_sum[-1] - weighted_X_prefix_sum[left_cluster_size - 1]) / \
                       (sample_weight_prefix_sum[-1] - sample_weight_prefix_sum[left_cluster_size - 1])

    # avoid using lists to allow numba.njit
    centroids[0] = left_center
    centroids[1] = right_center

    labels[sorted_indices[:left_cluster_size]] = 0
    labels[sorted_indices[left_cluster_size:]] = 1

    return centroids, labels


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
    # Convert the torch tensors to numpy arrays and to float32 so that numba can use them
    model_layer = [x.float().numpy() for x in model_layer]
    gradient_layer = [x.float().numpy() for x in gradient_layer]
    return _upscale_layer_njit(seed_lut, seed_weights, model_layer, gradient_layer, seed_bit, parent_bit)


def __load_values(seed_lut_path, seed_weights_path, layer_names, l):
    seed_lut_layer = torch.load(f"{seed_lut_path}/l{l}.pt")
    seed_weights_layer = torch.load(f"{seed_weights_path}/l{l}.pt")

    seed_lut_list, orig_weight_list = [], []
    for name in layer_names:
        seed_lut_list.append(seed_lut_layer[name].astype(np.float32))
        orig_weight_list.append(seed_weights_layer[name].astype(np.uint8))

    return seed_lut_list, orig_weight_list


def __save_results(parent_parameters_path, seed_precision, parent_precision, layer_names,
                   luts_by_modules_by_bit, parent_weights, l):
    # Note that it is important to cast the luts to fp16 before saving them,
    # as we previously casted them to fp32 to use with numba
    for i, bit in enumerate(range(seed_precision, parent_precision + 1)):
        output_lut_file_name = f"{parent_parameters_path}/lut_{bit}/l{l}.pt"
        os.makedirs(os.path.dirname(output_lut_file_name), exist_ok=True)
        lut_dict = {layer_names[j]: luts_by_modules_by_bit[i * len(layer_names) + j].astype(np.float16)
                    for j in range(len(layer_names))}
        torch.save(lut_dict, output_lut_file_name)

    parent_weight_dict = {layer_names[j]: parent_weights[j].astype(np.uint8)
                          for j in range(len(layer_names))}

    output_weights_layer_file_name = f"{parent_parameters_path}/weights/l{l}.pt"
    os.makedirs(os.path.dirname(output_weights_layer_file_name), exist_ok=True)
    torch.save(parent_weight_dict, output_weights_layer_file_name)


def _get_loader(seed_lut_path, seed_weights_path, layer_names):
    """Returns a function that loads the values for a given layer"""

    def load_values(l):
        return __load_values(seed_lut_path, seed_weights_path, layer_names, l)

    return load_values


def _get_saver(parent_parameters_path, seed_precision, parent_precision, layer_names):
    """Returns a function that saves the results for a given layer"""

    def save_results(luts_by_modules_by_bit, parent_weights, l):
        return __save_results(parent_parameters_path, seed_precision, parent_precision, layer_names,
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


def upscale(analyzer, seed_precision, parent_precision, seed_parameters_path,
            parent_parameters_path, gradients, cpu_count=None):
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

    assert seed_precision < parent_precision, "Parent precision should be higher than seed precision"
    print(f"Upscaling from {seed_precision} to {parent_precision}")

    print("Loading original model weights...")

    model_weights = analyzer.get_model_weights()

    ran = list(range(len(model_weights)))

    # Load gradients
    if isinstance(gradients, str):
        print("Loading gradients...")
        gradients = torch.load(gradients)
    else:
        assert isinstance(gradients, list), "gradients should be a string or a list"

    module_names = analyzer.module_names

    # Format the weights and gradients
    print("Formatting weights and gradients...")
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
        print("The following layers will be skipped as they have already been processed:")
        print(completed)
        print(f"To reprocess these layers, delete the corresponding files in {parent_parameters_path}")

    if not ran:
        print("All layers have already been processed. Exiting...")
        return

    seed_weights_path = f"{seed_parameters_path}/weights"
    seed_lut_path = f"{seed_parameters_path}/lut"

    print(f"Quantizing layers {ran}")

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

                luts_by_modules_by_bit, parent_weights = _upscale_layer(seed_lut_layer, seed_qweight_layer,
                                                                        model_weights_by_layer[l],
                                                                        gradients_by_layer[l],
                                                                        seed_precision,
                                                                        parent_precision)

                io_executor.submit(save_results, luts_by_modules_by_bit, parent_weights, l)
            print("Waiting for IO to finish...")
    else:
        for l in tqdm(ran, desc="Quantizing layers"):
            seed_lut_layer, seed_qweight_layer = load_values(l)

            luts_by_modules_by_bit, parent_weights = _upscale_layer(seed_lut_layer, seed_qweight_layer,
                                                                    model_weights_by_layer[l],
                                                                    gradients_by_layer[l],
                                                                    seed_precision,
                                                                    parent_precision)

            save_results(luts_by_modules_by_bit, parent_weights, l)
