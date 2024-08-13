import numba
import logging
import torch

import numpy as np
from tqdm.contrib.concurrent import process_map

from tqdm import tqdm


@numba.njit(cache=True)
def _module_get_threshold_from_range(weights, trange):
    assert len(weights.shape) == 1, "Weights must be 1D"
    # Assumes sorted weights, O(1)
    q1 = weights[len(weights) // 4]
    q3 = weights[3 * len(weights) // 4]
    low = q1 - trange * (q3 - q1)
    high = q3 + trange * (q3 - q1)
    larger_abs = max(abs(low), abs(high))
    return larger_abs


@numba.njit(cache=True)
def _module_get_outlier_count_from_threshold(weights, threshold):
    # Assumes sorted weights, O(log n)
    return np.searchsorted(weights, -threshold) + len(weights) - np.searchsorted(weights, threshold)


@numba.njit(cache=True)
def _module_get_outlier_count_from_range(weights, trange):
    # Assumes sorted weights, O(log n)
    threshold = _module_get_threshold_from_range(weights, trange)
    return _module_get_outlier_count_from_threshold(weights, threshold), threshold


@numba.njit(cache=True)
def _get_outlier_count_from_range(trange, sorted_flattened_weights):
    total_outliers = 0
    thresholds = np.empty(len(sorted_flattened_weights), dtype=np.float32)
    for i, module_weight in enumerate(sorted_flattened_weights):
        num_outliers, threshold = _module_get_outlier_count_from_range(module_weight, trange)
        thresholds[i] = threshold
        total_outliers += num_outliers
    return total_outliers, thresholds


def _process_module(module_data):
    layer_index, module_name, module_weight = module_data
    sorted_weights = np.sort(module_weight.flatten().cpu().data.to(torch.float32).numpy()).astype(np.float32)  # fp32 for numba
    total_params = module_weight.numel()
    return layer_index, module_name, sorted_weights, total_params


def _find_thresholds(analyzer, outlier_percent, tolerance=0.0001):
    assert outlier_percent < 50, "Outlier ratio must be less than 0.5"

    model_weights = [analyzer.get_layer_weights(l) for l in range(analyzer.num_layers)]

    tasks = []
    for layer_index, model_layer in enumerate(model_weights):
        for module_name in analyzer.module_names:
            module_weight = model_layer[module_name]
            tasks.append((layer_index, module_name, module_weight))

    sorted_flattened_weights = []
    total_params = 0

    # results = process_map(_process_module, tasks, chunksize=1, max_workers=None, desc="Preprocessing weights")
    results = []
    for task in tqdm(tasks):
        results.append(_process_module(task))


    for layer_index, module_name, sorted_weights, params in results:
        sorted_flattened_weights.append(sorted_weights)
        total_params += params

    # Find the trange by binary search
    low = 0
    high = 32  # this seems like an extra overkill upper bound but adjust if necessary
    thresholds = None
    logging.info(f"Begin trange search for outlier percent {outlier_percent}%")
    while low < high:
        mid = (low + high) / 2  # Note that this is a float as we are searching for a float threshold
        total_outliers, thresholds = _get_outlier_count_from_range(mid, sorted_flattened_weights)
        percent = total_outliers / total_params * 100
        logging.info(f"Search range: [{low:.5f}, {high:.4f}] Threshold: {mid:.5f}, Outlier ratio: {percent:.5f}%")
        if abs(percent - outlier_percent) < tolerance:
            logging.info(f"Found threshold: {mid:.5f}, Outlier ratio: {percent:.5f}%")
            break
        elif percent < outlier_percent:
            high = mid
        else:
            low = mid

    thresholds_by_module = [{} for _ in range(analyzer.num_layers)]
    idx = 0
    for layer_index in range(analyzer.num_layers):
        for module_name in analyzer.module_names:
            thresholds_by_module[layer_index][module_name] = thresholds[idx]
            idx += 1

    return thresholds_by_module


def _remove_outliers_by_threshold(analyzer, thresholds_by_module):
    sparse_model_weights = []

    for l in tqdm(range(analyzer.num_layers), desc="Removing threshold outliers"):
        model_layer = analyzer.get_layer_weights(l)
        sparse_model_weights_by_layer = {}
        for module_name in analyzer.module_names:
            module_weight = model_layer[module_name]
            threshold = thresholds_by_module[l][module_name]
            dense_mask = torch.abs(module_weight) < threshold
            sparse_mask = ~dense_mask
            # Save the sparse weights
            sparse_model_weights_by_layer[module_name] = module_weight * sparse_mask
            # Zero out the outliers
            module_weight[sparse_mask] = 0

        sparse_model_weights.append(sparse_model_weights_by_layer)

    return sparse_model_weights


def _remove_outliers_by_sensitivity(analyzer, gradients, sensitivity_outlier_percent):
    if sensitivity_outlier_percent == 0.0:
        return None

    sparse_model_weights = []

    for l in tqdm(range(analyzer.num_layers), desc="Removing sensitivity outliers"):
        model_layer = analyzer.get_layer_weights(l)
        sparse_model_weights_by_layer = {}
        for module_name in analyzer.module_names:
            module_weight = model_layer[module_name]
            gradient = gradients[l][module_name]  # this is a torch tensor
            # get the top sensitivity_outlier_percent% of the gradients
            topk = int(sensitivity_outlier_percent * gradient.numel() / 100)
            _, indices = torch.topk(gradient.abs().flatten(), topk)
            sparse_mask = torch.zeros_like(module_weight, dtype=torch.bool)
            sparse_mask.view(-1)[indices] = 1
            # Save the sparse weights
            sparse_model_weights_by_layer[module_name] = module_weight * sparse_mask
            # Zero out the outliers
            module_weight[sparse_mask] = 0

        sparse_model_weights.append(sparse_model_weights_by_layer)

    return sparse_model_weights


def remove_outliers(analyzer, gradients, sensitivity_outlier_percent, threshold_outlier_percent):
    # This removes the sensitivity outliers from the weights stored in the analyzer, and returns the removed weights
    sparse_model_weights_1 = _remove_outliers_by_sensitivity(analyzer, gradients, sensitivity_outlier_percent)

    # Find the thresholds to achieve the desired outlier percentage
    thresholds_by_module = _find_thresholds(analyzer, threshold_outlier_percent)

    # This removes the threshold outliers from the weights stored in the analyzer, and returns the removed weights
    sparse_model_weights_2 = _remove_outliers_by_threshold(analyzer, thresholds_by_module)

    # Add the two sets of sparse weights
    sparse_model_weights = []
    for l in range(analyzer.num_layers):
        sparse_model_weights_by_layer = {}
        for module_name in analyzer.module_names:
            sparse_out = sparse_model_weights_2[l][module_name]
            if sparse_model_weights_1 is not None:
                sparse_out += sparse_model_weights_1[l][module_name]
            sparse_model_weights_by_layer[module_name] = sparse_out.to_sparse()
        sparse_model_weights.append(sparse_model_weights_by_layer)

    return sparse_model_weights
