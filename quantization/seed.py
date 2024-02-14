import os

os.environ["OMP_NUM_THREADS"] = "1"  # this is necessary to parallelize the kmeans
from multiprocessing import Pool
import torch
import argparse
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import utils
import transformers

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model', type=str,
    help='model weights to load', required=True
)
parser.add_argument(
    '--model_type', type=str, default=None,
    help='model type', choices=['llama', 'opt', 'mistral', 'phi-2']
)
parser.add_argument(
    '--gradient', type=str,
    help='model gradients to load', required=True
)
parser.add_argument(
    '--bit', type=int, default=3,
    help='bitwidth', choices=[2, 3, 4, 5, 6, 7, 8],
)
parser.add_argument(
    '--range', type=str, default=None,
    help='range of layers to quantize'
)
parser.add_argument(
    '--output_folder', type=str, required=False,
    help='path to dump the output'
)

parser.add_argument(
    '--cores', type=int, default=os.cpu_count() - 1,
    help='number of cores to use for parallelization'
)


def kmeans_fit(args_tuple):
    """
    Helper function to parallelize the kmeans
    """
    X, sample_weight, n_clusters, random_state, n_init, max_iter = args_tuple
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    ).fit(
        X,
        sample_weight=sample_weight,
    )
    # re-label the clusters to be in order of increasing weight
    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_.reshape(-1)
    labels = kmeans.labels_

    # First argsort to get the indices of the sorted cluster centers
    sorted_cluster_indices = np.argsort(cluster_centers)

    # Then argsort again to get the indices of the sorted indices
    label_mapping = np.argsort(sorted_cluster_indices)

    # Apply the mapping to the labels
    labels = label_mapping[labels]

    # Get the new cluster centers
    cluster_centers = cluster_centers[sorted_cluster_indices]

    # Cast to appropriate types
    cluster_centers = cluster_centers.astype(np.float16)
    labels = labels.astype(np.uint8)

    return cluster_centers, labels


def main(output_folder, model_path, model_type, gradient_path, bit_width, cpu_count):
    lut_folder = f"{output_folder}/lut"
    if not os.path.exists(lut_folder):
        os.makedirs(lut_folder)

    weight_folder = f"{output_folder}/weights"
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model_weights = utils.get_model_weights(model, model_type)
    del model

    gradients = torch.load(gradient_path)

    print(f"Quantizing layers {list(range(len(model_weights)))}")

    pool = Pool(cpu_count)

    with tqdm(total=len(utils.get_module_names(model_type)) * len(model_weights), desc="Quantizing layers") as pbar:
        for l in range(len(model_weights)):
            if os.path.exists(f"{lut_folder}/l{l}.pt") and os.path.exists(f"{weight_folder}/l{l}.pt"):
                print(f"Skipping layer {l}, file already exists")
                pbar.update(len(utils.get_module_names(model_type)))
                continue

            gradient_layer = gradients[l]
            model_layer = model_weights[l]

            # generate kmeans tasks
            kmeans_jobs_by_module = []
            for name in utils.get_module_names(model_type):
                g = gradient_layer[name].float().numpy()

                module_weight = model_layer[name]
                _weights_np = module_weight.float().numpy()  # Syphon: Added the .float() here as BF16 was causing issues

                n_cluster = 2 ** bit_width
                kmeans_jobs_per_module = []
                # iterate over row, generating kmeans tasks
                for i in range(module_weight.shape[0]):
                    weights_np_temp = _weights_np[i, :]
                    weights_np = weights_np_temp.reshape(-1, 1)

                    weight_mask = weights_np_temp != 0
                    sample_weight = g[i, :]
                    sample_weight = sample_weight * weight_mask

                    if np.sum(sample_weight) == 0:
                        sample_weight = np.ones_like(sample_weight)

                    kmeans_jobs_per_module.append((weights_np, sample_weight, n_cluster, 0, "auto", 50))
                kmeans_jobs_by_module.append(kmeans_jobs_per_module)

            # run kmeans using a pool of processes
            kmeans_results_by_module = []
            for kmeans_jobs_per_module in kmeans_jobs_by_module:
                kmeans_results_per_module = list(pool.map(kmeans_fit, kmeans_jobs_per_module))
                kmeans_results_by_module.append(kmeans_results_per_module)
                pbar.update()

            lut_per_layer, weight_per_layer = {}, {}

            # postprocess kmeans results
            for i, name in enumerate(utils.get_module_names(model_type)):
                lut_per_row, weight_per_row = [], []
                module_weight = model_layer[name]
                for j in range(module_weight.shape[0]):
                    lut_per_row.append([kmeans_results_by_module[i][j][0]])
                    weight_per_row.append([kmeans_results_by_module[i][j][1]])

                lut_per_layer[name] = np.array(lut_per_row)
                weight_per_layer[name] = np.array(weight_per_row)

            # save parts
            torch.save(lut_per_layer, f"{lut_folder}/l{l}.pt")
            torch.save(weight_per_layer, f"{weight_folder}/l{l}.pt")

    pool.close()


if __name__ == "__main__":
    #args = parser.parse_args()
    # main(args.output_folder, args.model, args.model_type, args.gradient, args.bit, args.cores)
    main('../cache/seed/(opt-1.3b)-c4-w3', 'facebook/opt-1.3b',
         'opt', '../cache/gradients/(opt-1.3b)-c4.pt', 3, os.cpu_count())
