
import os
import os.path
import shutil
import logging

from .config import *
from ..analyzer import get_analyzer
from .gradients import get_gradients
from .quantize import seed_and_upscale
from .pack import pack
from .dense_and_sparse import remove_outliers
import torch

# Disable parallelism in tokenizers to prevent warnings when forking in the seed generation step
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logging with time sans date, level name, and message
logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s] %(message)s', datefmt='%H:%M:%S')


def any_precision_quantize(
        model,
        seed_precision=DEFAULT_SEED_PRECISION,
        parent_precision=DEFAULT_PARENT_PRECISION,
        mode='pack',
        yaml_path=None, cache_dir=DEFAULT_CACHE_DIR,
        dataset=DEFAULT_DATASET, seq_len=DEFAULT_SEQ_LEN, num_examples=DEFAULT_NUM_EXAMPLES,
        cpu_count=os.cpu_count(),
        overwrite_gradients=False,
        overwrite_quantize=False,
        overwrite_pack=False,
        random_state=None,
        group_count=1,
        dns=False,
        sensitivity_outlier_percent=0.05,
        threshold_outlier_percent=0.40,
        cpu_only=False,
):
    assert mode in ['gradients', 'quantize', 'pack'], \
        "mode must be one of 'gradients', 'quantize', or 'pack'. Use 'pack' to run the entire pipeline."

    if overwrite_gradients:
        if not overwrite_quantize:
            logging.warning("Parent model needs to be recalculated if gradients are recalculated. "
                            "Setting overwrite_quantize to True.")
            overwrite_quantize = True

    if overwrite_quantize:
        if not overwrite_pack:
            logging.warning("Packed model needs to be recalculated if parent model is recalculated. "
                            "Setting overwrite_pack to True.")
            overwrite_pack = True

    if mode == 'gradients':
        logging.info("Running: [Gradients]")
    elif mode == 'quantize':
        logging.info("Running: [Gradients -> Quantize]")
    else:
        logging.info("Running: [Gradients -> Quantize -> Pack]")

    model_string = model if isinstance(model, str) else model.name_or_path
    model_name = model_string.split("/")[-1]

    logging.info(f"Running Any-Precision Quantization on {model_name} with seed precision {seed_precision} and "
                 f"parent precision {parent_precision} using {dataset} for gradient calculation")

    # ------------------- Load model -------------------

    analyzer = get_analyzer(model, yaml_path=yaml_path, include_tokenizer=True, cpu_only=cpu_only)

    # ------------------- Set cache paths -------------------

    gradients_cache_path = (f"{cache_dir}/gradients/"
                            f"({model_name})-{dataset}_s{num_examples}_blk{seq_len}.pt")

    quantized_cache_path = (f"{cache_dir}/quantized/"
                          f"{'dns-' if dns else ''}({model_name})-w{parent_precision}_orig{seed_precision}"
                          f"-gc{group_count}-{dataset}_s{num_examples}_blk{seq_len}")

    model_output_path = (f"{cache_dir}/packed/"
                         f"anyprec-({model_name})-w{parent_precision}_orig{seed_precision}"
                         f"-gc{group_count}-{dataset}_s{num_examples}_blk{seq_len}")

    # ------------------- Gradients -------------------

    logging.info("------------------- Gradients -------------------")

    logging.info("Beginning gradient calculation...")
    # Calculate or load gradients
    if overwrite_gradients and os.path.exists(gradients_cache_path):
        # if the user wants to recalculate the gradients, delete the cached gradients
        logging.info(f"Detected cached gradients at {gradients_cache_path}. Will delete and recalculate.")
        os.remove(gradients_cache_path)

    # this will load and return the gradients if they exist, or calculate them if they don't
    model_gradients = get_gradients(
        analyzer=analyzer,
        dataset=dataset,
        seq_len=seq_len,
        num_examples=num_examples,
        save_path=gradients_cache_path,
        random_state=random_state,
    )
    logging.info("Gradient calculation complete.")

    if mode == 'gradients':
        return

    # ------------------- Dense & Sparse -------------------

    if dns:
        logging.info("------------------- Dense & Sparse -------------------")
        sparse_model_weights = remove_outliers(
            analyzer=analyzer,
            gradients=model_gradients,
            sensitivity_outlier_percent=sensitivity_outlier_percent,
            threshold_outlier_percent=threshold_outlier_percent,
        )

        sparse_path = f"{quantized_cache_path}/sparse"
        os.makedirs(sparse_path, exist_ok=True)
        for l in range(analyzer.num_layers):
            torch.save(sparse_model_weights[l], f"{sparse_path}/l{l}.pt")

        del sparse_model_weights

    # ------------------- Quantize: Seed + Upscale -------------------

    logging.info("------------------- Quantize: Seed + Upscale -------------------")

    # Calculate or load parent
    logging.info(f"Beginning {seed_precision}~{parent_precision}-bit Any-Precision Quantization...")
    # Note that this saves the seed model to the cache path and must be loaded for the upscale step
    if overwrite_quantize and os.path.exists(quantized_cache_path):
        # if the user wants to recalculate the seed, delete the cached seed
        logging.info(f"Detected cached parent at {quantized_cache_path}. Will delete and recalculate.")
        shutil.rmtree(quantized_cache_path)

    # this skips over existing layers in the cache, and doesn't overwrite them
    seed_and_upscale(
        analyzer=analyzer,
        gradients=model_gradients,
        output_folder=quantized_cache_path,
        seed_precision=seed_precision,
        parent_precision=parent_precision,
        cpu_count=cpu_count,
        random_state=random_state,
        group_count=group_count,
    )

    if mode == 'quantize':
        return

    del model_gradients  # free up memory
    analyzer.drop_original_weights()  # drop the original weights to save memory

    logging.info("Quantization(Seed + Upscale) complete.")

    # ------------------- Pack -------------------
    logging.info("------------------- Pack -------------------")

    # check for non-empty directory
    if os.path.exists(model_output_path) and os.path.isdir(model_output_path) and os.listdir(model_output_path):
        if overwrite_pack:
            logging.info(f"Model output path {model_output_path} already exists and is not empty. Will delete and "
                         f"re-pack.")
            shutil.rmtree(model_output_path)
        else:
            # if the user doesn't want to overwrite the pack, but the directory is not empty, skip packing
            logging.info(f"Model output path {model_output_path} already exists and is not empty. Will skip packing.")
            return

    pack(
        analyzer=analyzer,
        lut_path=quantized_cache_path,
        output_model_path=model_output_path,
        seed_precision=seed_precision,
        parent_precision=parent_precision,
        cpu_count=cpu_count,
        group_count=group_count,
        dns=dns,
    )

    logging.info("Packing complete.")
