# It is CRITICAL that the seed module is imported before any other module that uses numpy or torch
# as other imports may cause issues with threadpoolctl on certain machines, leading to performance drops.
from .seed import get_seed

import os
import os.path
import shutil
import torch
import argparse
import logging

from .config import *
from .utils import load_model, load_tokenizer
from .analyzer import get_analyzer
from .gradients import get_gradients
from .upscale import upscale
from .pack import pack

# Disable parallelism in tokenizers to prevent warnings when forking in the seed generation step
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logging with time sans date, level name, and message
logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s] %(message)s', datefmt='%H:%M:%S')


def quantize_any_precision(model,
                           seed_precision=DEFAULT_SEED_PRECISION,
                           parent_precision=DEFAULT_PARENT_PRECISION,
                           mode='upscale',
                           yaml_path=None, cache_dir=DEFAULT_CACHE_DIR,
                           dataset=DEFAULT_DATASET, seq_len=DEFAULT_SEQ_LEN, num_examples=DEFAULT_NUM_EXAMPLES,
                           cpu_count=os.cpu_count(),
                           recalculate_gradients=False,
                           recalculate_seed=False,
                           ):
    assert mode in ['gradients', 'seed', 'upscale'], \
        "mode must be one of 'gradients', 'seed', or 'upscale'. Use 'upscale' to run the entire pipeline."

    if recalculate_gradients:
        if not recalculate_seed:
            logging.warning("Seed model needs to be recalculated if gradients are recalculated. "
                            "Setting recalculate_seed to True.")
            recalculate_seed = True

    if mode == 'gradients':
        logging.info("Running: [Gradients]")
    elif mode == 'seed':
        logging.info("Running: [Gradients -> Seed]")
    else:
        logging.info("Running: [Gradients -> Seed -> Upscale]")

    model_string = model if isinstance(model, str) else model.name_or_path
    model_name = model_string.split("/")[-1]

    logging.info(f"Running Any-Precision Quantization on {model_name} with seed precision {seed_precision} and "
                 f"parent precision {parent_precision} using {dataset} for gradient calculation")

    # ------------------- Load model -------------------

    model = load_model(model)
    tokenizer = load_tokenizer(model_string)

    analyzer = get_analyzer(model, yaml_path=yaml_path)

    # ------------------- Gradients -------------------

    logging.info("------------------- Gradients -------------------")

    gradients_cache_path = f"{cache_dir}/gradients/({model_name})-{dataset}_s{num_examples}_blk{seq_len}.pt"

    # Calculate or load gradients
    if not recalculate_gradients and os.path.exists(gradients_cache_path):
        model_gradients = torch.load(gradients_cache_path)
        logging.info(f"Loaded cached gradients from {gradients_cache_path}")
    else:
        logging.info("Beginning gradient calculation...")
        # this will overwrite the gradients cache if it already exists
        model_gradients = get_gradients(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            seq_len=seq_len,
            num_examples=num_examples,
            analyzer=analyzer,
            save_path=gradients_cache_path,
        )
        logging.info("Gradient calculation complete.")

    if mode == 'gradients':
        return

    # ------------------- Seed -------------------

    logging.info("------------------- Seed -------------------")

    seed_cache_path = f"{cache_dir}/seed/({model_name})-w{seed_precision}-{dataset}_s{num_examples}_blk{seq_len}"

    # Calculate or load seed
    logging.info(f"Beginning {seed_precision}-bit seed model generation...")
    # Note that this saves the seed model to the cache path and must be loaded for the upscale step
    if recalculate_seed and os.path.exists(seed_cache_path):
        # if the user wants to recalculate the seed, delete the cached seed
        logging.info(f"Detected cached seed at {seed_cache_path}. Will delete and recalculate.")
        input(f"To proceed with the deletion of {seed_cache_path}, press Enter. Else, press Ctrl+C to abort.")
        shutil.rmtree(seed_cache_path)

    # this skips over existing layers in the cache, and doesn't overwrite them
    get_seed(
        model=model,
        gradients=model_gradients,
        bit_width=seed_precision,
        output_folder=seed_cache_path,
        analyzer=analyzer,
        cpu_count=cpu_count,
    )
    logging.info("Seed calculation complete.")

    if mode == 'seed':
        return

    # ------------------- Upscale -------------------

    logging.info("------------------- Upscale -------------------")

    parent_cache_path = (f"{cache_dir}/parent/({model_name})-w{parent_precision}_orig{seed_precision}"
                         f"-{dataset}_s{num_examples}_blk{seq_len}")

    upscale(
        model=model,
        seed_precision=seed_precision,
        parent_precision=parent_precision,
        analyzer=analyzer,
        seed_parameters_path=seed_cache_path,
        parent_parameters_path=parent_cache_path,
        gradients=model_gradients,
        cpu_count=cpu_count,
    )

    logging.info("Upscale complete.")

    # ------------------- Pack -------------------
    logging.info("------------------- Pack -------------------")

    model_output_path = (f"{cache_dir}/packed/anyprec-({model_name})-w{parent_precision}_orig{seed_precision}"
                         f"-{dataset}_s{num_examples}_blk{seq_len}")

    # check for non-empty directory
    if os.path.exists(model_output_path) and os.path.isdir(model_output_path) and os.listdir(model_output_path):
        logging.info(f"Model output path {model_output_path} already exists and is not empty.")
        input(f"To proceed and overwrite {model_output_path}, press Enter. Else, press Ctrl+C to abort.")

    pack(
        model=model,
        tokenizer=tokenizer,
        lut_path=parent_cache_path,
        output_model_path=model_output_path,
        seed_precision=seed_precision,
        parent_precision=parent_precision,
        analyzer=analyzer,
        cpu_count=cpu_count,
    )

    logging.info("Packing complete.")
