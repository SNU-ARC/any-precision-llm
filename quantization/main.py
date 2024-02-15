import os
import os.path
from transformers import AutoModelForCausalLM
import torch
import argparse
import logging

from config import *
import utils

import gradients
import seed
import upscale

# Logging with time sans date, level name, and message
logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s] %(message)s', datefmt='%H:%M:%S')


def quantize_any_precision(model,
                           seed_precision=DEFAULT_SEED_PRECISION,
                           parent_precision=DEFAULT_PARENT_PRECISION,
                           mode='upscale',
                           model_type=None, cache_dir=DEFAULT_CACHE_DIR,
                           dataset=DEFAULT_DATASET, seq_len=DEFAULT_SEQ_LEN, num_examples=DEFAULT_NUM_EXAMPLES,
                           cpu_count=os.cpu_count(),
                           use_cached_gradients=True):
    assert mode in ['gradients', 'seed', 'upscale'], \
        "mode must be one of 'gradients', 'seed', or 'upscale'. Use 'upscale' to run the entire pipeline."

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

    if isinstance(model, str):
        model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
    else:
        assert isinstance(model, AutoModelForCausalLM), "model must be a string or a transformers model"

    model_type = model_type if model_type is not None else utils.guess_model_type(model)

    # ------------------- Gradients -------------------

    logging.info("------------------- Gradients -------------------")

    gradients_cache_path = f"{cache_dir}/gradients/({model_name})-{dataset}_s{num_examples}_blk{seq_len}.pt"

    # Calculate or load gradients
    if use_cached_gradients and os.path.exists(gradients_cache_path):
        model_gradients = torch.load(gradients_cache_path)
        logging.info(f"Loaded cached gradients from {gradients_cache_path}")
    else:
        logging.info("Beginning gradient calculation...")
        model_gradients = gradients.get_gradients(
            model=model,
            dataset=dataset,
            seq_len=seq_len,
            num_examples=num_examples,
            model_type=model_type,
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
    seed.main(  # TODO: Fix performance issue when called from main.py
        model=model,
        gradients=model_gradients,
        bit_width=seed_precision,
        output_folder=seed_cache_path,
        model_type=model_type,
        cpu_count=cpu_count,
    )
    logging.info("Seed calculation complete.")

    if mode == 'seed':
        return

    # ------------------- Upscale -------------------

    logging.info("------------------- Upscale -------------------")

    parent_cache_path = (f"{cache_dir}/parent/({model_name})-w{parent_precision}_orig{seed_precision}"
                         f"-{dataset}_s{num_examples}_blk{seq_len}")

    upscale.upscale(
        model=model,
        seed_precision=seed_precision,
        parent_precision=parent_precision,
        model_type=model_type,
        seed_parameters_path=seed_cache_path,
        parent_parameters_path=parent_cache_path,
        gradients=model_gradients,
        cpu_count=cpu_count,
    )

    logging.info("Upscale complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model to any precision")
    parser.add_argument("model", type=str, help="The model to quantize")
    parser.add_argument("--seed_precision", type=int, help="The precision to quantize the seed to")
    parser.add_argument("--parent_precision", type=int, help="The precision to quantize the parent to")
    parser.add_argument("--mode", type=str, default="upscale", help="The mode to run in")
    parser.add_argument("--model_type", type=str, default=None, help="The type of model to use")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR, help="The directory to cache results in")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="The dataset to use")
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN, help="The sequence length to use")
    parser.add_argument("--num_examples", type=int, default=DEFAULT_NUM_EXAMPLES, help="The number of examples to use")
    parser.add_argument("--use_cached_gradients", type=bool, default=True, help="Whether to use cached gradients")

    args = parser.parse_args()

    # only pass options that are not None
    quantize_any_precision(**{k: v for k, v in args.__dict__.items() if v is not None})
