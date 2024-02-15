import os.path

from transformers import AutoModelForCausalLM
import torch
import argparse
import logging

from config import *
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
                           cache_gradients=True, cache_seed=True,
                           use_cached_gradients=True, use_cached_seed=True):
    assert mode in ['gradients', 'seed', 'upscale'], \
        "mode must be one of 'gradients', 'seed', or 'upscale'. Use 'upscale' to run the entire pipeline."

    if mode == 'gradients':
        assert cache_gradients, "There is no point in stopping at gradient calculation if the gradients are not cached."
        logging.info("Running: [Gradients]")
    elif mode == 'seed':
        assert cache_seed, "There is no point in stopping at seed quantization if the seed is not cached."
        logging.info("Running: [Gradients -> Seed]")
    else:
        logging.info("Running: [Gradients -> Seed -> Upscale]")

    model_string = model if isinstance(model, str) else model.name_or_path
    model_name = model_string.split("/")[-1]

    if isinstance(model, str):
        model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
    else:
        assert isinstance(model, AutoModelForCausalLM), "model must be a string or a transformers model"

    gradients_cache_path = f"{cache_dir}/gradients/({model_name})-{dataset}-s{num_examples}-blk{seq_len}.pt"

    # Calculate or load gradients
    if use_cached_gradients and os.path.exists(gradients_cache_path):
        model_gradients = torch.load(gradients_cache_path)
        logging.info(f"Loaded cached gradients from {gradients_cache_path}")
    else:
        logging.info("Beginning gradient calculation...")
        save_path = gradients_cache_path if cache_gradients else None  # Disable caching if cache_gradients is False
        model_gradients = gradients.get_gradients(model=model,
                                                  dataset=dataset,
                                                  seq_len=seq_len,
                                                  num_examples=num_examples,
                                                  model_type=model_type,
                                                  save_path=save_path)
        logging.info("Gradient calculation complete.")
        if save_path is not None:
            logging.info(f"Saved gradients to {save_path}")


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
    parser.add_argument("--cache_gradients", type=bool, default=True, help="Whether to cache gradients")
    parser.add_argument("--cache_seed", type=bool, default=True, help="Whether to cache the seed")
    parser.add_argument("--use_cached_gradients", type=bool, default=True, help="Whether to use cached gradients")
    parser.add_argument("--use_cached_seed", type=bool, default=True, help="Whether to use cached seed")

    args = parser.parse_args()

    quantize_any_precision(model=args.model,
                           seed_precision=args.seed_precision,
                           parent_precision=args.parent_precision,
                           mode=args.mode,
                           model_type=args.model_type,
                           cache_dir=args.cache_dir,
                           dataset=args.dataset,
                           seq_len=args.seq_len,
                           num_examples=args.num_examples,
                           cache_gradients=args.cache_gradients,
                           cache_seed=args.cache_seed,
                           use_cached_gradients=args.use_cached_gradients,
                           use_cached_seed=args.use_cached_seed)
