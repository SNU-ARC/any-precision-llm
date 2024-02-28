import transformers
import torch
from tqdm import tqdm
import os
import utils
import argparse
import logging
from config import *
from analyzer import get_analyzer

import datautils


def get_gradients(model,
                  tokenizer,
                  dataset=DEFAULT_DATASET,
                  seq_len=DEFAULT_SEQ_LEN,
                  num_examples=DEFAULT_NUM_EXAMPLES,
                  analyzer=None,
                  save_path=None):
    logging.info(f"Calculating gradients on dataset {dataset} with sequence length {seq_len} and "
                 f"{num_examples} examples...")
    logging.info(f"Fetching {dataset} dataset...")

    model = utils.load_model(model)
    tokenizer = utils.load_tokenizer(tokenizer)

    input_tokens = datautils.get_tokens(dataset, 'train', tokenizer, seq_len, num_examples)

    if analyzer is None:
        analyzer = get_analyzer(model)

    model = model.bfloat16()
    model.eval()
    model.cuda()

    layers = analyzer.get_layers()

    # Register hook to store the square of the gradients
    def square_grad_hook(grad):
        return grad.pow(2)

    for layer in layers:
        for module in analyzer.get_modules(layer).values():
            module.weight.register_hook(square_grad_hook)

    # Calculate gradients through loss.backward()
    for tokens in tqdm(input_tokens, desc="Calculating gradients"):
        tokens = tokens.cuda()
        tokens = tokens.unsqueeze(0)
        outputs = model(input_ids=tokens, labels=tokens)
        loss = outputs.loss
        loss.backward()

    # Harvest the gradients
    gradients = []
    for layer in layers:
        gradients_per_layer = {}
        for module_name, module in analyzer.get_modules(layer).items():
            gradients_per_layer[module_name] = module.weight.grad.cpu()
        gradients.append(gradients_per_layer)

    # Save the gradients to file
    if save_path is not None:
        logging.info(f"Saving gradients to {save_path}...")
        # add file extension if not present
        if not save_path.endswith('.pt'):
            save_path = save_path + '.pt'
        # check if the file already exists
        if os.path.exists(save_path):
            input(f"[WARNING] File {save_path} already exists. Press enter to overwrite or Ctrl+C to cancel.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(gradients, save_path)

    return gradients


if __name__ == "__main__":
    default_output_dir = '../cache/gradients'

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to use for gradient calculation")
    parser.add_argument("--model_name_or_path", type=str, help="Model to use for gradient calculation")
    parser.add_argument("--seq_len", type=int, default=None, help="Sequence length to use for gradient calculation")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Number of examples to use for gradient calculation")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Output directory for gradients")
    parser.add_argument("--model_type", type=str, default=None, help="Model type to use for gradient calculation")
    args = parser.parse_args()
    get_gradients(args.model_name_or_path,
                  args.model_type,
                  args.dataset,
                  args.seq_len,
                  args.num_examples,
                  args.output_dir)
