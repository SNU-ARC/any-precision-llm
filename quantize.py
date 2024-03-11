import argparse
from any_precision.quantization import quantize_any_precision

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model to any precision")
    parser.add_argument("model", type=str, help="The model to quantize")
    parser.add_argument("--seed_precision", type=int, help="The precision to quantize the seed to")
    parser.add_argument("--parent_precision", type=int, help="The precision to quantize the parent to")
    parser.add_argument("--mode", type=str, default="upscale", help="The mode to run in")
    parser.add_argument("--yaml_path", type=str, help="The path to the architecture config yaml file")
    parser.add_argument("--cache_dir", type=str, help="The directory to cache results in")
    parser.add_argument("--dataset", type=str, help="The dataset to use")
    parser.add_argument("--seq_len", type=int, help="The sequence length to use")
    parser.add_argument("--num_examples", type=int, help="The number of examples to use")
    parser.add_argument('--recalculate_gradients', action="store_true",
                        help="Whether to recalculate the gradients")
    parser.add_argument("--recalculate_seed", action="store_true",
                        help="Whether to recalculate the seed")

    args = parser.parse_args()

    # only pass options that are not None
    quantize_any_precision(**{k: v for k, v in args.__dict__.items() if v is not None})
