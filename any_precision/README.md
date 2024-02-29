# Any-Precision Model Quantization

The folder `quantization` provides the pipeline for quantizing LLMs to Any-Precision.

## Overview
The main script `quantization/main.py` provides the pipeline for quantizing LLMs to Any-Precision. The pipeline includes the following steps:

- Gradient calculation for model analysis.
- Generation of a seed model at a specified precision.
- Upscaling the seed model to a desired precision.
- Packing the upscaled model for deployment.

## Prerequisites

- Python 3.11
- PyTorch
- NumPy
- Any other dependencies listed in `requirements.txt`.

## Installation

1. Clone this repository.
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

To use this tool, run the main script with the required arguments:

```bash
python -m quantization.main <model> [options]
```

### Required Argument

- `model`: The model to quantize.

### Optional Arguments

- `--seed_precision`: The precision to quantize the seed to.
- `--parent_precision`: The precision to quantize the parent to.
- `--mode`: The mode to run in (`gradients`, `seed`, or `upscale`. Use `upscale` for full pipeline.).
- `--model_type`: The type of model to use.
- `--cache_dir`: The directory to cache results in.
- `--dataset`: The dataset to use for gradient calculation.
- `--seq_len`: The sequence length to use for gradient calculation.
- `--num_examples`: The number of examples to use for gradient calculation.
- `--recalculate_gradients`: Flag to recalculate the gradients.
- `--recalculate_seed`: Flag to recalculate the seed.

The default argument for some of the optional arguments are set in `config.py`.

### Example Command

```bash
python -m quantization.main mistralai/Mistral-7B-v0.1 --seed_precision 3 --parent_precision 8
```

Or using the default values:

```bash
python -m quantization.main mistralai/Mistral-7B-v0.1
```

The final packed models will be under `cache/packed` in the project directory.