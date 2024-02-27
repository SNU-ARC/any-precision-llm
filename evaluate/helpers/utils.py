""" Utility functions """

import datetime
import os
from collections import defaultdict
import pandas as pd
import re
import json


def get_timestamp():
    """ Get the current timestamp for prefixing log entries """
    return datetime.datetime.now().strftime("%H:%M:%S")


def logprint(verbose, *args, **kwargs):
    """ Print if verbose is True, and prefix with timestamp """
    assert isinstance(verbose, bool), "The first argument `verbose` must be a boolean."
    if verbose:
        print(f"[{get_timestamp()}]", end=" ")
        print(*args, **kwargs)


def vprint(verbose, *args, **kwargs):
    """ Print if verbose is True """
    assert isinstance(verbose, bool), "The first argument `verbose` must be a boolean."
    if verbose:
        print(*args, **kwargs)


def get_subdirs(path):
    if not os.path.exists(path):
        return []
    return [os.path.join(path, o) for o in sorted(os.listdir(path))
            if os.path.isdir(os.path.join(path, o))]


def get_files(path):
    if not os.path.exists(path):
        return []
    return [os.path.join(path, o) for o in sorted(os.listdir(path))
            if os.path.isfile(os.path.join(path, o))]

def get_base_models(include_prequant=False, relevant_models_only=False):
    """ Get the repo names of all base models """
    repo_names = [
        'meta-llama/Llama-2-7b-hf',
        'mistralai/Mistral-7B-v0.1',
        'facebook/opt-1.3b',
        'facebook/opt-2.7b',
        'facebook/opt-6.7b',
    ]
    if not relevant_models_only:
        repo_names.append('huggyllama/llama-7b')
        repo_names.append('microsoft/phi-2')

    if include_prequant:
        repo_names += ['TheBloke/Llama-2-7B-AWQ', 'TheBloke/Llama-2-7B-GPTQ', 'TheBloke/Mistral-7B-v0.1-AWQ']
    return repo_names


def base_model_name_to_hf_repo_name(base_model_name):
    """ Convert a base model name to the full HF repository name """
    if base_model_name == 'Llama-2-7b-hf':
        return 'meta-llama/' + base_model_name
    elif base_model_name == 'llama-7b':
        return 'huggyllama/' + base_model_name
    elif 'opt' in base_model_name:
        return 'facebook/' + base_model_name
    elif base_model_name == 'Mistral-7B-v0.1':
        return 'mistralai/' + base_model_name
    elif base_model_name == 'phi-2':
        return 'microsoft/' + base_model_name
    else:
        raise ValueError(f"Unknown base model name {base_model_name}")


def effective_bit_width(alg, precision, **kwargs):
    """ Calculate the effective bit width of different quantization methods """
    if 'fp_precision' in kwargs:
        fp_precision = kwargs['fp_precision']
    else:
        fp_precision = 16
    if alg.lower() in ('awq', 'gptq', 'autoawq', 'scaled awq', 'scaled gptq', 'rtn', 'scaled rtn'):
        assert 'group' in kwargs
        group = kwargs['group']

        zero_point = precision
        scale = fp_precision
        return precision + (scale + zero_point) / group
    elif alg.lower() in ('sqllm', 'squeezellm', 'scaled sqllm'):
        # the group size for sqllm should be the effective group size,
        # i.e. the weighted harmonic mean of the group sizes.
        if 'group' in kwargs:
            group = kwargs['group']
        else:
            group = sqllm_effective_group_size(kwargs['base_model'])
        clusters_per_group = 2 ** precision
        cluster_centers_per_group = fp_precision * clusters_per_group
        return precision + cluster_centers_per_group / group
    else:
        raise ValueError(f"Unknown quantization algorithm: {alg}")


def model_size(base_model_name, alg, precision, **kwargs):
    ebw = effective_bit_width(alg, precision, base_model=base_model_name, **kwargs)
    quantized_weight_count, non_quantized_weight_count = get_weight_counts(base_model_name)

    if 'fp_precision' in kwargs:
        fp_precision = kwargs['fp_precision']
    else:
        fp_precision = 16

    size = quantized_weight_count * ebw / 8 + non_quantized_weight_count * fp_precision / 8
    size_rounded = round(size)
    assert abs(size - size_rounded) < 1e-6, "Result should be an integer"

    return size_rounded


this_file = os.path.abspath(__file__)
local_dir = os.path.dirname(this_file)

weight_count_cache_file = os.path.join(local_dir, 'weight_count_cache.json')


def get_weight_counts(base_model_name):
    from transformers import AutoModelForCausalLM
    import json

    # Check if the weight count is cached
    if os.path.exists(weight_count_cache_file):
        with open(weight_count_cache_file, 'r') as f:
            cache = json.load(f)
        if base_model_name in cache:
            return cache[base_model_name]
    else:
        cache = {}
    
    model_repo = base_model_name_to_hf_repo_name(base_model_name)

    model = AutoModelForCausalLM.from_pretrained(model_repo, trust_remote_code=True)
    quantized_weight_count = 0
    non_quantized_weight_count = 0

    quantized_param_shapes, non_quantized_param_shapes = get_param_shapes(model)

    for shape in quantized_param_shapes:
        x, y = shape
        quantized_weight_count += x * y

    for shape in non_quantized_param_shapes:
        assert len(shape) in (1, 2), f"Unexpected shape: {shape}"
        non_quantized_weight_count += shape[0] if len(shape) == 1 else shape[0] * shape[1]

    # Cache the result and return
    cache[base_model_name] = (quantized_weight_count, non_quantized_weight_count)
    with open(weight_count_cache_file, 'w') as f:
        json.dump(cache, f, indent=4)

    return quantized_weight_count, non_quantized_weight_count


def find_matching_paren(string, start):
    """ Find the matching parenthesis for the parenthesis at index start """
    assert string[start] == '('
    count = 1
    for i in range(start + 1, len(string)):
        if string[i] == '(':
            count += 1
        elif string[i] == ')':
            count -= 1
        if count == 0:
            return i
    return -1


def name_splitter(full_model_name):
    """ Split a model name into its components """
    model_name = full_model_name.split('/')[-1]

    # Find the indices of the separators, skipping over parentheses
    separator_indexes = []
    i = 0
    while i < len(model_name):
        if model_name[i] == '-':
            separator_indexes.append(i)
        elif model_name[i] == '(':
            i = find_matching_paren(model_name, i)
        i += 1

    # Split the model name into its components, based on previously found separators
    fields = []
    start = 0
    for end in separator_indexes:
        fields.append(model_name[start:end])
        start = end + 1
    fields.append(model_name[start:])

    return fields


def model_name_parser(full_model_name):
    """ Parse a model name into its components """

    params = defaultdict(lambda: pd.NA)
    if full_model_name in get_base_models(include_prequant=True):
        params['base_model'] = full_model_name.split('/')[-1]
        return params
    fields = name_splitter(full_model_name)
    assert fields[1].startswith('(') and fields[1].endswith(')')

    if fields[0] == 'autogptq':
        params['Alg'] = 'GPTQ'
    elif fields[0] == 'awq':
        params['Alg'] = 'AWQ'
    elif fields[0] == 'autoawq':
        params['Alg'] = 'AutoAWQ'
    elif fields[0] == 'sqllm':
        params['Alg'] = 'SqLLM'
    elif fields[0] in ('up_sqllm', 'down_sqllm', 'anyprec'):
        params['Alg'] = 'Scaled SqLLM'
    elif fields[0] == 'scaled_awq':
        params['Alg'] = 'Scaled AWQ'
    elif fields[0] == 'scaled_gptq':
        params['Alg'] = 'Scaled GPTQ'
    elif fields[0] == 'rtn':
        params['Alg'] = 'RTN'
    elif fields[0] == 'scaled_rtn':
        params['Alg'] = 'Scaled RTN'
    else:
        raise ValueError(f"Unknown algorithm: {fields[0]}")

    if params['Alg'] in ('AWQ', 'GPTQ', 'AutoAWQ', 'RTN'):
        assert len(fields) >= 4, f"Insufficient fields for {full_model_name} of type {params['Alg']}"
        params['base_model'] = fields[1][1:-1]
        params['precision'] = int(fields[2][1:])
        params['group_size'] = int(fields[3][1:])
        if len(fields) >= 5:
            params['other_settings'] = ', '.join(fields[4:])
    elif params['Alg'] in ('Scaled AWQ', 'Scaled RTN'):
        assert len(fields) >= 4, f"Insufficient fields for {full_model_name} of type {params['Alg']}"
        params['base_model'] = fields[1][1:-1]
        match = re.match(r"w(?P<precision>\d+)_orig(?P<original_precision>\d+)",
                         fields[2])  # w4_orig3
        params['precision'] = int(match.group('precision'))
        params['group_size'] = int(fields[3][1:])
        if len(fields) >= 5:
            params['other_settings'] = ', '.join([f"orig{match.group('original_precision')}"] +
                                                 fields[4:])
    elif params['Alg'] == 'SqLLM':
        assert len(fields) >= 3, f"Insufficient fields for {full_model_name} of type {params['Alg']}"
        params['base_model'] = fields[1][1:-1]
        params['precision'] = int(fields[2][1:])
        # Write to stderr if group size is not specified
        params['group_size'] = sqllm_effective_group_size(params['base_model'])
        if len(fields) >= 4:
            params['other_settings'] = ', '.join(fields[3:])
    elif params['Alg'] == 'Scaled SqLLM':
        assert len(fields) >= 3, f"Insufficient fields for {full_model_name} of type {params['Alg']}"
        params['base_model'] = fields[1][1:-1]
        match = re.match(r"w(?P<precision>\d+)_orig(?P<original_precision>\d+)",
                         fields[2])  # w4_orig3
        params['precision'] = int(match.group('precision'))
        params['group_size'] = sqllm_effective_group_size(params['base_model'])
        params['other_settings'] = ', '.join([f"orig{match.group('original_precision')}"] +
                                             fields[3:])
    elif params['Alg'] == 'Scaled GPTQ':
        assert len(fields) >= 3, f"Insufficient fields for {full_model_name} of type {params['Alg']}"
        params['base_model'] = fields[1][1:-1]
        match = re.match(r"w(?P<precision>\d+)_orig(?P<original_precision>\d+)",
                         fields[2])  # w4_orig3
        params['precision'] = int(match.group('precision'))
        params['group_size'] = int(fields[3][1:])
        params['other_settings'] = ', '.join([f"orig{match.group('original_precision')}"] +
                                             fields[4:])
    else:
        raise RuntimeError(f"Control should not reach here. Unknown algorithm: {params['Alg']}")

    return params


def weighted_harmonic_mean(values, weights):
    """ Calculate the weighted harmonic mean of a list of values """
    assert len(values) == len(weights)
    return sum(weights) / sum(w / v for v, w in zip(values, weights))


sqllm_effective_group_size_cache_file = os.path.join(local_dir,
                                                     'sqllm_effective_group_size_cache.json')


def sqllm_effective_group_size(base_model):
    """ Calculate the effective group size for SqueezeLLM on a specific base model """
    # Just get the base model name, excluding the author
    if '/' in base_model:
        base_model = base_model.split('/')[-1]

    # Check if the result is cached
    if os.path.exists(sqllm_effective_group_size_cache_file):
        with open(sqllm_effective_group_size_cache_file, 'r') as f:
            cache = json.load(f)
        if base_model in cache:
            return cache[base_model]
    else:
        cache = {}

    # Calculate the effective group size
    print(f"[WARNING] Calculating effective group size for SqueezeLLM on {base_model}.")
    print(f"This may take a while, but the result will be cached for future use.")
    from transformers import AutoModelForCausalLM
    base_model_repo = base_model_name_to_hf_repo_name(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model_repo, trust_remote_code=True)

    group_sizes = []
    group_size_weights = []

    quantized_param_shapes, non_quantized_param_shapes = get_param_shapes(model)
    for shape in quantized_param_shapes:
        x, y = shape
        group_sizes.append(y)
        group_size_weights.append(x * y)

    # Calculate the weighted harmonic mean of the group sizes
    effective_group_size = weighted_harmonic_mean(group_sizes, group_size_weights)
    print(f"Effective group size: {effective_group_size}")

    # Cache the result and return
    cache[base_model] = effective_group_size
    with open(sqllm_effective_group_size_cache_file, 'w') as f:
        json.dump(cache, f, indent=4)

    return effective_group_size


def get_layers_name(model):
    if 'llama' in model.__class__.__name__.lower():
        return 'model.layers'
    elif 'opt' in model.__class__.__name__.lower():
        return 'model.decoder.layers'
    elif 'mistral' in model.__class__.__name__.lower():
        return 'model.layers'
    elif 'phi' in model.__class__.__name__.lower():
        return 'model.layers'
    else:
        raise ValueError(f"Unknown model class name: {model.__class__.__name__}")


def get_param_shapes(model):
    """ Get the shapes of all parameters in a model """
    quantized_param_shapes = []
    non_quantized_param_shapes = []

    layers_name = get_layers_name(model)

    for name, param in model.named_parameters():
        if name.startswith(layers_name + '.'):
            # This is a layer parameter
            if len(param.shape) == 2:
                # This is a weight matrix
                quantized_param_shapes.append(param.shape)
            elif len(param.shape) == 1:
                # This is some other layer parameter
                non_quantized_param_shapes.append(param.shape)
            else:
                raise RuntimeError(f"Unexpected parameter shape for {name}: {param.shape}")
        else:
            # This is a non-layer parameter
            non_quantized_param_shapes.append(param.shape)

    return quantized_param_shapes, non_quantized_param_shapes


def get_tokenizer_type(model_path):
    if 'llama-7b' in model_path.lower():
        tokenizer_type = 'llama'
    elif 'llama-2-7b' in model_path.lower():
        tokenizer_type = 'llama-2'
    elif 'opt' in model_path.lower():
        tokenizer_type = 'opt'
    elif 'mistral' in model_path.lower():
        tokenizer_type = 'mistral'
    elif 'phi-2' in model_path.lower():
        tokenizer_type = 'phi-2'
    elif 'gemma' in model_path.lower():
        tokenizer_type = 'gemma'
    else:
        tokenizer_type = None

    return tokenizer_type


def extended_calib_data_for_autogptq(dataset_name, model_path, sample_count, block_size=512):
    """
    Returns a list of examples for calibration.
    Note that the block size has slightly different meanings for different datasets,
    as I am using the original code from the AutoGPTQ/AWQ repos.
    """
    import pickle

    # Check if the examples are cached
    gptq_examples_cache = os.path.join(os.path.dirname(__file__), 'gptq_examples_cache.pkl')
    tokenizer_type = get_tokenizer_type(model_path)
    key = f"{dataset_name}.{tokenizer_type}.{sample_count}.{block_size}"
    if os.path.exists(gptq_examples_cache):
        with open(gptq_examples_cache, 'rb') as f:
            cache = pickle.load(f)
        if key in cache:
            print(f"Found cached examples for {key}")
            return cache[key]
    else:
        cache = {}

    from transformers import AutoTokenizer
    from . import original_datautils
    from . import awq_calib_data
    import torch

    if dataset_name == 'pileval':
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        train_data = awq_calib_data.get_calib_dataset(tokenizer=tokenizer,
                                                      n_samples=sample_count,
                                                      block_size=block_size)
    else:
        train_data = original_datautils.get_loaders(dataset_name,
                                                    nsamples=sample_count,
                                                    seed=0,
                                                    seqlen=block_size,
                                                    model=model_path)[0]
        train_data = [inp for inp, tar in train_data]

    examples = [{"input_ids": inp.squeeze(0),
                 'attention_mask': torch.ones_like(inp.squeeze(0))} for inp in train_data]

    # Cache the result and return
    cache[key] = examples
    with open(gptq_examples_cache, 'wb') as f:
        pickle.dump(cache, f)
    return examples
