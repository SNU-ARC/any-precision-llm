""" Utility functions """

import datetime
import os


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


def get_tokenizer_type(model_path):
    if 'llama-2' in model_path.lower():
        tokenizer_type = 'llama-2'
    elif 'llama' in model_path.lower():
        tokenizer_type = 'llama'
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
