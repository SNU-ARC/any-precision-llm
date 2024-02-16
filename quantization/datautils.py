from datasets import load_dataset
import random
import numpy as np
import logging


def get_wikitext2(split):
    assert split in ['train', 'validation', 'test'], f"Unknown split {split} for wikitext2"

    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    return data['text']


def get_ptb(split, slice_unk=True):
    assert split in ['train', 'validation', 'test'], f"Unknown split {split} for ptb"

    data = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    data_list = data['sentence']

    if slice_unk:
        data_list = [s.split() for s in data_list]

    return data_list


def get_c4(split):
    assert split in ['train', 'validation'], f"Unknown split {split} for c4"

    if split == 'train':
        data = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
    else:
        assert split == 'validation'
        data = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )

    return data['text']


def get_pileval(split):
    if split != 'validation':
        logging.warning(f"Pileval only has a validation split, but got split={split}. Using validation split.")
    data = load_dataset("mit-han-lab/pile-val-backup", split="validation")

    return data['text']


def sample_and_tokenize(texts, tokenizer, seq_len, num_samples, seed=0):
    assert num_samples <= len(texts), \
        f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"
    random.seed(seed)
    np.random.seed(seed)

    selected_indices = set()

    samples = []
    for _ in range(num_samples):
        idx = random.randint(0, len(texts) - 1)
        if idx in selected_indices:  # we don't want to sample the same text twice
            continue
        text = texts[idx]

        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
        if len(tokens) < seq_len:  # if the text is too short, we skip it
            continue

        tokens = tokens[:seq_len]

        selected_indices.add(idx)
        samples.append(tokens)

    return samples


def get_dataset(dataset_name, split):
    if dataset_name == 'wikitext2':
        return get_wikitext2(split)
    elif dataset_name == 'ptb':
        return get_ptb(split)
    elif dataset_name == 'c4':
        return get_c4(split)
    elif dataset_name == 'pileval':
        return get_pileval(split)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def get_tokens(dataset_name, split, tokenizer, seq_len, num_samples, seed=0):
    texts = get_dataset(dataset_name, split)
    return sample_and_tokenize(texts, tokenizer, seq_len, num_samples, seed)
