from helpers import dataloader
from tqdm import tqdm
import torch
from helpers.utils import vprint, logprint, model_name_parser, base_model_name_to_hf_repo_name, get_tokenizer_type
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
from models import AutoAPLoader


# from algorithms.SqueezeLLM.llama import load_quant as sqllm_load_quant


@torch.no_grad()
def auto_model_load(model_path, device_map='auto', dtype=torch.float16, verbose=True):
    """
    Args:
        model_path: path of the model to evaluate
        device_map: device to use for evaluation
        dtype: the dtype to use for evaluation, either torch.float16 or torch.float32
        verbose: whether to print progress

    Returns:
        (tokenizer, model) tuple loaded from the given path, with the given device and dtype.
    """
    logprint(verbose, "Loading tokenizer and model...")

    if 'anyprec-' in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoAPLoader.from_quantized(model_path).cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype,
                                                     trust_remote_code=True).cuda()

    # logprint(verbose, f"{model.__class__.__name__} model loaded to device: {model.device}")

    tokenizer_type = get_tokenizer_type(model_path)

    if tokenizer_type is None:
        logprint(verbose, f"Unknown tokenizer type for {model_path}. Cannot use cached input tokens.")

    return tokenizer_type, tokenizer, model


@torch.no_grad()
def evaluate_ppl(model, tokenizer, testcases, verbose=True, chunk_size=2048, tokenizer_type=None):
    """
    Args:
        model: model to evaluate
        tokenizer: tokenizer to use
        testcases: testcases names to evaluate on, passed on to dataloader.get_loaders
        verbose: whether to print progress
        chunk_size: the size of the chunks into which the test set is split
        tokenizer_type: set to llama, llama-2, or opt to use cached input tokens
                        for the corresponding test set

    Returns:
        A dictionary of perplexity scores, with keys being the testcases names and values being the perplexity scores.

    Note that the perplexity scores are calculated over non-overlapping chunks of the test set.
    """

    if hasattr(model, 'set_precision'):
        is_anyprec = True
    else:
        is_anyprec = False

    model.eval()

    results = {}

    supported_bits = model.precisions if is_anyprec else [None]

    for bit in supported_bits:
        if is_anyprec:
            logprint(verbose, f"Setting model precision to {bit}-bit...")
            model.set_precision(bit)

        for testcase_name in testcases:
            vprint(verbose, f"---------------------- {testcase_name} ----------------------")

            input_tokens = load_input_tokens(tokenizer_type, testcase_name, tokenizer, verbose)

            # input_tokens.to(model.device)
            input_tokens.to('cuda')

            logprint(verbose, "Calculating perplexity...")

            seq_len = input_tokens.input_ids.size(1)
            nsamples = seq_len // chunk_size  # floor(seq_len / chunk_size)

            neg_log_likelihoods = []
            for i in tqdm(range(nsamples), disable=not verbose):
                begin_loc = i * chunk_size

                input_ids = input_tokens.input_ids[:, begin_loc:begin_loc + chunk_size]

                # add BOS token, necessary for Gemma-7B, harmless for other models
                # https://github.com/huggingface/transformers/issues/29250
                input_ids[:, 0] = tokenizer.bos_token_id

                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                    neg_log_likelihood = outputs.loss
                    neg_log_likelihoods.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(neg_log_likelihoods).mean())
            logprint(verbose, f"Perplexity: {ppl.item()}")

            results[f"{bit}-bit:{testcase_name}"] = ppl.item()

        if not is_anyprec:
            break

    return results


def load_input_tokens(tokenizer_type, testcase_name, tokenizer, verbose):
    """ Load input tokens from cache if available, otherwise load from dataloader and save to cache. """
    input_tokens_cache_path = f"input_tokens_cache/dataloader-{tokenizer_type}-{testcase_name}-test.pt"
    if tokenizer_type and os.path.exists(input_tokens_cache_path):
        logprint(verbose, f"Loading cached input tokens from {input_tokens_cache_path}...")
        input_tokens = torch.load(input_tokens_cache_path)
    else:
        logprint(verbose, "Loading test set...")

        raw_text = dataloader.get_loaders(testcase_name)

        logprint(verbose, "Tokenizing test set...")

        input_tokens = tokenizer(raw_text, return_tensors='pt')
        # save input_tokens to cache
        if tokenizer_type:
            logprint(verbose, f"Caching input tokens to {input_tokens_cache_path}...")
            # we must create the directory if it doesn't exist
            os.makedirs(os.path.dirname(input_tokens_cache_path), exist_ok=True)
            torch.save(input_tokens, input_tokens_cache_path)

    return input_tokens
