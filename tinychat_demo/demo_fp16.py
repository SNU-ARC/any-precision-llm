from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time
import torch

import numpy as np
from attributedict.collections import AttributeDict

from prompt_templates import get_prompter, get_stop_token_ids
from stream_gen import StreamGenerator


gen_params = AttributeDict(
    [
        ("seed", -1),  # RNG seed
        ("n_threads", 1),  # TODO: fix this
        ("n_predict", 512),  # new tokens to predict
        ("n_parts", -1),  # amount of model parts (-1: determine from model dimensions)
        ("n_ctx", 512),  # context size
        ("n_batch", 512),  # batch size for prompt processing (must be >=32 to use BLAS)
        ("n_keep", 0),  # number of tokens to keep from initial prompt
        ("n_vocab", 50272),  # vocabulary size
        # sampling parameters
        ("logit_bias", dict()),  # logit bias for specific tokens: <int, float>
        ("top_k", 40),  # <= 0 to use vocab size
        ("top_p", 0.95),  # 1.0 = disabled
        ("tfs_z", 1.00),  # 1.0 = disabled
        ("typical_p", 1.00),  # 1.0 = disabled
        ("temp", 0.70),  # 1.0 = disabled
        ("repeat_penalty", 1.10),  # 1.0 = disabled
        (
            "repeat_last_n",
            64,
        ),  # last n tokens to penalize (0 = disable penalty, -1 = context size)
        ("frequency_penalty", 0.00),  # 0.0 = disabled
        ("presence_penalty", 0.00),  # 0.0 = disabled
        ("mirostat", 0),  # 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
        ("mirostat_tau", 5.00),  # target entropy
        ("mirostat_eta", 0.10),  # learning rate
    ]
)

# Logging with time sans date, level name, and message
logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s] %(message)s', datefmt='%H:%M:%S')

def stream_output(output_stream):
    print(f"ASSISTANT: ", end="", flush=True)
    pre = 0
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            print(" ".join(output_text[pre:now]), end=" ", flush=True)
            pre = now
    print(" ".join(output_text[pre:]), flush=True)
    if "timing" in outputs and outputs["timing"] is not None:
        timing = outputs["timing"]
        context_tokens = timing["context_tokens"]
        context_time = timing["context_time"]
        total_tokens = timing["total_tokens"]
        generation_time_list = timing["generation_time_list"]
        generation_tokens = len(generation_time_list)
        average_speed = (context_time + np.sum(generation_time_list)) / (
            context_tokens + generation_tokens
        )
        print("=" * 50)
        print("Speed of Inference")
        print("-" * 50)
        print(
            f"Generation Stage : {np.average(generation_time_list) * 1000:.2f} ms/token"
        )
        print("=" * 50)
    return " ".join(output_text)

if __name__ == '__main__':
    model_path = 'meta-llama/Llama-2-7b-hf'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model = model.eval().cuda()

    model_type = 'llama'
    model_prompter = get_prompter(model_type, model_path, empty_prompt=True)
    stop_token_ids = get_stop_token_ids(model_type, model_path)

    count = 0
    while True:
        input_prompt = "Large language models are "
        if input_prompt == "":
            print("EXIT...")
            break
        model_prompter.insert_prompt(input_prompt)
        output_stream = StreamGenerator(
            model,
            tokenizer,
            model_prompter.model_input,
            gen_params,
            device='cuda',
            stop_token_ids=stop_token_ids,
        )
        outputs = stream_output(output_stream)
        count += 1
        break

