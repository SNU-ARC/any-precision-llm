import torch
from any_precision.modules import AnyPrecisionForCausalLM
from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM
import logging
import time

# Logging with time sans date, level name, and message
logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s] %(message)s', datefmt='%H:%M:%S')

if __name__ == '__main__':
    model_path = './cache/packed/anyprec-(Llama-2-7b-hf)-w8_orig3-gc1-c4_s100_blk512'
    original_model_path = 'meta-llama/Llama-2-7b-hf'

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    streamer = TextStreamer(tokenizer)

    model = AnyPrecisionForCausalLM.from_quantized(model_path)
    model = model.eval().cuda()

    # Warm up CUDA cache for stable performance
    print("~~~~~~~ Warming up CUDA cache ~~~~~~~")
    input_context = "A CUDA cache warm-up is needed to"
    input_ids = tokenizer.encode(input_context, return_tensors="pt").cuda()
    output = model.generate(
        input_ids,
        precision=min(model.precisions),
        max_new_tokens=32,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    print("~~~~~~~ Warm up complete ~~~~~~~\n")

    # Now begin bit-width benchmarking
    input_context = input("Prompt/Context: ")
    input_ids = tokenizer.encode(input_context, return_tensors="pt").cuda()

    results = {}
    for precision in model.precisions:
        print(f"=============== generation with {precision}-bit precision ===============")
        torch.cuda.synchronize()
        start_time = time.time()
        output = model.generate(
            input_ids,
            precision=precision,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate generation speed
        token_count = len(output[0]) - len(input_ids[0])
        tokens_per_second = token_count / (end_time - start_time)
        ms_per_token = 1 / tokens_per_second * 1000

        results[f"{precision}-bit"] = (tokens_per_second, ms_per_token)

        print(f"\n( Generation speed: {tokens_per_second:.1f} tok/s | Latency: {ms_per_token:.2f} ms/tok )\n")

    # Clear memory
    del model
    torch.cuda.empty_cache()

    # Benchmark the original model
    print(f"=============== generation with fp16 precision ===============")
    model = AutoModelForCausalLM.from_pretrained(original_model_path, torch_dtype=torch.float16).eval().cuda()
    torch.cuda.synchronize()
    start_time = time.time()
    output = model.generate(
        input_ids,
        max_length=128,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate generation speed
    token_count = len(output[0]) - len(input_ids[0])
    tokens_per_second = token_count / (end_time - start_time)
    ms_per_token = 1 / tokens_per_second * 1000

    results["fp16"] = (tokens_per_second, ms_per_token)

    print(f"\n( Generation speed: {tokens_per_second:.1f} tok/s | Latency: {ms_per_token:.2f} ms/tok )\n")

    print("=============== Summary ===============")
    print(f"\nModel: {model_path}\n")

    for precision, (tokens_per_second, ms_per_token) in results.items():
        print(f"{precision}: {tokens_per_second:.1f} tok/s | {ms_per_token:.2f} ms/tok")
