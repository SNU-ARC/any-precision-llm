import torch
from any_precision.modules import AnyPrecisionForCausalLM
from transformers import AutoTokenizer, TextStreamer
import logging
import time

# Logging with time sans date, level name, and message
logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s] %(message)s', datefmt='%H:%M:%S')

if __name__ == '__main__':
    model_path = './cache/packed/anyprec-(Llama-2-7b-hf)-w8_orig3-gc1-c4_s100_blk512'

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    streamer = TextStreamer(tokenizer)

    model = AnyPrecisionForCausalLM.from_quantized(model_path)
    model = model.eval().cuda()

    # Warm up CUDA cache for stable performance
    print("~~~~~~~ Warming up CUDA cache ~~~~~~~")
    input_context = "Models can run slow on first run because"
    input_ids = tokenizer.encode(input_context, return_tensors="pt").cuda()
    output = model.generate(input_ids, precision=3, max_length=64, pad_token_id=tokenizer.eos_token_id,
                            streamer=streamer)
    print("~~~~~~~ Warm up complete ~~~~~~~\n")

    # Now begin bit-width benchmarking
    input_context = "Large Language Models are"
    input_ids = tokenizer.encode(input_context, return_tensors="pt").cuda()

    results = {}
    for precision in model.precisions:
        print(f"=============== generation with {precision}-bit precision ===============")
        torch.cuda.synchronize()
        start_time = time.time()
        output = model.generate(input_ids, precision=precision, max_length=128, pad_token_id=tokenizer.eos_token_id,
                                streamer=streamer)
        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate generation speed
        token_count = len(output[0])
        tokens_per_second = token_count / (end_time - start_time)
        ms_per_token = 1 / tokens_per_second * 1000

        results[precision] = (tokens_per_second, ms_per_token)

        print(f"\n( Generation speed: {tokens_per_second:.1f} tok/s | Latency: {ms_per_token:.2f} ms/tok )\n")

    print("=============== Summary ===============")
    print(f"\nModel: {model_path}\n")

    for precision, (tokens_per_second, ms_per_token) in results.items():
        print(f"{precision}-bit: {tokens_per_second:.1f} tok/s | {ms_per_token:.2f} ms/tok")
