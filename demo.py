from any_precision.modules import AnyPrecisionForCausalLM
from transformers import AutoTokenizer, TextStreamer
import logging
import time

# Logging with time sans date, level name, and message
logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s] %(message)s', datefmt='%H:%M:%S')


if __name__ == '__main__':
    model_path = './cache/packed/anyprec-(Llama-2-7b-hf)-w8_orig3-gc1-c4_s100_blk512'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    streamer = TextStreamer(tokenizer)

    model = AnyPrecisionForCausalLM.from_quantized(model_path, precisions=[3, 4, 5, 6])
    model = model.eval().cuda()

    input_context = "Large Language Models are"
    input_ids = tokenizer.encode(input_context, return_tensors="pt").cuda()

    for precision in model.precisions:
        print(f"=============== generation with {precision}-bit precision ===============")
        start_time = time.time()
        output = model.generate(input_ids, precision=precision, max_length=64, pad_token_id=tokenizer.eos_token_id,
                                streamer=streamer)
        end_time = time.time()
        # print the generation speed
        token_count = len(output[0])
        print(f"[[ Generation speed: {token_count / (end_time - start_time):.2f} tokens per second ]")
        print(f"[[ Latency per token: {(end_time - start_time) / token_count * 1000:.2f} ms ]]")