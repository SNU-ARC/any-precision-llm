from any_precision.models import AutoAPLoader
from transformers import AutoTokenizer
import logging

# Logging with time sans date, level name, and message
logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s] %(message)s', datefmt='%H:%M:%S')


if __name__ == '__main__':
    model_path = './any_precision/cache/packed/anyprec-(gemma-2b)-w8_orig3-c4_s100_blk512'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoAPLoader.from_quantized(model_path, precisions=[3, 4, 5, 6])
    model = model.eval().cuda()

    input_context = "Large Language Models are"
    input_ids = tokenizer.encode(input_context, return_tensors="pt").cuda()

    for precision in model.precisions:
        print(f"=============== generation with {precision}-bit precision ===============")
        output = model.generate(input_ids, precision=precision, max_length=64, pad_token_id=tokenizer.eos_token_id)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(output_text + '\n')
