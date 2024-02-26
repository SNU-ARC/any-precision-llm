from models.auto import AutoAPLoader
from transformers import AutoTokenizer
import logging

# Logging with time sans date, level name, and message
logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s] %(message)s', datefmt='%H:%M:%S')


if __name__ == '__main__':
    model_path = './cache/packed/anyprec-(opt-1.3b)-w8_orig3-c4_s100_blk512'
    model_path = './cache/packed/anyprec-(phi-2)-w8_orig3-c4_s100_blk512'
    supported_bits = [3, 4, 5, 6, 7, 8]


    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoAPLoader.from_quantized(model_path, supported_bits=supported_bits)
    model = model.eval().cuda()

    input_context = "Explain what Large Language Models are and how they work."
    input_ids = tokenizer.encode(input_context, return_tensors="pt").cuda()

    for bit in supported_bits:
        print(f"=============== generation with {bit} bits ===============")
        model.change_bits(bit)
        output = model.generate(input_ids, max_length=64, pad_token_id=tokenizer.eos_token_id)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(output_text)
