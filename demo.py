from models.opt import OPTAPForCausalLM
from transformers import AutoTokenizer

if __name__ == '__main__':
    model_path = './cache/packed/anyprec-(opt-1.3b)-w8_orig3-c4_s100_blk512'
    supported_bits = [3, 4, 5, 6, 7, 8]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = OPTAPForCausalLM.from_quantized(model_path, supported_bits=supported_bits)
    model = model.eval().cuda()

    input_context = "What's the Large Language Model in Natural Language Processing?"
    input_ids = tokenizer.encode(input_context, return_tensors="pt").cuda()

    for bit in supported_bits:
        print(f"=============== generation with {bit} bits ===============")
        model.change_bits(bit)
        output = model.generate(input_ids, max_length=64)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(output_text)
