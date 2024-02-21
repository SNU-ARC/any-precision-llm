from models.opt import OPTAPForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

if __name__ == '__main__':
    model_path = './cache/packed/anyprec-(opt-1.3b)-w8_orig3-c4_s100_blk512'
    supported_bits = [3, 4]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = OPTAPForCausalLM.from_quantized(model_path, supported_bits=supported_bits)
    model = model.eval().cuda()

    input_context = "Yellow cat"
    input_ids = tokenizer.encode(input_context, return_tensors="pt")


    print("=============== generation with 4 bits ===============")
    model.change_bits(4)
    output = model.generate(input_ids.cuda(), max_length=256)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_text)

    print("=============== generation with 3 bits ===============")
    model.change_bits(3)
    output = model.generate(input_ids.cuda(), max_length=256)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_text)