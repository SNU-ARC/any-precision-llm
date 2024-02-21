from models.opt import OPTAPForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

if __name__ == '__main__':
    o_model_path = 'facebook/opt-1.3b'
    q_model_path = './cache/packed/anyprec-(opt-1.3b)-w8_orig3-c4_s100_blk512.pt'
    supported_bits = [3, 4]

    tokenizer = AutoTokenizer.from_pretrained(o_model_path)
    config = AutoConfig.from_pretrained(o_model_path, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(o_model_path)
    model = OPTAPForCausalLM.from_quantized(q_model_path, o_model_path, config.max_position_embeddings, supported_bits=supported_bits)
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