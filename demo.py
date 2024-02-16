import transformers
from models.opt import OPTAPForCausalLM

if __name__ == '__main__':

    q_model_path = 'cache/models/opt-1.3b.pt'
    o_model_path = 'facebook/opt-1.3b'
    tokenizer = transformers.AutoTokenizer.from_pretrained(o_model_path)
    config = transformers.AutoConfig.from_pretrained(o_model_path, trust_remote_code=True)
    model = OPTAPForCausalLM.from_quantized(q_model_path, o_model_path,config.max_position_embeddings, supported_bits=[4,8])

    import pdb
    pdb.set_trace()

    input_context = "do you know somewhere to go on weekend?"
    input_ids = tokenizer.encode(input_context, return_tensors="pt")
    #output = model.generate(input_ids.to('cuda'), max_length=256)
    output = model(input_ids.to('cuda'))
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_text)
