from transformers import AutoTokenizer
from models.llama import LlamaAPForCausalLM

if __name__ == '__main__':

    model_path = 'cache/models/opt-1.3b.pt'
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaAPForCausalLM.from_quantized(model_path, supported_bits=[4,8])

