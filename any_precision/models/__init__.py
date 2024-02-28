from transformers import AutoConfig
from .llama import LlamaAPForCausalLM
from .mistral import MistralAPForCausalLM
from .opt import OPTAPForCausalLM
from .auto import AutoAPForCausalLM
import logging


class AutoAPLoader:
    @staticmethod
    def from_quantized(
            quant_model_path,
            *args,
            **kwargs
    ):
        config = AutoConfig.from_pretrained(quant_model_path)
        if config.anyprec_model_type == "llama":
            ap_class = LlamaAPForCausalLM
        elif config.anyprec_model_type == "mistral":
            ap_class = MistralAPForCausalLM
        elif config.anyprec_model_type == "opt":
            ap_class = OPTAPForCausalLM
        elif config.anyprec_model_type == "auto":
            logging.warning("Loading auto-quantized model, will try use AutoAPForCausalLM")
            ap_class = AutoAPForCausalLM
        else:
            raise ValueError(f"Unsupported model type: {config.anyprec_model_type}")

        return ap_class.from_quantized(quant_model_path, *args, **kwargs)
