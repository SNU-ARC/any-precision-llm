from transformers import AutoConfig
from .base import BaseAPForCausalLM
from .auto import AutoAPForCausalLM
import logging
import yaml
import os


class AutoAPLoader:
    @staticmethod
    def from_quantized(
            quant_model_path,
            *args,
            **kwargs
    ):
        config = AutoConfig.from_pretrained(quant_model_path)
        if config.anyprec_model_type == "auto":
            logging.warning("Loading auto-quantized model, will try use AutoAPForCausalLM")
            return AutoAPForCausalLM.from_quantized(quant_model_path, *args, **kwargs)
        else:
            try:
                dirpath = os.path.dirname(os.path.realpath(__file__))
                with open(os.path.join(dirpath, f'../configs/{config.anyprec_model_type}.yaml')) as f:
                    model_config = yaml.safe_load(f)
            except FileNotFoundError:
                raise ValueError(f"Unsupported model type: {config.anyprec_model_type}")
            except:
                raise ValueError(f"Failed to load model config: {config.anyprec_model_type}")
            return BaseAPForCausalLM.from_quantized(quant_model_path, model_config, *args, **kwargs)
