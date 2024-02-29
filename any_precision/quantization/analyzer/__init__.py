from .base import ModelAnalyzer
from .auto import AutoAnalyzer
import logging
import yaml
import os


def get_analyzer(model, model_type=None):
    if model_type is None:
        if model.config.architectures[0] == "LlamaForCausalLM":
            model_type = "llama"
        elif model.config.architectures[0] == "OPTForCausalLM":
            model_type = "opt"
        elif model.config.architectures[0] == "Phi2ForCausalLM":
            model_type = "phi"
        elif model.config.architectures[0] == "MistralForCausalLM":
            model_type = "mistral"
        else:
            logging.warning((f"Attempting to use AutoAnalyzer to quantize unknown model type:"
                             f" {model.config.architectures[0]}"))
            logging.warning("This may not work as expected!")
            return AutoAnalyzer(model)

    try:
        dirpath = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dirpath, f'../../models/{model_type}.yaml')) as f:
            model_config = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Unknown model type: {model_type}. Leave model_type as None to use AutoAnalyzer.")
    return ModelAnalyzer(model, model_config=model_config)
