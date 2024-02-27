from .llama import LlamaAnalyzer
from .opt import OPTAnalyzer
from .phi import Phi2Analyzer
from .mistral import MistralAnalyzer
from .auto import AutoAnalyzer

import logging


def get_analyzer(model, model_type=None):
    if model_type:
        if model_type == 'llama':
            return LlamaAnalyzer(model)
        elif model_type == 'opt':
            return OPTAnalyzer(model)
        elif model_type == 'phi':
            return Phi2Analyzer(model)
        elif model_type == 'mistral':
            return MistralAnalyzer(model)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Leave model_type as None to use AutoAnalyzer.")
    else:
        if model.config.architectures[0] == "LlamaForCausalLM":
            return LlamaAnalyzer(model)
        elif model.config.architectures[0] == "OPTForCausalLM":
            return OPTAnalyzer(model)
        elif model.config.architectures[0] == "Phi2ForCausalLM":
            return Phi2Analyzer(model)
        elif model.config.architectures[0] == "MistralForCausalLM":
            return MistralAnalyzer(model)
        else:
            logging.warning((f"Attempting to use AutoAnalyzer to quantize unknown model type:"
                             f" {model.config.architectures[0]}"))
            logging.warning("This may not work as expected!")
            return AutoAnalyzer(model)

