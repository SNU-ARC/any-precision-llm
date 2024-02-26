from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM
import torch
import logging

from llama import LlamaAnalyzer
from opt import OPTAnalyzer
from phi import Phi2Analyzer
from mistral import MistralAnalyzer


class ModelAnalyzer(ABC):
    def __init__(self, model: AutoModelForCausalLM):
        self.model = model

    def get_layers(self):
        layers_name = self.get_layers_name()
        module = self.get_model()
        for attrib_name in layers_name.split('.'):
            module = getattr(module, attrib_name)
        return module

    def get_modules(self, layer):
        module_names = self.get_module_names()
        modules = {}
        for module_name in module_names:
            module = layer
            for attrib_name in module_name.split('.'):
                module = getattr(module, attrib_name)
            modules[module_name] = module
        return modules

    def get_model_weights(self):
        layers = self.get_layers()
        model_layers = []
        for layer in layers:
            layer_data = {}
            modules = self.get_modules(layer)
            for name, module in modules.items():
                layer_data[name] = module.weight.data.cpu()
            model_layers.append(layer_data)
        return model_layers

    def get_model(self):
        model_name = self.get_model_name()
        module = self.model
        for attrib_name in model_name.split('.'):
            module = getattr(module, attrib_name)
        return module

    @property
    @abstractmethod
    def model_type(self):
        pass

    @abstractmethod
    def get_module_names(self):
        pass

    @abstractmethod
    def get_model_name(self):
        pass

    @abstractmethod
    def get_layers_name(self):
        pass


def get_analyzer(model):
    if model.config.architectures[0] == "LlamaForCausalLM":
        return LlamaAnalyzer(model)
    elif model.config.architectures[0] == "OPTForCausalLM":
        return OPTAnalyzer(model)
    elif model.config.architectures[0] == "Phi2ForCausalLM":
        return Phi2Analyzer(model)
    elif model.config.architectures[0] == "MistralForCausalLM":
        return MistralAnalyzer(model)
    else:
        logging.warning((f"Attempting to use AutoAnalyzer to quantize unknown model type:",
                         f" {model.config.architectures[0]}"))
        logging.warning("This may not work as expected!")
        return AutoAnalyzer(model)


class AutoAnalyzer(ModelAnalyzer):
    @property
    def model_type(self):
        return "auto"

    def get_module_names(self):
        layers = self.get_layers()
        # find all linear layers
        module_names = []
        for name, module in layers.named_modules():
            if isinstance(module, torch.nn.Linear):
                module_names.append(name)

    def get_model_name(self):
        return "model"  # TODO: implement this

    def get_layers_name(self):
        # Find attribute that ends with DecoderLayer
        for name, module in self.get_layers():
            if name.endswith("DecoderLayer"):
                return name

