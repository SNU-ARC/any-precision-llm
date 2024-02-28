from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM
import torch


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

