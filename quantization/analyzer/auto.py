from .base import ModelAnalyzer
import torch


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
        return module_names

    def get_model_name(self):
        return "model"  # TODO: implement this

    def get_layers_name(self):
        # Find attribute that ends with DecoderLayer
        model = self.get_model()
        for name, module in model.named_children():
            if isinstance(module, torch.nn.ModuleList):
                return name
