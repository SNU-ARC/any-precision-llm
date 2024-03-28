from transformers import AutoModelForCausalLM, PreTrainedModel
import torch
import yaml
import os
import logging
from .utils import load_model, load_tokenizer


def get_analyzer(model, yaml_path=None, include_tokenizer=False):
    # Load model from string if necessary
    model = load_model(model)

    # Anyprecision quantized model
    if hasattr(model.config, 'anyprec'):
        return ModelAnalyzer.from_arch_config(model, model.config.anyprec['arch_config'])

    # Unspecified model quantization config
    if yaml_path is None:
        dirpath = os.path.dirname(os.path.realpath(__file__))
        yaml_dir = os.path.join(dirpath, f'./architectures/')
        assert len(model.config.architectures) == 1, "Model has multiple architectures"
        # Check if there is a yaml file for the model architecture
        for file in os.listdir(yaml_dir):
            if file.endswith(".yaml"):
                with open(os.path.join(yaml_dir, file)) as f:
                    yaml_contents = yaml.safe_load(f)
                if model.config.architectures[0] == yaml_contents['architecture']:
                    return ModelAnalyzer.from_arch_config(model, yaml_contents['arch_config'], include_tokenizer)
        else:
            # If no yaml file is found, use AutoQuantConfig
            logging.warning((f"Attempting to use AutoArchConfig for architecture:"
                             f" {model.config.architectures[0]}"))
            logging.warning("This may not work as expected!")
            return ModelAnalyzer.from_autoconfig(model, include_tokenizer=include_tokenizer)

    # Specified model quantization config
    else:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Specified yaml file does not exist: {yaml_path}")
        with open(yaml_path) as f:
            quant_config = yaml.safe_load(f)
        return ModelAnalyzer.from_arch_config(model, quant_config, include_tokenizer=include_tokenizer)


class ModelAnalyzer:
    """ModelAnalyzer is a class that provides an interface to access relevant model information for quantization.

    This class is intended to work for any model type, and the model-specific information should be passed in the
    constructor. Alternatively, you can instantiate from a yaml file using the from_yaml method.
    """

    def __init__(self, model: AutoModelForCausalLM, module_names, model_name, layers_name, include_tokenizer=False):
        self.model = model
        self.module_names = module_names
        self.model_name = model_name
        self.layers_name = layers_name
        self.config = model.config
        self.state_dict = model.state_dict()
        self.dropped_original_weights = False
        self.num_layers = len(self.get_layers())
        self.tokenizer = None
        if include_tokenizer:
            self.tokenizer = load_tokenizer(model)
        self._model_weights = {}

    @classmethod
    def from_arch_config(cls, model: AutoModelForCausalLM, quant_config: dict, include_tokenizer=False):
        return cls(model, **quant_config, include_tokenizer=include_tokenizer)

    def get_arch_config(self):
        quant_config = {
            "module_names": self.module_names,
            "model_name": self.model_name,
            "layers_name": self.layers_name,
        }
        return quant_config

    @classmethod
    def from_autoconfig(cls, model: AutoModelForCausalLM, include_tokenizer=False):
        """Instantiate a ModelAnalyzer from an AutoConfig."""
        auto_config = AutoArchConfig(model)
        return cls(model, **auto_config.to_dict(), include_tokenizer=include_tokenizer)

    def get_layers(self):
        """Return the layers of the model."""
        if self.dropped_original_weights:
            raise ValueError("Original weights have been dropped")
        module = self.get_model()
        for attrib_name in self.layers_name.split('.'):
            module = getattr(module, attrib_name)
        return module

    def get_modules(self, layer):
        """Return the relevant modules of the layer."""
        modules = {}
        for module_name in self.module_names:
            module = layer
            for attrib_name in module_name.split('.'):
                module = getattr(module, attrib_name)
            modules[module_name] = module
        return modules

    def get_layer_weights(self, layer_idx):
        """Return the relevant weights of the model."""
        if self.dropped_original_weights:
            raise ValueError("Original weights have been dropped")
        if layer_idx in self._model_weights:
            return self._model_weights[layer_idx]
        layers = self.get_layers()
        layer_data = {}
        modules = self.get_modules(layers[layer_idx])
        for name, module in modules.items():
            layer_data[name] = module.weight.data.cpu()
        self._model_weights[layer_idx] = layer_data
        return layer_data

    def get_model(self):
        """Return the model."""
        if self.dropped_original_weights:
            raise ValueError("Original weights have been dropped")
        module = self.model
        for attrib_name in self.model_name.split('.'):
            module = getattr(module, attrib_name)
        return module

    def drop_original_weights(self):
        weight_key_prefixes = [f'{self.model_name}.{self.layers_name}.{i}' for i in range(self.num_layers)]
        weight_key_postfix = 'weight'
        for prefix in weight_key_prefixes:
            for module_name in self.module_names:
                key = f"{prefix}.{module_name}.{weight_key_postfix}"
                self.state_dict.pop(key)

        self.model = None
        self._model_weights.clear()
        self.dropped_original_weights = True


class AutoArchConfig:
    def __init__(self, model):
        self.model = model

    def to_dict(self):
        return {
            "module_names": self.get_module_names(),
            "model_name": self.get_model()[0],
            "layers_name": self.get_layers()[0],
        }

    def get_module_names(self):
        layers_name, layers = self.get_layers()
        first_layer = next(layers.children())
        # find all linear layers
        module_names = []
        for name, module in first_layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                module_names.append(name)
        return module_names

    def get_model(self):
        for name, module in self.model.named_modules():
            if module is not self.model and isinstance(module, PreTrainedModel):
                return name, module
        else:
            raise ValueError("Model not found")

    def get_layers(self):
        model_name, model = self.get_model()
        for name, module in model.named_children():
            if isinstance(module, torch.nn.ModuleList):
                return name, module
        else:
            raise ValueError("Model layers not found")
