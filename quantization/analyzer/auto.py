from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM


class ModelAnalyzer(ABC):
    def __init__(self, model: AutoModelForCausalLM):
        self.model = model

    def get_layers(self):
        layers_name = self.get_layers_name()
        layers = [getattr(self.get_model(), layer_name) for layer_name in layers_name]
        return layers

    def get_modules(self, layer):
        module_names = self.get_module_names()
        modules = [getattr(layer, module_name) for module_name in module_names]
        return modules

    def get_model_weights(self):
        layers = self.get_layers()
        model_layers = []
        module_names = self.get_module_names()

        for layer in layers:
            layer_data = {}
            modules = self.get_modules(layer)

            assert len(modules) == len(module_names), \
                "number of modules and module names don't match: {} vs {}".format(len(modules), len(module_names))

            for module, name in zip(modules, module_names):
                layer_data[name] = module.weight.data.cpu()

            model_layers.append(layer_data)

        return model_layers

    @abstractmethod
    def get_module_names(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_layers_name(self):
        pass


class LlamaAnalyzer(ModelAnalyzer):
    def get_module_names(self):
        return [
            'self_attn.q_proj',
            'self_attn.k_proj',
            'self_attn.v_proj',
            'self_attn.o_proj',
            'mlp.gate_proj',
            'mlp.up_proj',
            'mlp.down_proj',
        ]

    def get_model(self):
        return self.model.model

    def get_layers_name(self):
        return "model.layers"


class OPTAnalyzer(ModelAnalyzer):
    def get_module_names(self):
        return [
            'self_attn.q_proj',
            'self_attn.k_proj',
            'self_attn.v_proj',
            'self_attn.o_proj',
            'fc1',
            'fc2',
        ]

    def get_model(self):
        return self.model.model.decoder

    def get_layers_name(self):
        return "model.decoder.layers"


class Phi2Analyzer(ModelAnalyzer):
    def get_module_names(self):
        return [
            'self_attn.q_proj',
            'self_attn.k_proj',
            'self_attn.v_proj',
            'self_attn.o_proj',
            'self_attn.dense',
            'mlp.fc1',
            'mlp.fc2',
        ]

    def get_model(self):
        return self.model.model

    def get_layers_name(self):
        return "model.layers"
