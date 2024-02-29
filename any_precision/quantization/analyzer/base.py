from transformers import AutoModelForCausalLM


class ModelAnalyzer():
    def __init__(self, model: AutoModelForCausalLM, model_config=None):
        self.model = model
        self.model_config = model_config if model_config is not None else dict()

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
    def model_type(self):
        return self.model_config['model_type']

    def get_module_names(self):
        return self.model_config['module_names']

    def get_model_name(self):
        return self.model_config['model_name']

    def get_layers_name(self):
        return self.model_config['layers_name']
