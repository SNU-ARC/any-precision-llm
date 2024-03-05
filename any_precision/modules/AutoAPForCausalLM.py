import torch
import logging
from .BaseAPForCausalLM import BaseAPForCausalLM


class AutoAPForCausalLM(BaseAPForCausalLM):
    def __init__(self, model, tokenizer, *args, **kwargs):
        super().__init__(model, tokenizer, *args, **kwargs)
        self._model_layers = self.auto_detect_model_layers()
        self._layer_type = self.auto_detect_layer_type()
        self._max_new_tokens_key = self.auto_detect_max_new_tokens_key()
        self._embed_tokens = self.auto_detect_embed_tokens()

        if self._layer_type is None:
            raise ValueError("Failed to auto detect layer type")
        else:
            logging.info(f"Auto detected layer type: {self._layer_type}")
        if self._max_new_tokens_key is None:
            raise ValueError("Failed to auto detect max_new_tokens_key")
        else:
            logging.info(f"Auto detected max_new_tokens_key: {self._max_new_tokens_key}")
        if self._model_layers is None:
            raise ValueError("Failed to auto detect model layers")
        else:
            logging.info(f"Auto detected model layers: {self._model_layers}")
        if self._embed_tokens is None:
            raise ValueError("Failed to auto detect embed tokens")
        else:
            logging.info(f"Auto detected embed tokens: {self._embed_tokens}")

    def auto_detect_layer_type(self):
        for layer in self.get_model_layers():
            layer_class_name = layer.__class__.__name__
            if layer_class_name.endswith("DecoderLayer"):
                return layer_class_name
        else:
            return None

    def auto_detect_model_layers(self):
        def _auto_detect_model_layers(module, parent_name=''):
            for name, module in module.named_children():
                full_name = f'{parent_name}.{name}' if parent_name else name
                if isinstance(module, torch.nn.ModuleList):
                    return full_name  # Return the path when ModuleList is found
                else:
                    module_list_path = _auto_detect_model_layers(module, full_name)
                    if module_list_path:
                        return module_list_path
            return None  # Return None if no ModuleList is found

        module_path = _auto_detect_model_layers(self.model)
        if module_path is None:
            return None
        else:
            # get the relevant attribute from the model
            module = self.model
            for attribute in module_path.split('.'):
                module = getattr(module, attribute)
            return module

    def auto_detect_max_new_tokens_key(self):
        key = "max_position_embeddings"
        if key in self.config.__dict__:
            return key
        else:
            return None

    def auto_detect_embed_tokens(self):
        """Recursively find all instances of torch.nn.Embedding in the model and return their attribute names
        """

        def _auto_detect_embed_tokens(module, parent_name=''):
            for name, module in module.named_children():
                full_name = f'{parent_name}.{name}' if parent_name else name
                if isinstance(module, torch.nn.Embedding):
                    yield full_name
                else:
                    yield from _auto_detect_embed_tokens(module, full_name)

        return list(_auto_detect_embed_tokens(self.model))

    @property
    def layer_type(self):
        if self._layer_type is None:
            raise ValueError("Layer type not set")
        return self._layer_type

    @property
    def max_new_tokens_key(self):
        if self._max_new_tokens_key is None:
            raise ValueError("max_new_tokens_key not set")
        return self._max_new_tokens_key

    def fuse_layers(self):
        raise NotImplementedError("AutoAP does not support layer fusion")

    def get_model_layers(self):
        if self._model_layers is None:
            raise ValueError("Model layers not set")
        for name, module in self._model_layers.named_children():
            yield module
