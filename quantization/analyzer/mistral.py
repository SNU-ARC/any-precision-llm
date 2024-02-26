from .base import ModelAnalyzer


class MistralAnalyzer(ModelAnalyzer):
    @property
    def model_type(self):
        return "mistral"

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

    def get_model_name(self):
        return "model"

    def get_layers_name(self):
        return "layers"
