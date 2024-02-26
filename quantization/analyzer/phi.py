from auto import ModelAnalyzer


class Phi2Analyzer(ModelAnalyzer):
    @property
    def model_type(self):
        return "phi-2"

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

    def get_model_name(self):
        return "model"

    def get_layers_name(self):
        return "layers"
