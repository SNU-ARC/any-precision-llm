from auto import ModelAnalyzer


class OPTAnalyzer(ModelAnalyzer):
    @property
    def model_type(self):
        return "opt"

    def get_module_names(self):
        return [
            'self_attn.q_proj',
            'self_attn.k_proj',
            'self_attn.v_proj',
            'self_attn.out_proj',
            'fc1',
            'fc2',
        ]

    def get_model_name(self):
        return "model.decoder"

    def get_layers_name(self):
        return "layers"
