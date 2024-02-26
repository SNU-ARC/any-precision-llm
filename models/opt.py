from .base import BaseAPForCausalLM


class OPTAPForCausalLM(BaseAPForCausalLM):
    @property
    def layer_type(self):
        return "OPTDecoderLayer"

    @property
    def max_new_tokens_key(self):
        return "max_position_embeddings"

    def fuse_layers(self):
        raise NotImplementedError("OPT does not support layer fusion")

    def get_model_layers(self):
        return self.model.model.decoder.layers

    def get_act_for_scaling(self):
        return dict(
            is_scalable=False
        )

    def move_embed(self, device: str):
        self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.to(device)
        self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.to(device)
