from .base import BaseAPForCausalLM



class MistralAPForCausalLM(BaseAPForCausalLM):
    @property
    def layer_type(self):
        return "MistralDecoderLayer"

    @property
    def max_new_tokens_key(self):
        return "max_position_embeddings"

    def fuse_layers(self):
        raise NotImplementedError("Mistral does not support layer fusion")

    def get_model_layers(self):
        return self.model.model.layers

    def move_embed(self, device: str):
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(device)
