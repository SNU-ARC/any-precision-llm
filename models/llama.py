import tqdm
from .base import BaseAPForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as OldLlamaDecoderLayer,
    LlamaForCausalLM as OldLlamaForCausalLM
)


class LlamaAPForCausalLM(BaseAPForCausalLM):
    @property
    def layer_type(self):
        return "LlamaDecoderLayer"

    @property
    def max_new_tokens_key(self):
        return "max_position_embeddings"

    def fuse_layers(self):
        raise NotImplementedError("Llama does not support layer fusion")

    def get_model_layers(self):
        return self.model.model.layers

    def get_act_for_scaling(self):
        return dict(
            is_scalable=False
        )

    def move_embed(self, device: str):
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(device)
