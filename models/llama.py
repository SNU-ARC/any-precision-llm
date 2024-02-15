import tqdm
from .base import BaseAPForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as OldLlamaDecoderLayer,
    LlamaForCausalLM as OldLlamaForCausalLM
)

class LlamaAPForCausalLM(BaseAPForCausalLM):
    layer_type = "LlamaDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: OldLlamaForCausalLM):
        #fuser = LlamaFuser(model)
        #fuser.fuse_transformer()
        fuser = None
        pass

    @staticmethod
    def get_model_layers(model: OldLlamaForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldLlamaDecoderLayer):
        return dict(
            is_scalable=False
        )

    @staticmethod
    def move_embed(model: OldLlamaForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

