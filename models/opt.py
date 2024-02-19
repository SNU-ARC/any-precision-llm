from .base import BaseAPForCausalLM
from transformers.models.opt.modeling_opt import (
    OPTDecoderLayer as OldOPTDecoderLayer,
    OPTForCausalLM as OldOPTForCausalLM
)

class OPTAPForCausalLM(BaseAPForCausalLM):
    layer_type = "OPTDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def get_model_layers(model: OldOPTForCausalLM):
        return model.model.decoder.layers

    @staticmethod
    def get_act_for_scaling(module: OldOPTDecoderLayer):
        return dict(
            is_scalable=False
        )

    @staticmethod
    def move_embed(model: OldOPTForCausalLM, device: str):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)

