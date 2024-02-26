import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
)
from accelerate.big_modeling import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
import os
from abc import ABC, abstractmethod
from .AnyPrecisionLinear import AnyPrecisionLinear


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_AP_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, AnyPrecisionLinear)}


def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


class BaseAPForCausalLM(nn.Module, ABC):
    def __init__(
            self, model, model_type, is_quantized, config
    ):
        super().__init__()
        self.model: PreTrainedModel = model
        self.model_type: str = model_type
        self.is_quantized: bool = is_quantized
        self.search_result = None
        self.config: PretrainedConfig = config

    def to(self, device: str):
        return self.model.to(device)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        with torch.inference_mode():
            return self.model.generate(*args, **kwargs)

    @staticmethod
    def _load_config(
            model_path,
            trust_remote_code=True,
    ):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return config

    @classmethod
    def from_quantized(
            cls,
            quant_model_path,
            max_new_tokens=None,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            safetensors=True,
            is_quantized=True,
            fuse_layers=False,
            device_map="balanced",
            offload_folder=None,
            exclude_modules=None,
            supported_bits=None
    ):
        # [STEP 1-2] Load weights path and configs
        config = cls._load_config(
            quant_model_path,
            trust_remote_code,
        )

        # [STEP 3] Load model : TODO fix flash attention to option
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                # attn_implementation="flash_attention_2",
            )

        APModel = cls(
            model,
            config.model_type,
            is_quantized=is_quantized,
            config=config
        )

        # Prepare AnyPrecisionLinear layers, replace nn.Linear
        APModel._load_quantized_modules(
            exclude_modules=exclude_modules,
            supported_bits=supported_bits,
        )

        APModel.tie_weights()

        # Look for the weights file and load it
        for file in os.listdir(quant_model_path):
            file_path = os.path.join(quant_model_path, file)
            if file.endswith('.bin'):
                q_model = torch.load(file_path, map_location="cpu")
                break
        else:
            raise FileNotFoundError(f"No weights file found in {quant_model_path}")

        device_map = {key: 'cpu' for key in q_model.keys()}

        # loads the weights into modules and distributes
        # across available devices automatically
        load_checkpoint_and_dispatch(
            APModel.model,
            checkpoint=quant_model_path,
            device_map=device_map,
            no_split_module_classes=[APModel.layer_type],
            offload_folder=offload_folder,
            dtype=torch_dtype,
        )

        # Dispath to devices
        if fuse_layers:
            APModel.fuse_layers()

        APModel.refine_maxbits_linears()

        return APModel

    def _load_quantized_modules(
            self, exclude_modules=None, supported_bits=None
    ):
        # Get blocks of model
        layers = self.get_model_layers()

        if exclude_modules is None:
            exclude_modules = ['lm_head']

        for layer in tqdm(layers, desc="Replacing layers..."):
            # Get every linear layer in a block
            named_linears = get_named_linears(layer)

            # Replace nn.Linear with AnyPrecisionLinear
            for name, module in named_linears.items():
                if name in exclude_modules:
                    continue

                wqlinear = AnyPrecisionLinear(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                    supported_bits=supported_bits,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                set_op_by_name(layer, name, wqlinear)

            torch.cuda.empty_cache()
            gc.collect()

    def refine_maxbits_linears(self):
        # Get blocks of model
        layers = self.get_model_layers()
        for layer in tqdm(layers, desc="Replacing layers..."):
            # Get every linear layer in a block and refine bits
            named_linears = get_AP_linears(layer)
            for _, module in named_linears.items():
                module.refine_bits()

        torch.cuda.empty_cache()
        gc.collect()

    def change_bits(self, changed_bits):
        layers = self.get_model_layers()
        for layer in layers:
            # Get every linear layer in a block and refine bits
            named_linears = get_AP_linears(layer)
            for _, module in named_linears.items():
                module.change_bits(changed_bits)

    def tie_weights(self):
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()

    @abstractmethod
    def get_model_layers(self):
        pass

    @abstractmethod
    def fuse_layers(self):
        pass

    @abstractmethod
    def get_act_for_scaling(self):
        pass

    @abstractmethod
    def move_embed(self, device: str):
        pass

    @property
    @abstractmethod
    def layer_type(self):
        pass

    @property
    @abstractmethod
    def max_new_tokens_key(self):
        pass
