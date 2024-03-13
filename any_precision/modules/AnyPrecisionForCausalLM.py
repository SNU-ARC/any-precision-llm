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

from .AnyPrecisionLinear import AnyPrecisionLinear
from any_precision.analyzer.analyzer import get_analyzer


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


class AnyPrecisionForCausalLM(nn.Module):
    def __init__(
            self, model, is_quantized, config, precisions, supported_bits
    ):
        super().__init__()
        self.model: PreTrainedModel = model
        self.is_quantized: bool = is_quantized
        self.search_result = None
        self.config: PretrainedConfig = config
        self.precisions = precisions
        self.supported_bits = supported_bits
        self.precision = max(self.precisions)
        self.analyzer = get_analyzer(model)

    def forward(self, *args, **kwargs):
        if 'precision' in kwargs:
            prev_precision = self.precision
            precision = kwargs.pop('precision')
            self.set_precision(precision)
        else:
            prev_precision = self.precision

        results = self.model.forward(*args, **kwargs)

        self.set_precision(prev_precision)
        return results

    def generate(self, *args, **kwargs):
        if 'precision' in kwargs:
            prev_precision = self.precision
            precision = kwargs.pop('precision')
            self.set_precision(precision)
        else:
            prev_precision = self.precision

        with torch.inference_mode():
            results = self.model.generate(*args, **kwargs)

        self.set_precision(prev_precision)
        return results

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
            precisions=None
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

        supported_bits = list(range(config.anyprec['seed_precision'],
                                    config.anyprec['parent_precision'] + 1))
        if precisions is None:
            precisions = supported_bits
        else:
            assert len(precisions) == len(set(precisions)), "Precisions must be unique"
            assert all(bit in supported_bits for bit in precisions), \
                f"Supported bits {precisions} must be a subset of model supported bits {supported_bits}"

        ap_model = cls(
            model,
            is_quantized=is_quantized,
            config=config,
            precisions=precisions,
            supported_bits=supported_bits,
        )

        # Replace to AnyPrecisionLinear layers
        ap_model._load_quantized_modules()

        ap_model.tie_weights()

        device_map = {key: 'cpu' for key in ap_model.model.state_dict().keys()}

        # loads the weights into modules and distributes
        # across available devices automatically
        load_checkpoint_and_dispatch(
            ap_model.model,
            checkpoint=quant_model_path,
            device_map=device_map,
            no_split_module_classes=[ap_model.layer_type],
            offload_folder=offload_folder,
            dtype=torch_dtype,
        )

        # Dispath to devices
        if fuse_layers:
            ap_model.fuse_layers()

        ap_model.prune_precisions()

        return ap_model

    def _load_quantized_modules(self):
        # Get blocks of model
        layers = self.get_model_layers()

        for layer in tqdm(layers, desc="Replacing layers..."):
            # Get every linear layer in a block
            named_linears = get_named_linears(layer)

            # Replace nn.Linear with AnyPrecisionLinear
            for name, module in self.analyzer.get_modules():
                wqlinear = AnyPrecisionLinear(
                    module.in_features, module.out_features,
                    self.supported_bits,
                    bias=module.bias is not None,
                    precisions=self.precisions,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                set_op_by_name(layer, name, wqlinear)

            torch.cuda.empty_cache()
            gc.collect()

    def prune_precisions(self):
        # Get blocks of model
        layers = self.get_model_layers()
        for layer in layers:
            named_linears = get_AP_linears(layer)
            for _, module in named_linears.items():
                module.prune_precisions()

        torch.cuda.empty_cache()
        gc.collect()

    def set_precision(self, precision):
        layers = self.get_model_layers()
        for layer in layers:
            # Get every linear layer in a block and refine bits
            named_linears = get_AP_linears(layer)
            for _, module in named_linears.items():
                module.set_precision(precision)
        self.precision = precision

    def tie_weights(self):
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()

    def get_model_layers(self):
        module = self.model
        for attrib_name in self.config.anyprec['arch_config']['model_name'].split('.'):
            module = getattr(module, attrib_name)
        return getattr(module, self.config.anyprec['arch_config']['layers_name'])

    def fuse_layers(self):
        if 'fuse_target_layers' not in self.model_config:
            raise NotImplementedError("This model does not support layer fusion")
        # TODO implement layer fusion
        pass

    @property
    def layer_type(self):
        for layer in self.get_model_layers():
            layer_class_name = layer.__class__.__name__
            if layer_class_name.endswith("DecoderLayer"):
                return layer_class_name
        return None

    @property
    def device(self):
        return self.model.device
