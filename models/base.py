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

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_AP_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, AnyPrecisionLinear)}

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)

class BaseAPForCausalLM(nn.Module):
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
    def fuse_layers(model):
        pass

    def _load_config(
        self,
        model_path,
        trust_remote_code=True,
        max_new_tokens=4096,
    ):
        # # [STEP 1] Download model if path is not a directory
        # if not os.path.isdir(model_path):
        #     ignore_patterns = ["*msgpack*", "*h5*", "optimizer.pt"]
        #     if safetensors:
        #         ignore_patterns.extend(["*.pt*", "*.bin*", "consolidated*"])
        #     else:
        #         ignore_patterns.append("*.safetensors*")

        #     model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

        # # Load model config and set max generation length
        # if max_new_tokens is None and hasattr(self, "max_new_tokens_key"):
        #     config = AutoConfig.from_pretrained(
        #         model_path, trust_remote_code=trust_remote_code, **config_kwargs
        #     )
        #     config.max_new_tokens = getattr(config, self.max_new_tokens_key, 2048)
        #     # To add the generate support for Multi-modal models as well
        #     if hasattr(config, "text_config"):
        #         config.text_config.max_new_tokens = getattr(
        #             config, self.max_new_tokens_key, 2048
        #         )
        # else:
        #     max_new_tokens = 2048 if max_new_tokens is None else max_new_tokens
        #     config = AutoConfig.from_pretrained(
        #         model_path, trust_remote_code=trust_remote_code, **config_kwargs
        #     )
        #     config.max_new_tokens = max_new_tokens

        return config

    @classmethod
    def from_pretrained(
        self,
        model_path,
        model_type,
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code=True,
        safetensors=False,
        device_map=None,
        **model_init_kwargs,
    ):
        # Get weights path and quant config : TODO what is this parameters?
        model_weights_path, config = self._load_config(
            self, model_path, "", safetensors, trust_remote_code=trust_remote_code
        )

        # If not quantized, must load with AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_weights_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            use_safetensors=safetensors,
            device_map=device_map,
            **model_init_kwargs,
        )

        model.eval()

        return self(
            model,
            model_type,
            is_quantized=False,
            config=config,
        )

    @classmethod
    def from_quantized(
        self,
        quant_model_path,
        origin_model_path,
        max_new_tokens=None,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        safetensors=True,
        is_quantized=True,
        fuse_layers=False,
        device_map="balanced",
        offload_folder=None,
        exclude_modules = None,
        supported_bits=None
    ):

        # [STEP 1-2] Load weights path and configs
        config = self._load_config(
            self,
            origin_model_path,
            trust_remote_code,
            max_new_tokens=max_new_tokens,
        )

        # [STEP 3] Load model : TODO fix flash attention to option
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                # attn_implementation="flash_attention_2",
            )

        # Prepare WQLinear layers, replace nn.Linear
        self._load_quantized_modules(
            self,
            model,
            exclude_modules=exclude_modules,
            supported_bits=supported_bits,
        )

        model.tie_weights()

        skip_bits = [3,4,5,6,7,8]
        for bit in supported_bits:
            skip_bits.remove(bit)
        skip_keys = [ f'lut{bit}' for bit in skip_bits]

        q_model = torch.load(quant_model_path)

        device_map = dict()
        for key in q_model.keys():
            if key.split('.')[-1] in skip_keys:
                device_map[key] = 'cpu'
            else:
                device_map[key] = 'cuda:0'

        # loads the weights into modules and distributes
        # across available devices automatically
        load_checkpoint_and_dispatch(
            model,
            checkpoint=quant_model_path,
            device_map=device_map,
            no_split_module_classes=[self.layer_type],
            offload_folder=offload_folder,
            dtype=torch_dtype,
            #skip_keys=skip_keys,
        )

        # Dispath to devices
        if fuse_layers:
            self.fuse_layers(model)

        self.refine_maxbits_linears(self, model)

        return self(
            model,
            config.model_type,
            is_quantized=is_quantized,
            config=config
        )

    def _load_quantized_modules(
        self, model, exclude_modules = None, supported_bits=None
    ):
        # Get blocks of model
        layers = self.get_model_layers(model)
        
        if exclude_modules is None:
            exclude_modules = ['lm_head']

        for layer in tqdm(layers, desc="Replacing layers..."):
            # Get every linear layer in a block
            named_linears = get_named_linears(layer)

            # Replace nn.Linear with AnyPrecisionLinear
            for name, module in named_linears.items():
                if name in exclude_modules:
                    continue

                wqlinear = AnyPrecisionLinear(module.in_features, module.out_features, module.bias is not None,
                                   supported_bits, module.weight.device, module.weight.dtype)
                set_op_by_name(layer, name, wqlinear)

            torch.cuda.empty_cache()
            gc.collect()

    def refine_maxbits_linears(
        self, model
    ):

        # Get blocks of model
        layers = self.get_model_layers(model)

        for i in tqdm(range(len(layers)), desc="Replacing layers..."):
            layer = layers[i]

            # Get every linear layer in a block
            named_linears = get_AP_linears(layer)

            for name, module in named_linears.items():

                module.refine_bits()

        torch.cuda.empty_cache()
        gc.collect()