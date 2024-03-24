import torch
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer


def load_model(model_str_or_model, dtype=torch.bfloat16):
    """Returns a model from a string or a model object. If a string is passed, it will be loaded from the HuggingFace"""
    if isinstance(model_str_or_model, str):
        model = AutoModelForCausalLM.from_pretrained(
            model_str_or_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map='cpu'
        )
    else:
        assert isinstance(model_str_or_model, PreTrainedModel), "model must be a string or a PreTrainedModel"
        model = model_str_or_model
    return model


def is_transformers_tokenizer(obj):
    # List of methods typically found in a transformers tokenizer
    methods_to_check = [
        '__call__', 'encode', 'decode', 'tokenize',
        'convert_tokens_to_ids', 'convert_ids_to_tokens',
        'save_pretrained', 'from_pretrained'
    ]

    # Check if all the methods are present in the object
    return all(hasattr(obj, method) for method in methods_to_check)


def load_tokenizer(model_str_or_model_or_tokenizer):
    """Returns a tokenizer from the model string or model object or tokenizer object"""
    if isinstance(model_str_or_model_or_tokenizer, str):
        model_str = model_str_or_model_or_tokenizer
        return AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    elif isinstance(model_str_or_model_or_tokenizer, PreTrainedModel):
        model_str = model_str_or_model_or_tokenizer.name_or_path
        return AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    else:
        assert is_transformers_tokenizer(model_str_or_model_or_tokenizer), \
            f"Unsupported type for model_str_or_model_or_tokenizer: {type(model_str_or_model_or_tokenizer)}"
        return model_str_or_model_or_tokenizer
