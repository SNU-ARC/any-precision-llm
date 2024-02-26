from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, AutoTokenizer
import logging


def guess_model_type(model):
    assert isinstance(model, PreTrainedModel), f"Expected model to be a PreTrainedModel, got {type(model)}"
    class_name = model.__class__.__name__

    class_name_lower = class_name.lower()

    if "opt" in class_name_lower:
        model_type = "opt"
    elif "llama" in class_name_lower:
        model_type = "llama"
    elif "mistral" in class_name_lower:
        model_type = "mistral"
    elif "phi" in class_name_lower:
        model_type = "phi-2"
    elif "gemma" in class_name_lower:
        model_type = "gemma"
    else:
        raise RuntimeError(f"Failed to guess model type from model object: {class_name}")

    # Convert the above to logs
    logging.info(f"Guessing model type from model object")
    logging.info(f"Class name: {class_name}")
    logging.info(f"Guesed model type: {model_type}")

    return model_type


def load_model(model_str_or_model):
    """Returns a model from a string or a model object. If a string is passed, it will be loaded from the HuggingFace"""
    if isinstance(model_str_or_model, str):
        model = AutoModelForCausalLM.from_pretrained(model_str_or_model, trust_remote_code=True)
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
