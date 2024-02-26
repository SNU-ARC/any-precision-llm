from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, AutoTokenizer
import logging


def get_modules(layer, model_type):
    if model_type in ('llama', 'mistral', 'gemma'):
        return [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ]
    elif model_type == 'opt':
        return [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.out_proj,
            layer.fc1,
            layer.fc2,
        ]
    elif model_type == 'phi-2':
        return [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.dense,
            layer.mlp.fc1,
            layer.mlp.fc2,
        ]
    else:
        raise NotImplementedError(f"Unsupported model type {model_type}")


def get_module_names(model_type):
    if model_type == "opt":
        return ["q", "k", "v", "o", "up", "down"]
    elif model_type in ("mistral", "llama", "gemma"):
        return ["q", "k", "v", "o", "gate", "up", "down"]
    elif model_type == "phi-2":
        return ["q", "k", "v", "o", "up", "down"]
    else:
        raise NotImplementedError(f"Model type {model_type} not supported")


def get_sequential(model_type):
    if model_type == "opt":
        return [
            'self_attn.q_proj',
            'self_attn.k_proj',
            'self_attn.v_proj',
            'self_attn.out_proj',
            'fc1',
            'fc2',
        ]
    elif model_type in ("mistral", "llama", "gemma"):
        return [
            'self_attn.q_proj',
            'self_attn.k_proj',
            'self_attn.v_proj',
            'self_attn.o_proj',
            'mlp.gate_proj',
            'mlp.up_proj',
            'mlp.down_proj',
        ]
    elif model_type == "phi-2":
        return [
            'self_attn.q_proj',
            'self_attn.k_proj',
            'self_attn.v_proj',
            'self_attn.dense',
            'mlp.fc1',
            'mlp.fc2',
        ]
    else:
        raise NotImplementedError(f"Model type {model_type} not supported")


def get_model(model, model_type):
    if model_type == 'opt':
        return model.model.decoder
    elif model_type in ('llama', 'mistral', 'gemma'):
        return model.model
    elif model_type == 'phi-2':
        return model.model
    else:
        raise NotImplementedError(f"Model type {model_type} not supported")


def get_layers(model, model_type):
    _model = get_model(model, model_type)
    if model_type == 'opt':
        return _model.layers
    elif model_type in ('llama', 'mistral', 'gemma'):
        return _model.layers
    elif model_type == 'phi-2':
        return _model.layers
    else:
        raise NotImplementedError(f"Model type {model_type} not supported")


def get_layers_name(model_type):
    if model_type == 'opt':
        return 'model.decoder.layers'
    elif model_type in ('llama', 'mistral', 'gemma'):
        return 'model.layers'
    elif model_type == 'phi-2':
        return 'model.layers'
    else:
        raise NotImplementedError(f"Model type {model_type} not supported")


def get_embedding(model, model_type):
    _model = get_model(model, model_type)
    if model_type == "opt":
        return [_model.embed_tokens, _model.embed_positions]
    elif model_type in ("llama", "mistral", "gemma"):
        return [_model.embed_tokens]
    elif model_type == "phi-2":
        return [_model.embed_tokens]
    else:
        raise NotImplementedError(f"Model type {model_type} not supported")


def get_norm(model, model_type):
    _model = get_model(model, model_type)
    if model_type == "opt":
        return _model.final_layer_norm
    elif model_type in ("llama", "mistral", "gemma"):
        return _model.norm
    elif model_type == "phi-2":
        return _model.final_layer_norm
    else:
        raise NotImplementedError(f"Model type {model_type} not supported")


def get_model_weights(model, model_type):
    layers = get_layers(model, model_type)
    model_layers = []
    module_names = get_module_names(model_type)

    for layer in layers:
        layer_data = {}
        modules = get_modules(layer, model_type)

        assert len(modules) == len(module_names), \
            "number of modules and module names don't match: {} vs {}".format(len(modules), len(module_names))

        for module, name in zip(modules, module_names):
            layer_data[name] = module.weight.data.cpu()

        model_layers.append(layer_data)

    return model_layers


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
