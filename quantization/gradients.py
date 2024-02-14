import transformers
import torch
from tqdm import tqdm
import os

### THESE HARDCODED ARGS WILL BE REPLACED BY CLI ARGS
dataset = 'c4'
model_name_or_path = 'facebook/opt-1.3b'
seq_len = 512
num_examples = 100
output_dir = '../cache/gradients'
model_type = 'opt'


def get_modules(layer, model_type):
    if model_type in ('llama', 'mistral'):
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


def get_layers(model, model_type):
    if model_type in ('llama', 'mistral'):
        return model.model.layers
    elif model_type == 'opt':
        return model.model.decoder.layers
    elif model_type == 'phi-2':
        return model.model.layers
    else:
        raise NotImplementedError(f"Unsupported model type {model_type}")


def train():
    from datautils import get_loaders
    print("Calibration with " +dataset)
    dataloader, testloader = get_loaders(dataset, model=model_name_or_path, seqlen=seq_len, nsamples=num_examples)

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = model.bfloat16()
    model.cuda()

    layers = get_layers(model, model_type)

    def square_grad_hook(grad):
        return grad.pow(2)

    for layer in layers:
        for module in get_modules(layer, model_type=model_type):
            module.weight.register_hook(square_grad_hook)

    for data in tqdm(dataloader):
        data = data[0]
        x = data.cuda()
        outputs = model(input_ids=x, labels=x)
        loss = outputs.loss
        loss.backward()

    gradients = []
    for layer in layers:
        gradients_per_layer = {}
        for module in get_modules(layer, model_type=model_type):
            gradients_per_layer[module] = module.weight.grad

    save_path = f"{output_dir}/({model_name_or_path.split('/')[-1]})-{dataset}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(gradients, save_path)


if __name__ == "__main__":
    train()
