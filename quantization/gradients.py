import transformers
import torch
from tqdm import tqdm
import os
import utils

### THESE HARDCODED ARGS WILL BE REPLACED BY CLI ARGS
dataset = 'c4'
model_name_or_path = 'facebook/opt-1.3b'
seq_len = 512
num_examples = 100
output_dir = '../cache/gradients'
model_type = 'opt'


def train():
    from datautils import get_loaders
    print("Calibration with " + dataset)
    dataloader, testloader = get_loaders(dataset, model=model_name_or_path, seqlen=seq_len, nsamples=num_examples)

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = model.bfloat16()
    model.cuda()

    layers = utils.get_layers(model, model_type)

    def square_grad_hook(grad):
        return grad.pow(2)

    for layer in layers:
        for module in utils.get_modules(layer, model_type=model_type):
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
        for module, module_name in zip(utils.get_modules(layer, model_type=model_type),
                                       utils.get_module_names(model_type=model_type)):
            gradients_per_layer[module_name] = module.weight.grad.cpu()
        gradients.append(gradients_per_layer)

    save_path = f"{output_dir}/({model_name_or_path.split('/')[-1]})-{dataset}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(gradients, save_path)


if __name__ == "__main__":
    train()
