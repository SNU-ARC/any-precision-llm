import transformers
import torch
from tqdm import tqdm
import os
import utils
import argparse

default_output_dir = '../cache/gradients'

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='c4', help="Dataset to use for gradient calculation")
parser.add_argument("--model_name_or_path", type=str, default='facebook/opt-1.3b',
                    help="Model to use for gradient calculation")
parser.add_argument("--seq_len", type=int, default=512, help="Sequence length to use for gradient calculation")
parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to use for gradient calculation")
parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Output directory for gradients")
parser.add_argument("--model_type", type=str, default=None, help="Model type to use for gradient calculation")


def calculate_gradients(model, model_type, dataset, seq_len, num_examples, output_path):
    from datautils import get_loaders
    print("Calibration with " + dataset)
    # TODO: remove model from get_loaders
    dataloader, testloader = get_loaders(dataset, model=model, seqlen=seq_len, nsamples=num_examples)

    if isinstance(model, str):
        model = transformers.AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
    else:
        assert isinstance(model, transformers.AutoModelForCausalLM), "Model must be a string or a transformers model"

    if model_type is None:
        model_type = utils.get_model_type(model)

    model = model.bfloat16()
    model.eval()
    model.cuda()

    layers = utils.get_layers(model, model_type)

    # Register hook to store the square of the gradients
    def square_grad_hook(grad):
        return grad.pow(2)

    for layer in layers:
        for module in utils.get_modules(layer, model_type=model_type):
            module.weight.register_hook(square_grad_hook)

    # Calculate gradients through loss.backward()
    for data in tqdm(dataloader):
        data = data[0]
        x = data.cuda()
        outputs = model(input_ids=x, labels=x)
        loss = outputs.loss
        loss.backward()

    # Harvest the gradients
    gradients = []
    for layer in layers:
        gradients_per_layer = {}
        for module, module_name in zip(utils.get_modules(layer, model_type=model_type),
                                       utils.get_module_names(model_type=model_type)):
            gradients_per_layer[module_name] = module.weight.grad.cpu()
        gradients.append(gradients_per_layer)

    # Save the gradients to file
    if output_path is not None:
        # check if the file already exists
        if os.path.exists(output_path):
            input(f"[WARNING] File {output_path} already exists. Press enter to overwrite or Ctrl+C to cancel.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(gradients, output_path)

    return gradients


if __name__ == "__main__":
    args = parser.parse_args()
    calculate_gradients(args.model_name_or_path,
                        args.model_type,
                        args.dataset,
                        args.seq_len,
                        args.num_examples,
                        args.output_dir)
