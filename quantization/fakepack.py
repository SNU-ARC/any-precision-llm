import time

import torch

import transformers

import json
import os
from tqdm import tqdm

import torch.nn as nn

def round_to_nearest_pole_sim(w, poles):
    """
    w: weight values (1d vector)
    poles: tuple of values

    Round the numbers in w to the nearest value in poles.
    """
    stack = []
    for c in poles:
        diff = (w - c).abs()
        stack.append(diff)
    diff = torch.stack(stack)
    idx = diff.argmin(axis=0)
    aug = 0
    for i, c in enumerate(poles):
        aug += (idx == i) * c
    return aug

# drop-in layer replacement class
class QuantLinearLUT(nn.Module):
    def __init__(self, bits, infeatures, outfeatures, bias, include_sparse=False, numvals=0, topX=10):
        super().__init__()
        #if bits not in [3,4]:
        #    raise NotImplementedError("Only 3 and 4 bits is supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits

        self.register_buffer('qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32))
        if bias:
            self.include_bias = True
            self.register_buffer('bias', torch.zeros((outfeatures)))
        else:
            self.include_bias = False
            self.bias = None
        self.register_buffer('lookup_table', torch.zeros((outfeatures, 2**self.bits), dtype=torch.float32))

        self.include_sparse = include_sparse
        self.numvals = numvals
        self.topX = topX
        if numvals > 0:
            self.register_buffer('rows', torch.zeros(outfeatures+1, dtype=torch.int32))
            self.register_buffer('cols', torch.zeros(numvals, dtype=torch.int32))
            self.register_buffer('vals', torch.zeros(numvals, dtype=torch.float32))
        if topX > 0:
            self.register_buffer('full_rows', torch.zeros((infeatures, topX), dtype=torch.float32))
            self.register_buffer('full_row_indices', torch.zeros(topX, dtype=torch.int32))

    def pack_(self, linear, lookup_table, include_sparse):
        self.register_buffer('weight', torch.zeros((self.outfeatures, self.infeatures), dtype=torch.float16))

        if self.include_bias: #linear.bias is not None:
            self.bias = linear.bias.clone() #todo: check this condition

        # self.lookup_table = lookup_table.float()
        lut,outliers = lookup_table

        # handle dense matrix
        intweight = linear.weight.data.clone()

        if include_sparse:
            outliers = outliers.to_dense()

        assert (self.outfeatures == len(lut))

        num_channels = len(lut)
        for channel in range(num_channels):
            centroid, indices = lut[channel][0] # last 0 is for group 0
            centroid = torch.from_numpy(centroid).to(torch.float16)
            indices = torch.from_numpy(indices)
            self.weight[channel] = centroid[indices.long()]


# function to iterate through model layers and replace with our LUT-based layer
def make_quant_lut(module, names, bits, name='', include_sparse=False, numvals=None, topX=0):
    if isinstance(module, QuantLinearLUT):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 not in names:
            continue
        num = 0
        if numvals:
            num = getattr(numvals[name1])
        delattr(module, attr)
        setattr(
            module,
            attr,
            QuantLinearLUT(
                bits,
                tmp.in_features,
                tmp.out_features,
                tmp.bias is not None,
                include_sparse=include_sparse,
                numvals=num,
                topX=topX,
            ),
        )

    for name1, child in module.named_children():
        make_quant_lut(
            child,
            names,
            bits,
            name + '.' + name1 if name != '' else name1,
            include_sparse=include_sparse,
            numvals=numvals,
            topX=topX,
        )



# function to find layers in the network (either for packing or for replacement)
def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


@torch.no_grad()
def llama_sequential(model, folder, include_sparse, updated_format, bits):
    try:
        model_type = type(model.model).__name__
    except AttributeError:
        model_type = type(model.transformer).__name__
    if model_type in ('LlamaModel', 'MistralModel', 'PhiModel'):
        layers = model.model.layers
    elif model_type == 'OPTModel':
        layers = model.model.decoder.layers
    else:
        raise NotImplementedError(f"Model type {model_type} not supported")

    quantizers = {}
    for i in tqdm(range(len(layers)), desc='Loading LUTs...'):
        lut_layer = torch.load(f"{folder}/lut_{bits}/l{i}.pt")
        weights_layer = torch.load(f"{folder}/weights/l{i}.pt")

        if model_type in ('LlamaModel', 'MistralModel'):
            sequential_lut = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
            sequential_lut_real_name = {
                'q': 'self_attn.q_proj',
                'k': 'self_attn.k_proj',
                'v': 'self_attn.v_proj',
                'o': 'self_attn.o_proj',
                'gate': 'mlp.gate_proj',
                'up': 'mlp.up_proj',
                'down': 'mlp.down_proj'
            }
        elif model_type == 'OPTModel':
            sequential_lut = ['q', 'k', 'v', 'o', 'up', 'down']
            sequential_lut_real_name = {
                'q': 'self_attn.q_proj',
                'k': 'self_attn.k_proj',
                'v': 'self_attn.v_proj',
                'o': 'self_attn.out_proj',
                'up': 'fc1',
                'down': 'fc2'
            }
        elif model_type == 'PhiModel':
            sequential_lut = ['q', 'k', 'v', 'o', 'up', 'down']
            sequential_lut_real_name = {
                'q': 'self_attn.q_proj',
                'k': 'self_attn.k_proj',
                'v': 'self_attn.v_proj',
                'o': 'self_attn.dense',
                'up': 'mlp.fc1',
                'down': 'mlp.fc2'
            }
        else:
            raise NotImplementedError(f"Model type {model_type} not supported")

        for s in sequential_lut:
            lut = lut_layer[s]
            weights = weights_layer[s]
            # obtain the MSB bits
            mask = (1 << bits) - 1
            weights = (weights >> (8 - bits)) & mask
            params = []
            for j in range(lut.shape[0]):
                params.append([])
                for k in range(lut.shape[1]):
                    params[-1].append((lut[j, k], weights[j, k]))
            outliers = None
            name = sequential_lut_real_name[s]
            if model_type in ('LlamaModel', 'MistralModel', 'PhiModel'):
                quantizers['model.layers.%d.%s' % (i, name)] = [params, outliers]
            elif model_type == 'OPTModel':
                quantizers['model.decoder.layers.%d.%s' % (i, name)] = [params, outliers]
            else:
                raise NotImplementedError(f"Model type {model_type} not supported")

    return quantizers


def llama_pack(model, quantizers, wbits, include_sparse):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant_lut(model, quantizers, wbits, include_sparse=include_sparse)

    qlayers = find_layers(model, [QuantLinearLUT])
    sparsedict = {}

    for name in tqdm(qlayers, desc='Packing layers...'):
        lookup_table = quantizers[name]
        layers[name].cpu()
        qlayers[name].pack_(layers[name], lookup_table, include_sparse)
        # qlayers[name].pack(layers[name], lookup_table, include_sparse)
        if include_sparse:
            sparsedict[name] = qlayers[name].vals.shape[-1]

    return model, sparsedict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='llama model to load'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 5, 6, 7, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--save', type=str, required=True,
        help='Save quantized checkpoint under this name.'
    )

    # sparse args
    parser.add_argument(
        '--folder', type=str, default='',
        help='Path to folder containing luts and outliers.'
    )
    parser.add_argument(
        '--include_sparse', action='store_true',
        help='Whether loaded checkpoint has sparse matrix.'
    )
    parser.add_argument(
        '--updated-format', action='store_true',
        help='Whether to use the new PTB and C4 eval'
    )

    args = parser.parse_args()
    assert not args.include_sparse, "Sparse not supported yet"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype='auto'
    )
    model.eval()

    print('Running llama_sequential')
    tick = time.time()
    quantizers = llama_sequential(
        model=model,
        folder=args.folder,
        include_sparse=args.include_sparse,
        updated_format=True,
        bits=args.wbits,
    )
    print(f"llama_sequential took {time.time() - tick} seconds.")

    print("Running llama_pack")
    tick = time.time()
    model, numvals = llama_pack(
        model=model,
        quantizers=quantizers,
        wbits=args.wbits,
        include_sparse=args.include_sparse,
    )
    print(f"llama_pack took {time.time() - tick} seconds.")

    model_dict = model.state_dict()

    if args.include_sparse:
        # need to merge in sparse dict
        for k, v in numvals.items():
            model_dict['sparse_threshold.' + k] = v

    # create directory to save model
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)

    print(f"Saving model to {args.save}")
    # save model
    torch.save(model_dict, args.save + "/pytorch_model.bin")

    # get directory to save quant_config
    data = {
        "wbits": args.wbits
    }
    output_fn = os.path.join(args.save, "quant_config.json")

    # save quant_config
    with open(output_fn, 'w') as f:
        json.dump(data, f, indent=4)

    print("Save complete.")
