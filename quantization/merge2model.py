import torch
import os
import numpy as np
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import pack


class QuantConfig:
    ### THESE HARDCODED ARGS WILL BE REPLACED BY CLI ARGS
    model_name_or_path = 'facebook/opt-1.3b'
    upscale_output_dir = '../cache/parent/(opt-1.3b)-w8_orig3-c4_s100_blk512'
    output_model_dir = '../cache/models'

class ModelConfig:
    def __init__(self, key, prefix, suffix):
        self.keys = key
        self.prefix = prefix
        self.suffix = suffix


model = AutoModelForCausalLM.from_pretrained(QuantConfig.model_name_or_path, trust_remote_code=True)
model = model.state_dict()

output_model_path = os.path.join(QuantConfig.output_model_dir, QuantConfig.model_name_or_path.split('/')[-1] + '.pt')
param_path = QuantConfig.upscale_output_dir

newmodel = model.copy()

opt_keys = ['q', 'k', 'v', 'o', 'up', 'down']
llama_keys = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
mistral_keys = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']

opt_prefix = 'model.decoder.layers.'
llama_prefix = 'model.layers.'
mistral_prefix = 'model.layers.'

opt_suffix = ['.self_attn.q_proj', '.self_attn.k_proj', '.self_attn.v_proj', '.self_attn.out_proj', '.fc1', '.fc2']
llama_suffix = ['.self_attn.q_proj', '.self_attn.k_proj', '.self_attn.v_proj', '.self_attn.o_proj', '.mlp.gate_proj',
                '.mlp.up_proj', '.mlp.down_proj']
mistral_suffix = ['.self_attn.q_proj', '.self_attn.k_proj', '.self_attn.v_proj', '.self_attn.o_proj', '.mlp.gate_proj',
                  '.mlp.up_proj', '.mlp.down_proj']

MODELDICT = {
    "llama": ModelConfig(llama_keys, llama_prefix, llama_suffix),
    "opt": ModelConfig(opt_keys, opt_prefix, opt_suffix),
    "mistral": ModelConfig(mistral_keys, mistral_prefix, mistral_suffix),
}

for key, conf in MODELDICT.items():
    if key in param_path:
        modelConfig = conf
        break

bits = [3, 4, 5, 6, 7, 8]
num_layers = len(os.listdir(os.path.join(param_path, 'weights')))

for layeridx in tqdm(range(num_layers)):
    weightpath = os.path.join(param_path, f'weights', f'l{layeridx}.pt')
    curlayer = torch.load(weightpath)

    for keyidx in range(len(modelConfig.keys)):
        N = len(curlayer[modelConfig.keys[keyidx]])
        K = len(curlayer[modelConfig.keys[keyidx]][0][0])
        # qweight = np.empty([N, K], dtype='uint8')
        # for i in range(N):
        #     qweight[i] = curlayer[modelConfig.keys[keyidx]][i][0]
        qweight = curlayer[modelConfig.keys[keyidx]]

        bitarray = np.empty(0, dtype=np.uint8)
        for bit in range(8):
            curbitpack = np.packbits(torch.tensor(qweight&(1<<(7-bit))).to(torch.bool))
            bitarray = np.append(bitarray, curbitpack)

        # permute bitmap @ Section 5.3
        bitarray = bitarray.reshape((8, N, K // 8))
        weighttensor = pack.permute_bitmaps_int32(bitarray)
        weighttensor = torch.from_numpy(weighttensor)


        newpath = f'{modelConfig.prefix}{layeridx}{modelConfig.suffix[keyidx]}.qweight'
        newmodel[newpath] = weighttensor

        wpath = f'{modelConfig.prefix}{layeridx}{modelConfig.suffix[keyidx]}.weight'
        del newmodel[wpath]

    for bit in bits:
        curpath = os.path.join(param_path, f'lut_{bit}', f'l{layeridx}.pt')
        curlayer = torch.load(curpath)

        # shapes are (out_features, 1, in_features)

        for keyidx in range(len(modelConfig.keys)):
            N = len(curlayer[modelConfig.keys[keyidx]])
            LUTsize = len(curlayer[modelConfig.keys[keyidx]][0][0])
            curLUT = np.empty([N, LUTsize], dtype='float16')

            for i in range(N):
                curLUT[i] = curlayer[modelConfig.keys[keyidx]][i][0]

            newpath = f'{modelConfig.prefix}{layeridx}{modelConfig.suffix[keyidx]}.lut{bit}'
            newmodel[newpath] = torch.tensor(curLUT)

os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
torch.save(newmodel, output_model_path)
