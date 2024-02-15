
class QuantConfig():
    ### THESE HARDCODED ARGS WILL BE REPLACED BY CLI ARGS
    dataset = 'c4'
    model_name_or_path = 'facebook/opt-1.3b'
    seq_len = 512
    num_examples = 100
    output_dir = '../cache/gradients'
    model_type = 'opt'
    upscale_output_dir = '../cache/parent/(opt-1.3b)-c4-w8_orig3'
    output_model_dir = '../cache/models'