from any_precision.evaluate import eval
from any_precision.evaluate.helpers import utils

parents = utils.get_subdirs('./cache/upscaled')

for parent in parents:
    eval.fake_pack(parent, verbose=True)
