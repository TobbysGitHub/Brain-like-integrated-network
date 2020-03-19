import sys

import model.train
import interface.img_gen.train


def train(num_units_regions=None,
          dim_attention_global=None,
          epochs=None):
    argv = ['train']

    if num_units_regions is not None:
        argv.append('--num_units_regions')
        for num in num_units_regions:
            argv.append(str(num))
    if dim_attention_global is not None:
        argv.extend(['--dim_attention_global', str(dim_attention_global)])
    if epochs is not None:
        argv.extend(['--epochs', str(epochs)])

    sys.argv = argv
    model_name = model.train.main()

    argv = ['train', '--model_repr', model_name]

    sys.argv = argv
    interface.img_gen.train.main()
