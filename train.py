import sys

import model.train
import interface.img_gen.train


def train(num_units_regions=None,
          dim_attention_global=None,
          epochs=None,
          img_gen_opt_dict=None,
          **kwargs):
    argv = ['train']

    if num_units_regions is not None:
        argv.append('--num_units_regions')
        for num in num_units_regions:
            argv.append(str(num))
    if dim_attention_global is not None:
        argv.extend(['--dim_attention_global', str(dim_attention_global)])
    if epochs is not None:
        argv.extend(['--epochs', str(epochs)])

    for k, v in kwargs.items():
        argv.append('--' + k)
        argv.append(str(v))

    sys.argv = argv
    model_name = model.train.main()

    argv = ['train', '--model_repr', model_name]
    if img_gen_opt_dict is not None:
        for k, v in img_gen_opt_dict.items():
            argv.append('--' + k)
            argv.append(str(v))

    sys.argv = argv
    interface.img_gen.train.main()


if __name__ == '__main__':
    train(batch_size=256, epochs=0, img_gen_opt_dict=dict(epochs=0))
