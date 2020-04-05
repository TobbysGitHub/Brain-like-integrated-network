import sys

import model.train
import interface.img_gen.train
import interface.casual_test.test


def train_model(num_units_regions=None, **kwargs):
    argv = ['train']

    if num_units_regions is not None:
        argv.append('--num_units_regions')
        for num in num_units_regions:
            argv.append(str(num))

    for k, v in kwargs.items():
        argv.append('--' + k)
        argv.append(str(v))

    sys.argv = argv
    model_domain = model.train.main()

    return model_domain


def train_gen_net(model_domain, **kwargs):
    argv = ['train', '--model_domain', model_domain]
    for k, v in kwargs.items():
        argv.append('--' + k)
        argv.append(str(v))

    sys.argv = argv
    mode = interface.img_gen.train.main()
    return mode


def test_casual(model_domain, mode, **kwargs):
    argv = ['test', '--model_domain', model_domain, '--mode', str(mode)]
    for k, v in kwargs.items():
        argv.append('--' + k)
        argv.append(str(v))
    sys.argv = argv
    interface.casual_test.test.main()


def train(model_opt=None, img_gen_opt=None, casual_test_opt=None):
    model_domain = train_model(**model_opt)
    if img_gen_opt is not None:
        gen_net_mode = train_gen_net(model_domain, **img_gen_opt)
        if casual_test_opt is not None:
            test_casual(model_domain, gen_net_mode, **casual_test_opt)


if __name__ == '__main__':
    train(model_opt=dict(batch_size=256, epochs=0, data_file='cubic', cache_mix=2),
          img_gen_opt=dict(epochs=0, mode=2),
          casual_test_opt=dict())
