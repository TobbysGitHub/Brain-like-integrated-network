import torch
from torchvision import utils
import torch.nn.functional as F
import numpy as np

from dataset.dataset import prepare_data_loader
from dirs.dirs import *
from interface.casual_test import opt_parser
from interface.img_gen.image_gen_net import ImageGenNet
from model.brain_like_model import Model
from tensor_board import tb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class State:
    def __init__(self, save_dir):
        self.save_dir = save_dir


def creat_model(model_opt, mode):
    model = Model(model_opt).to(device)
    model.load_state_dict(torch.load(model_opt.state_dict, map_location=device))
    return model


def creat_net(model, save_dir):
    dim_inputs = model.encoder.dim_outputs
    net = ImageGenNet(dim_inputs).to(device)
    state_dict_dir = '{}/gen_net_state_dict'.format(save_dir)
    net.load_state_dict(torch.load(state_dict_dir, map_location=device))
    return net


def prepare_tensorboard(state, opt, model_opt):
    tb.creat_writer(steps_fn=None,
                    log_dir='{}/{}/{}_mode{}'.format(INTERFACE_RUNS_DIR, opt.model_repr, model_opt.data_file, opt.mode))


def predict(gen_net, model, model_opt, state):
    file = model_opt.data_file + '.eval'
    data_loader = prepare_data_loader(batch_size=model_opt.batch_size, file=file,
                                      early_cuda=model_opt.early_cuda,
                                      shuffle=False)
    loss = []
    for batch in data_loader:
        for n, inputs in enumerate(batch):
            if n < 64:
                model(inputs)
                continue
            elif n < 71:
                model(None)
            else:
                (enc_outputs, agg_outputs, att_outputs, mem_outputs), attention, weights = model(None)
                img_gen = gen_net(agg_outputs)
                img = inputs[:, -96 * 96:].view_as(img_gen)

                loss.append(F.mse_loss(img_gen, img).item())
                break
    tb.writer.add_text('ploss', str(np.mean(loss)))

    imgs = []
    imgs_gen = []
    for batch in data_loader:
        for n, inputs in enumerate(batch):
            if n < 64:
                model(inputs)
            elif n < 80:
                (enc_outputs, agg_outputs, att_outputs, mem_outputs), attention, weights = model(None)
                img_gen = gen_net(agg_outputs[:16])
                imgs_gen.append(img_gen)
                imgs.append(inputs[:16, -96 * 96:])
            else:
                break
        break

    imgs_gen = torch.stack(imgs_gen, dim=1).view(16, 16, 1, 96, 96)  # 16_batch * 16_frame * 1,96,96
    imgs = torch.stack(imgs, dim=1).view(16, 16, 1, 96, 96)

    img_cat = torch.stack([imgs, imgs_gen], dim=1)  # 16_batch * 2 * 16_frame * 1,96,96
    img_cat = img_cat.view(-1, 1, 96, 96)

    file = '{}/{}'.format(state.save_dir, 'prediction.png')
    utils.save_image(img_cat, file, nrow=16, normalize=True)
    grid_img = utils.make_grid(img_cat, nrow=16, normalize=True)
    tb.writer.add_image('prediction', grid_img)


def main():
    opt = opt_parser.parse_opt()
    model_opt = torch.load('{}/{}/opt'.format(MODEL_DOMAIN_DIR, opt.model_repr))
    save_dir = '{}/{}/{}_mode{}'.format(MODEL_DOMAIN_DIR, opt.model_repr, model_opt.data_file, opt.mode)

    model = creat_model(model_opt, opt.mode)
    gen_net = creat_net(model, save_dir)

    state = State(save_dir)
    prepare_tensorboard(state, opt, model_opt)

    with torch.no_grad():
        predict(gen_net, model, model_opt=model_opt, state=state)


if __name__ == '__main__':
    import sys

    sys.argv = ['', '--model_repr', 'Apr02_10-28-34_model__unit_n8_d2_@d16_@unit_d2_@mem256_@groups8_cubic']
    main()
