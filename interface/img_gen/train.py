import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as utils
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt

from dirs.dirs import MODEL_DOMAIN_DIR, INTERFACE_RUNS_DIR, DATA_DIR
from interface.img_gen import opt_parser
from interface.img_gen.image_gen_net import ImageGenNet
from interface.img_gen.prepare.prepare_data import gen_batch
from model.brain_like_model import Model
from model.train import prepare_data_loader
from tensor_board import tb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainState:
    def __init__(self, steps):
        self.steps = steps


def train_batch(gen_net, batch, optim, mode, state):
    assert mode in [0, 1]
    img, *inputs = batch
    state.steps += 1
    outputs = gen_net(inputs[mode])
    loss = F.mse_loss(img, outputs.view_as(img))

    optim.zero_grad()
    loss.backward()
    optim.step()

    if mode == 0:
        tb.add_scalar(loss0=loss.item())
    else:
        tb.add_scalar(loss1=loss.item())


def train(gen_net, model, optim, opt, model_opt, state):
    data_loader = prepare_data_loader(batch_size=model_opt.batch_size, file='car-racing.32', shuffle=True)

    for epoch in tqdm(range(opt.epochs), mininterval=2, leave=True):
        batch_gen = gen_batch(model, data_loader, opt.batch_size)

        for batch in batch_gen:
            train_batch(gen_net, batch, optim, opt.mode, state)


def visualize(gen_net, model, model_opt, mode, file, nrow=8):
    data_loader = prepare_data_loader(batch_size=model_opt.batch_size, file='car-racing.16', shuffle=False)
    batch_gen = gen_batch(model, data_loader, 32)

    img, *inputs = next(batch_gen.__iter__())
    img_gen = gen_net(inputs[mode])

    num = img.shape[0]
    img, img_gen = img.view(num, -1), img_gen.view(num, -1)

    img = torch.stack([img, img_gen], dim=1).view(-1, 1, 96, 96) \
        .expand(-1, 3, -1, -1) \
        .view(num // 8, 8, 2, 3, 96, 96) \
        .transpose(2, 1) \
        .contiguous() \
        .view(-1, 3, 96, 96)

    utils.save_image(img, file, nrow=nrow, normalize=True, scale_each=True)


def main():
    opt = opt_parser.parse_opt()

    model_opt = torch.load('{}/{}/opt'.format(MODEL_DOMAIN_DIR, opt.model_repr))
    model = Model(model_opt).to(device)
    model.load_state_dict(torch.load(model_opt.state_dict, map_location=device))

    if opt.mode == 0:
        dim_inputs = model_opt.dim_attention_global
    else:
        dim_inputs = model.encoder.dim_outputs
    gen_net = ImageGenNet(dim_inputs).to(device)
    state = TrainState(0)
    tb.creat_writer(steps_fn=lambda: state.steps, log_dir='{}/{}'.format(INTERFACE_RUNS_DIR, opt.model_repr))

    optim = torch.optim.Adam(gen_net.parameters(), lr=1e-3)

    train(gen_net, model, optim, opt, model_opt, state)

    with torch.no_grad():
        visualize(gen_net, model, model_opt=model_opt, mode=opt.mode,
                  file='{}/{}/image_gen{}.png'.format(MODEL_DOMAIN_DIR, opt.model_repr, opt.mode))


if __name__ == '__main__':
    main()
