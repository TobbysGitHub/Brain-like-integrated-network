import os

import torch
import torchvision.utils as utils
import torch.nn.functional as F
import numpy as np

from dirs.dirs import *
from interface.img_gen import opt_parser
from interface.img_gen.image_gen_net import ImageGenNet
from interface.img_gen.prepare_data import gen_batch
from model.brain_like_model import Model
from dataset.dataset import prepare_data_loader
from tensor_board import tb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainState:
    def __init__(self, steps):
        self.steps = steps
        self.epoch = 0
        self.batch = 0


def save(gen_net, dir):
    file = '{}/gen_net_state_dict'.format(dir)

    torch.save(gen_net.state_dict(), f=file)


def train_batch(gen_net, batch, optim, mode, state):
    assert mode in [0, 1]
    img, *inputs = batch
    state.steps += 1
    outputs = gen_net(inputs[mode])

    img = img.view(-1, 96, 96)
    outputs = outputs.view(-1, 96, 96)

    loss = F.mse_loss(img, outputs)

    optim.zero_grad()
    loss.backward()
    optim.step()

    tb.add_scalar(loss=loss.item())


def train(gen_net, model, optim, opt, model_opt, state):
    file = model_opt.data_file + '.train'
    data_loader = prepare_data_loader(batch_size=model_opt.batch_size,
                                      file=file,
                                      early_cuda=model_opt.early_cuda,
                                      shuffle=True)

    for epoch in range(opt.epochs):
        batch_gen = gen_batch(model, data_loader, opt.batch_size, state.epoch)

        for batch in batch_gen:
            train_batch(gen_net, batch, optim, opt.mode, state)
            state.batch += 1
        state.epoch += 1


def visualize(gen_net, model, opt, model_opt, dir, nrow=8, displays=32):
    file = model_opt.data_file + '.eval'
    data_loader = prepare_data_loader(batch_size=model_opt.batch_size, file=file,
                                      early_cuda=model_opt.early_cuda,
                                      shuffle=False)
    batch_gen = gen_batch(model, data_loader, batch_size=displays)

    for img, *inputs in batch_gen:
        break

    img_gen = gen_net(inputs[opt.mode])

    imgs = torch.stack([img, img_gen], dim=1) \
        .view(displays // 8, 8, 2, 1, 96, 96) \
        .transpose(2, 1) \
        .contiguous() \
        .view(-1, 1, 96, 96)

    file = '{}/{}'.format(dir, model_opt.data_file)
    torch.save(imgs, file)
    utils.save_image(imgs, file + '.png', nrow=nrow, normalize=True)
    grid_img = utils.make_grid(imgs, nrow=nrow, normalize=True)
    tb.writer.add_image(file, grid_img)


def eval(gen_net, model, opt, model_opt):
    file = model_opt.data_file + '.eval'
    data_loader = prepare_data_loader(batch_size=model_opt.batch_size, file=file,
                                      early_cuda=model_opt.early_cuda,
                                      shuffle=True)
    batch_gen = gen_batch(model, data_loader, opt.batch_size)
    loss = []

    for batch in batch_gen:
        img, *inputs = batch
        outputs = gen_net(inputs[opt.mode])

        img = img.view(-1, 96, 96)
        outputs = outputs.view(-1, 96, 96)

        loss.append(F.mse_loss(img, outputs).item())

    loss = np.mean(loss)

    tb.writer.add_text('eval_loss', str(loss))


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
    tb.creat_writer(steps_fn=lambda: state.steps,
                    log_dir='{}/{}/{}_mode{}'.format(INTERFACE_RUNS_DIR, opt.model_repr, model_opt.data_file, opt.mode))

    optim = torch.optim.Adam(gen_net.parameters(), lr=1e-3)

    train(gen_net, model, optim, opt, model_opt, state)
    dir = '{}/{}/{}_mode{}'.format(MODEL_DOMAIN_DIR, opt.model_repr, model_opt.data_file, opt.mode)
    if not os.path.exists(dir):
        os.makedirs(dir)
    save(gen_net, dir)
    with torch.no_grad():
        visualize(gen_net, model, opt=opt, model_opt=model_opt, dir=dir)
        eval(gen_net, model, opt=opt, model_opt=model_opt)


if __name__ == '__main__':
    main()
