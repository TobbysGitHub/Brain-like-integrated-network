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
    def __init__(self, steps, gen_net, save_dir):
        self.steps = steps
        self.gen_net = gen_net
        self.save_dir = save_dir
        self.epoch = 0
        self.batch = 0
        self.early_stop = False
        self.best_loss = np.inf
        self.best_epoch = 0
        self.state_dict = '{}/gen_net_state_dict'.format(self.save_dir)

    def monitor(self, eval_loss):

        if eval_loss < self.best_loss:
            self.best_epoch = self.epoch
            self.best_loss = eval_loss

        elif self.epoch - self.best_epoch > 5:
            self.early_stop = True
            tb.writer.add_text('loss', str(self.best_loss))


def save(state):
    if os.path.exists(state.state_dict):
        os.remove(state.state_dict)
    torch.save(state.gen_net.state_dict(), f=state.state_dict)


def train_epoch(gen_net, data, optim, mode, state):
    for batch in data:
        train_batch(gen_net, batch, optim, mode, state)
        state.batch += 1


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


def eval_epoch(gen_net, data, mode, state):
    loss_list = []
    for batch in data:
        eval_batch(gen_net, batch, mode, loss_list)
    loss = np.mean(loss_list)
    tb.writer.add_scalar('eval_loss', loss, state.epoch)
    return loss


def eval_batch(gen_net, batch, mode, loss_list):
    img, *inputs = batch
    outputs = gen_net(inputs[mode])

    img = img.view(-1, 96, 96)
    outputs = outputs.view(-1, 96, 96)

    loss_list.append(F.mse_loss(img, outputs).item())


def train(gen_net, model, optim, train_data_loader, eval_data_loader, opt, state):
    for _ in range(opt.epochs):
        train_data = gen_batch(model, train_data_loader, opt.batch_size)
        train_epoch(gen_net, train_data, optim, opt.mode, state)
        state.epoch += 1

        eval_data = gen_batch(model, eval_data_loader, opt.batch_size)
        with torch.no_grad():
            eval_loss = eval_epoch(gen_net, eval_data, opt.mode, state)
        state.monitor(eval_loss)
        if state.best_epoch == state.epoch:
            save(state)
        if state.early_stop:
            return


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


def creat_model(model_opt):
    model = Model(model_opt).to(device)
    model.load_state_dict(torch.load(model_opt.state_dict, map_location=device))
    return model


def creat_net(model, mode):
    if mode == 0:
        dim_inputs = model.dim_attention_global
    else:
        dim_inputs = model.encoder.dim_outputs
    return ImageGenNet(dim_inputs).to(device)


def prepare_tensorboard(state, opt, model_opt):
    tb.creat_writer(steps_fn=lambda: state.steps,
                    log_dir='{}/{}/{}_mode{}'.format(INTERFACE_RUNS_DIR, opt.model_repr, model_opt.data_file, opt.mode))


def main():
    opt = opt_parser.parse_opt()
    model_opt = torch.load('{}/{}/opt'.format(MODEL_DOMAIN_DIR, opt.model_repr))
    model = creat_model(model_opt)
    gen_net = creat_net(model, opt.mode)
    optim = torch.optim.Adam(gen_net.parameters(), lr=1e-3)

    save_dir = '{}/{}/{}_mode{}'.format(MODEL_DOMAIN_DIR, opt.model_repr, model_opt.data_file, opt.mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    state = TrainState(0, gen_net, save_dir)
    prepare_tensorboard(state, opt, model_opt)

    train_data_loader = prepare_data_loader(batch_size=model_opt.batch_size,
                                            file=model_opt.data_file + '.train',
                                            early_cuda=model_opt.early_cuda,
                                            shuffle=True)

    eval_data_loader = prepare_data_loader(batch_size=model_opt.batch_size,
                                           file=model_opt.data_file + '.eval',
                                           early_cuda=model_opt.early_cuda,
                                           shuffle=True)

    save(state)
    train(gen_net, model, optim, train_data_loader, eval_data_loader, opt, state)

    with torch.no_grad():
        gen_net.load_state_dict(torch.load(state.state_dict, map_location=device))
        visualize(gen_net, model, opt=opt, model_opt=model_opt, dir=save_dir)


if __name__ == '__main__':
    main()
