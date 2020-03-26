import torch
import torchvision.utils as utils
import torch.nn.functional as F
import numpy as np

from dirs.dirs import *
from interface.img_gen import opt_parser
from interface.img_gen.image_gen_net import ImageGenNet
from interface.img_gen.prepare.prepare_data import gen_batch
from model.brain_like_model import Model
from dataset.dataset import prepare_data_loader
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

    img = img.view(-1, 96, 96)
    outputs = outputs.view(-1, 96, 96)

    loss = F.mse_loss(img, outputs)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if mode == 0:
        tb.add_scalar(loss=loss.item())
    else:
        tb.add_scalar(loss1=loss.item())


def train(gen_net, model, optim, opt, model_opt, state):
    file = model_opt.data_file + '.train'
    data_loader = prepare_data_loader(batch_size=model_opt.batch_size,
                                      file=file,
                                      rotations=model_opt.rotations,
                                      shuffle=True)

    for epoch in range(opt.epochs):
        batch_gen = gen_batch(model, data_loader, opt.batch_size)

        for batch in batch_gen:
            train_batch(gen_net, batch, optim, opt.mode, state)


def visualize(gen_net, model, opt, model_opt, dir, nrow=8, displays=32):
    file = model_opt.data_file + '.eval'
    data_loader = prepare_data_loader(batch_size=model_opt.batch_size, file=file,
                                      rotations=model_opt.rotations,
                                      shuffle=False)
    num_rotations = len(model_opt.rotations)
    num_max = displays * num_rotations
    batch_gen = gen_batch(model, data_loader, batch_size=num_max)

    for img, *inputs in batch_gen:
        break

    img_gen = gen_net(inputs[opt.mode])

    for i in range(num_rotations):
        img_i = img[i::num_rotations]
        img_gen_i = img_gen[i::num_rotations]

        imgs = torch.stack([img_i, img_gen_i], dim=1).view(-1, 1, 96, 96) \
            .expand(-1, 3, -1, -1) \
            .view(displays // 8, 8, 2, 3, 96, 96) \
            .transpose(2, 1) \
            .contiguous() \
            .view(-1, 3, 96, 96)

        file = '{}/{}{}_{}'.format(dir, model_opt.data_file, str(opt.mode), model_opt.rotations[i])
        torch.save(imgs, file)
        utils.save_image(imgs, file + '.png', nrow=nrow, normalize=True)
        grid_img = utils.make_grid(imgs, nrow=nrow, normalize=True)
        tb.writer.add_image(file, grid_img)


def eval(gen_net, model, opt, model_opt):
    file = model_opt.data_file + '.eval'
    data_loader = prepare_data_loader(batch_size=model_opt.batch_size, file=file,
                                      rotations=model_opt.rotations,
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

    if opt.mode == 0:
        tb.writer.add_text('eval_loss', str(loss))
    else:
        tb.writer.add_text('eval_loss1', str(loss))


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
        dir = '{}/{}/'.format(MODEL_DOMAIN_DIR, opt.model_repr)
        visualize(gen_net, model, opt=opt, model_opt=model_opt, dir=dir)
        # file='{}/{}/image_gen{}'.format(MODEL_DOMAIN_DIR, opt.model_repr, opt.mode))
        eval(gen_net, model, opt=opt, model_opt=model_opt)


if __name__ == '__main__':
    main()
