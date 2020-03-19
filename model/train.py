import os
import logging

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dirs.dirs import MODEL_DOMAIN_DIR, DATA_DIR, MODEL_RUNS_DIR
from model import opt_parser
from model.function.flip_grad import flip_grad
from model.function.loss_fn import contrastive_loss
from model.brain_like_model import Model
from tensor_board import tb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainingState:
    def __init__(self, steps):
        self.steps = steps
        self.min_loss = np.inf
        self.early_stop = False
        self.total = 0
        self.inc = 0

    def loss(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.inc = 0
        else:
            self.inc += 1

        self.total += 1
        if self.total > 10 and self.inc >= 4:
            self.early_stop = True


def load(dir, file_name):
    return torch.load(os.path.join(dir, file_name))


def save(model, steps):
    dir = '{}/{}'.format(MODEL_DOMAIN_DIR, model.extra_repr())
    if not os.path.exists(dir):
        os.makedirs(dir)

    f_model = '{}/model_state_dict.{}'.format(dir, steps)
    torch.save(model.state_dict(), f=f_model)

    opt = model.opt
    f_old = opt.state_dict
    opt.state_dict = f_model
    f_opt = '{}/opt'.format(dir)
    torch.save(model.opt, f=f_opt)
    if f_old is not None and os.path.dirname(f_old) == os.path.dirname(f_model):
        os.remove(f_old)


def collate_fn0(batch):
    """

    :batch: batch_size * 512 * (96*96+4)
    """
    batch = torch.stack(batch).float()
    batch_size = batch.shape[0]
    assert batch.shape == (batch_size, 512, 96 * 96 + 4)
    # trim the first 13 frames for length alignment
    batch = batch[:, -499:, :]  # batch_size * 499 * (96 * 96 + 4)
    # (sizedim - size) / step + 1 = (499 - 259) // 16 + 1 =16
    batch = batch.unfold(dimension=1, size=259, step=16)  # batch_size * 16 * (96*96+4) * 259
    # (259 - 4)//1 + 1 = 256
    batch = batch.unfold(dimension=3, size=4, step=1)  # batch_size * 16 * (96*96+4) * 256 * 4
    batch = batch.permute(3, 0, 1, 4, 2)
    batch = batch.contiguous().view(256, batch_size * 16, 4 * (96 * 96 + 4))  # frames256 * parallels * dim_inputs
    return batch


def collate_fn1(batch):
    """

    :batch: batch_size * 512 * (96*96+4)
    """
    batch = torch.stack(batch).float()
    batch_size = batch.shape[0]
    assert batch.shape == (batch_size, 512, 96 * 96 + 4)
    # trim the first 13 frames for length alignment
    batch = batch[:, -499:, :]  # batch_size * 499 * (96 * 96 + 4)
    # (sizedim - size) / step + 1 = (499 - 259) // 16 + 1 =16
    batch = batch.unfold(dimension=1, size=259, step=16)  # batch_size * 16 * (96*96+4) * 259
    # # (259 - 4)//1 + 1 = 256
    # batch = batch.unfold(dimension=3, size=4, step=1)  # batch_size * 16 * (96*96+4) * 256 * 4
    batch = batch.permute(3, 0, 1, 2)
    batch = batch.contiguous().view(259, batch_size * 16, (96 * 96 + 4))  # frames259 * parallels * dim_inputs//4

    return iter_frame(batch)


def iter_frame(batch):
    frames = []
    for i, frame in enumerate(batch):
        frames.append(frame)
        if len(frames) < 4:
            continue
        yield torch.cat(frames, dim=-1)
        frames.pop(0)


def prepare_data_loader(batch_size, file='car-racing.64', shuffle=True):
    data = load(DATA_DIR, file).float().to(device)
    assert batch_size % 16 == 0
    data_loader = DataLoader(dataset=data,
                             batch_size=batch_size // 16,
                             shuffle=shuffle,
                             collate_fn=collate_fn0,
                             drop_last=True)

    return data_loader


def optimize(optimizers, enc_outputs, agg_outputs, att_outputs, mem_outputs, weights):
    cortex_loss, background_loss = contrastive_loss(enc_outputs, agg_outputs, att_outputs, mem_outputs, weights)
    optimizers[0].zero_grad()
    optimizers[1].zero_grad()
    cortex_loss.backward(retain_graph=True)
    # background_loss.backward(retain_graph=True)
    optimizers[0].step()
    # hippocampus_loss = - cortex_loss
    # hippocampus_loss.backward(retain_graph=True)
    flip_grad(optimizers[1])
    optimizers[1].step()

    return background_loss


def train_batch(batch, model, optimizers, state):
    sum_loss = 0
    counter = 0

    model.reset()
    desc = '  - (Training frames)'
    for inputs in tqdm(batch, mininterval=2, desc=desc, leave=False, total=256):
        state.steps += 1
        results = model(inputs)
        if results is None:
            continue

        (enc_outputs, agg_outputs, att_outputs, mem_outputs), attention, weights = results
        loss = optimize(optimizers, enc_outputs, agg_outputs, att_outputs, mem_outputs, weights)
        sum_loss += loss.item()
        counter += 1

    state.loss(sum_loss / counter)
    save(model, state.steps)


def train(model, data_loader, optimizers, epochs, state):
    desc = '  - (Training epoch)'
    for epoch in tqdm(range(epochs), mininterval=2, desc=desc, leave=True):
        desc = '  - (Training batch)'
        for batch in tqdm(data_loader, mininterval=2, desc=desc, leave=False):
            train_batch(batch, model, optimizers, state)
            if state.early_stop:
                return

        if state.early_stop:
            return


def main():
    opt = opt_parser.parse_opt()
    # opt = torch.load('{}/model__unit_n8_d8_@d16_@unit_d4_@mem256_@groups8_Mar10_21-00-02/opt')
    model = Model(opt).to(device)
    print(model)

    # opt.state_dict = '{}/model__unit_n8_d8_@d16_@unit_d4_@mem256_@groups8_Mar10_21-00-02/model_state_dict.15150' \
    #     .format(MODEL_DOMAIN_DIR)

    if opt.state_dict is not None:
        model.load_state_dict(torch.load(opt.state_dict, map_location=device))
        state = TrainingState(int(opt.state_dict.split('.')[-1]))
    else:
        state = TrainingState(0)

    tb.creat_writer(steps_fn=lambda: state.steps, log_dir='{}/{}'.format(MODEL_RUNS_DIR, model.extra_repr()))

    optimizers = [optim.Adam([dict(params=model.encoder.parameters(), lr=1e-4),
                              dict(params=model.aggregator.parameters(), lr=1e-3)]),
                  optim.Adam(model.hippocampus.parameters(), lr=1e-3),
                  ]

    data_loader = prepare_data_loader(batch_size=opt.batch_size)

    save(model, state.steps)
    train(model, data_loader, optimizers, opt.epochs, state)

    return model.extra_repr()


if __name__ == '__main__':
    main()
