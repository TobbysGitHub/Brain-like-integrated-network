import os
import logging

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dirs import MODEL_DOMAIN_DIR, DATA_DIR, MODEL_RUNS_DIR
from model import opt_parser
from model.function.flip_grad import flip_grad
from model.function.loss_fn import contrastive_loss
from model.brain_like_model import Model
from tensor_board import tb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(dir, file_name):
    return torch.load(os.path.join(dir, file_name), map_location=device)


def monitor(model, loss):
    global early_stopped
    opt = model.opt
    if hasattr(opt, 'loss') and opt.loss <= loss:
        last_update = int(opt.state_dict.split('.')[-1])
        if tb.steps[0] - last_update > opt.early_stop:
            early_stopped = True
        return

    dir = '{}/{}'.format(MODEL_DOMAIN_DIR, model.extra_repr())
    if not os.path.exists(dir):
        os.makedirs(dir)

    f = '{}/model_state_dict.{}'.format(dir, tb.steps[0])
    torch.save(model.state_dict(), f=f)
    opt.loss = loss.item()
    opt.state_dict = f
    f = '{}/opt'.format(dir)
    torch.save(opt, f=f)


def collate_fn(batch):
    """

    :batch: batch_size * 512 * (96*96+4)
    """
    batch = torch.stack(batch).float()
    batch_size = batch.shape[0]
    assert batch.shape == (batch_size, 512, 96 * 96 + 4)
    # trim the first 13 frames for length alignment
    batch = batch[:, -499:, :-1]  # batch_size * 499 * (96 * 96 + 3)
    # (sizedim - size) / step + 1 = (499 - 259) // 16 + 1 =16
    batch = batch.unfold(dimension=1, size=259, step=16)  # batch_size * 16 * (96*96+3) * 259
    # (259 - 4)//1 + 1 = 256
    batch = batch.unfold(dimension=3, size=4, step=1)  # batch_size * 16 * (96*96+3) * 256 * 4
    batch = batch.permute(3, 0, 1, 4, 2)
    batch = batch.contiguous().view(256, batch_size * 16, 4 * (96 * 96 + 3))  # frames256 * parallels * dim_inputs

    return batch


def prepare_data_loader(batch_size, file='car-racing.64', shuffle=True, device=None):
    data = load(DATA_DIR, file).float().to(device)
    assert batch_size % 16 == 0
    data_loader = DataLoader(dataset=data,
                             batch_size=batch_size // 16,
                             shuffle=shuffle,
                             collate_fn=collate_fn,
                             drop_last=True)

    return data_loader


def optimize(optimizers, enc_outputs, agg_outputs, att_outputs, memories, weights):
    cortex_loss, background_loss = contrastive_loss(enc_outputs, agg_outputs, att_outputs, memories, weights)
    optimizers[0].zero_grad()
    cortex_loss.backward(retain_graph=True)
    # background_loss.backward(retain_graph=True)
    optimizers[0].step()
    #
    # hippocampus_loss = - cortex_loss
    # optimizers[1].zero_grad()
    # hippocampus_loss.backward(retain_graph=True)
    flip_grad(optimizers[1])
    optimizers[1].step()

    return background_loss


def train_batch(batch, model, optimizers, opt, epoch):
    model.cortex.reset()
    batch_size, frames, _ = batch.shape

    agg_outputs_preview, memories = None, None

    desc = '  - (Training epoch:{}/{})   '.format(epoch, opt.epochs)
    # when not None?
    # enc_outputs >>| agg_outputs_preview >>| attention >> agg_outputs >> memories >>| weights, att_outputs
    # for index in tqdm(range(frames - 3), mininterval=2, desc=desc, leave=None):
    for inputs in tqdm(batch, mininterval=2, desc=desc, leave=None):
        tb.steps[0] = 1 + tb.steps[0]
        if agg_outputs_preview is not None:
            if memories is not None:
                attention, weights, att_outputs = model.hippocampus(agg_outputs_preview.detach(), memories)
            else:
                attention = model.hippocampus(agg_outputs_preview.detach())
                weights, att_outputs = None, None
        else:
            attention, weights, att_outputs = (None,) * 3

        enc_outputs, agg_outputs, agg_outputs_preview = model.cortex(inputs, att_outputs)

        if att_outputs is not None:
            loss = optimize(optimizers, enc_outputs, agg_outputs, att_outputs, memories[1], weights)
        else:
            loss = np.inf

        if attention is not None:
            memories = (attention, enc_outputs)

        if tb.steps[0] > 1000 and tb.steps[0] % 10 == 0 and not loss == np.inf:
            # if True:
            monitor(model, loss)


def train(model, data_loader, optimizers, opt):
    for epoch in range(opt.epochs):
        for batch in data_loader:
            train_batch(batch, model, optimizers, opt, epoch)
            if early_stopped:
                break

        if early_stopped:
            break


def main():
    global model_name, early_stopped
    early_stopped = False
    opt = opt_parser.parse_opt()
    # opt = torch.load('{}/model__unit_n8_d8_@d16_@unit_d4_@mem256_@groups8_Mar10_21-00-02/opt')

    model = Model(opt).to(device)
    model_name = model.extra_repr()
    logging.warning(model)

    tb.creat_model_writer('{}/{}'.format(MODEL_RUNS_DIR, model.extra_repr()))

    # opt.state_dict = '{}/model__unit_n8_d8_@d16_@unit_d4_@mem256_@groups8_Mar10_21-00-02/model_state_dict.15150' \
    #     .format(MODEL_DOMAIN_DIR)

    if opt.state_dict is not None:
        model.load_state_dict(torch.load(opt.state_dict, map_location=device))
        tb.steps[0] = int(opt.state_dict.split('.')[-1])

    optimizers = [optim.Adam([dict(params=model.cortex.encoder_parameters(), lr=1e-4),
                              dict(params=model.cortex.aggregator_parameters(), lr=1e-3)]),
                  optim.Adam(model.hippocampus.parameters(), lr=1e-3),
                  ]

    data_loader = prepare_data_loader(batch_size=opt.batch_size)

    with torch.autograd.detect_anomaly():
        train(model, data_loader, optimizers, opt)


if __name__ == '__main__':
    main()
