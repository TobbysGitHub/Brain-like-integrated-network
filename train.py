import os
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import opt_parser
from model.function.loss_fn import contrastive_loss, cosine_loss
from model.model import Model
from tb import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TENSOR_BOARD_STEPS = 10


def load(dir, file_name):
    return torch.load(os.path.join(dir, file_name))


# def collate_fn(batch):
#     return torch.stack(batch, dim=1)


def prepare_data_loader(batch_size):
    data = load('data', 'car-racing.64')
    data_loader = DataLoader(dataset=data,
                             batch_size=batch_size,
                             shuffle=True,
                             # collate_fn=collate_fn,
                             drop_last=True)

    return data_loader


def optimize(optimizers, enc_outputs, agg_outputs, att_outputs, memories, weights):
    if att_outputs is None:
        return (None,) * 4

    cortex_loss, background_loss = contrastive_loss(enc_outputs, agg_outputs, memories, weights)
    optimizers[0].zero_grad()
    # cortex_loss.backward(retain_graph=True)
    background_loss.backward(retain_graph=True)
    # nn.utils.clip_grad_norm_()
    optimizers[0].step()

    cos_loss = cosine_loss(enc_outputs, att_outputs)
    hippocampus_loss = - cortex_loss
    # hippocampus_loss = cos_loss - cortex_loss
    # optimizers[1].zero_grad()
    # hippocampus_loss.backward(retain_graph=True)
    # optimizers[1].step()

    return cortex_loss, cos_loss, hippocampus_loss, background_loss


def tensor_board_record(enc_outputs, agg_outputs, att_outputs, weights, l1, l2, l3, l4):
    if l1 is not None:
        writer.add_scalar('cortex_loss', l1.item(), steps[0])
        writer.add_scalar('cosine_loss', l2.item(), steps[0])
        writer.add_scalar('hippocampus_loss', l3.item(), steps[0])
        writer.add_scalar('background_loss', l4.item(), steps[0])

    if steps[0] % TENSOR_BOARD_STEPS == 1:
        model.eval()
        if enc_outputs is not None:
            writer.add_histogram('enc_outputs', enc_outputs, steps[0])
            writer.add_histogram('enc_outputs_span', enc_outputs.max(0)[0] - enc_outputs.min(0)[0], steps[0])
        if agg_outputs is not None:
            writer.add_histogram('agg_outputs', agg_outputs, steps[0])
        if att_outputs is not None:
            writer.add_histogram('att_outputs', att_outputs, steps[0])
        if weights is not None:
            writer.add_histogram('weights_max', weights.max(1)[0], steps[0])

        writer.add_histogram('temperature', model.memory.temperature, steps[0])
        model.train()


def train_batch(batch, model, optimizers, opt):
    batch_size, frames, _ = batch.shape

    attention, att_outputs, agg_outputs_preview, weights, memories = (None,) * 5

    model.cortex.reset()

    desc = '  - (Training)   '
    for index in tqdm(range(frames - 3), mininterval=2, desc=desc, leave=None):
        steps[0] = 1 + steps[0]
        if agg_outputs_preview is not None:
            attention = model.hippocampus(agg_outputs_preview.detach())
            if model.memory.loaded and attention is not None:
                att_outputs, weights, memories = model.memory(attention)

        inputs = batch[:, index + 200:index + 200 + 4:, :-1].float().contiguous().view(batch_size, -1)

        # 这里要将 att_outputs 梯度隔绝
        # gradient isolation on att_outputs
        enc_outputs, agg_outputs, agg_outputs_preview = model.cortex(inputs,
                                                                     None if att_outputs is None else att_outputs.detach())

        l1, l2, l3, l4 = optimize(optimizers, enc_outputs, agg_outputs, att_outputs, memories, weights)

        tensor_board_record(enc_outputs, agg_outputs, att_outputs, weights, l1, l2, l3, l4)

        if index % opt.mem_interval == 0:
            model.memory.load(attention, enc_outputs)


def train(model, data_loader, optimizers, opt):
    for epoch in range(opt.epochs):
        desc = '  - (Training)   '
        for batch in tqdm(data_loader, mininterval=2, desc=desc, leave=False):
            train_batch(batch, model, optimizers, opt)
        torch.save(model.state_dict(), f='model_state/epoch_' + str(epoch))


def main():
    global model
    opt = opt_parser.parse_opt()

    num_units_regions = [32]

    model = Model(num_units_regions, opt)


    # neuralI = NeuralInterface(dim_inputs=cortex.dim_outputs,
    #                           dim_hidden=opt.dim_hid_ni,
    #                           dim_outputs=3)

    optimizers = [optim.Adam([dict(params=model.cortex.encoder_parameters(), lr=1e-4),
                              dict(params=model.cortex.aggregator_parameters(), lr=1e-3)]),
                  optim.Adam((*model.hippocampus.parameters(), *model.memory.parameters()), lr=1e-4),
                  # optim.Adam(neuralI.parameters()),
                  ]

    data_loader = prepare_data_loader(batch_size=opt.batch_size)

    train(model, data_loader, optimizers, opt)


if __name__ == '__main__':
    main()
