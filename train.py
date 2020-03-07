import os
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import opt_parser
from model.function.loss_fn import contrastive_loss, cosine_loss
from model.model import Model
import tb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load(dir, file_name):
    return torch.load(os.path.join(dir, file_name))


def collate_fn(batch):
    """

    :batch: 16 * 512 * (96*96+4)
    """
    batch = torch.stack(batch).float()
    assert batch.shape == (16, 512, 96 * 96 + 4)
    batch = batch[:, :, :-1]
    # (sizedim - size) / step + 1 = (512 - 200) // 20 + 1 =16
    batch = batch.unfold(1, 200, 20)  # 16 * 16 * (96*96+3) * 200
    # (200 - 4)//1 + 1 = 197
    batch = batch.unfold(3, 4, 1)  # 16 * 16 * (96*96+3) * 197 * 4
    batch = batch.permute(3, 0, 1, 4, 2)
    batch = batch.contiguous().view(197, 256, 4 * (96 * 96 + 3))

    return batch


def prepare_data_loader():
    data = load('data', 'car-racing.64')
    data_loader = DataLoader(dataset=data,
                             batch_size=16,
                             shuffle=True,
                             collate_fn=collate_fn,
                             drop_last=True)

    return data_loader

    # if l1 is not None:
    #     writer.add_scalar('cortex_loss', l1.item(), steps[0])
    #     writer.add_scalar('cosine_loss', l2.item(), steps[0])
    #     writer.add_scalar('hippocampus_loss', l3.item(), steps[0])
    #     writer.add_scalar('background_loss', l4.item(), steps[0])
    #
    # if steps[0] % TENSOR_BOARD_STEPS == 1:
    #     if enc_outputs is not None:
    #         writer.add_histogram('enc_outputs', enc_outputs, steps[0])
    #         writer.add_histogram('enc_outputs_span', enc_outputs.max(0)[0] - enc_outputs.min(0)[0], steps[0])
    #     if agg_outputs is not None:
    #         writer.add_histogram('agg_outputs', agg_outputs, steps[0])
    #     if att_outputs is not None:
    #         writer.add_histogram('att_outputs', att_outputs, steps[0])
    #     if weights is not None:
    #         writer.add_histogram('weights_max', weights.max(1)[0], steps[0])
    #
    #     model.eval()
    #     writer.add_histogram('temperature', model.hippocampus.temperature, steps[0])
    #     model.train()


def optimize(optimizers, enc_outputs, agg_outputs, att_outputs, memories, weights):
    cortex_loss, background_loss = contrastive_loss(enc_outputs, agg_outputs, memories, weights)
    optimizers[0].zero_grad()
    cortex_loss.backward(retain_graph=True)
    # background_loss.backward(retain_graph=True)
    optimizers[0].step()
    #
    hippocampus_loss = - cortex_loss
    optimizers[1].zero_grad()
    hippocampus_loss.backward(retain_graph=True)
    optimizers[1].step()


def train_batch(batch, model, optimizers, opt):
    model.cortex.reset()
    batch_size, frames, _ = batch.shape

    agg_outputs_preview, memories = None, None

    desc = '  - (Training)   '
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

        # inputs = batch[:, index + 200:index + 200 + 4:, :-1].float().contiguous().view(batch_size, -1)

        enc_outputs, agg_outputs, agg_outputs_preview = model.cortex(inputs, att_outputs)

        if att_outputs is not None:
            optimize(optimizers, enc_outputs, agg_outputs, att_outputs, memories[1], weights)

        if attention is not None:
            memories = (attention, enc_outputs)

        tb.histogram(
            # enc_outputs=enc_outputs,
            # agg_outputs=agg_outputs,
            # att_outputs=att_outputs,
            weights=weights,
            weights_max=None if weights is None else weights.max(1)[0])


def train(model, data_loader, optimizers, opt):
    for epoch in range(opt.epochs):
        desc = '  - (Training)   '
        for batch in tqdm(data_loader, mininterval=2, desc=desc, leave=False):
            train_batch(batch, model, optimizers, opt)
        # torch.save(model.state_dict(), f='model_state/epoch_0_' + str(epoch))


def main():
    global model
    opt = opt_parser.parse_opt()

    num_units_regions = [8]

    model = Model(num_units_regions, opt)

    # neuralI = NeuralInterface(dim_inputs=cortex.dim_outputs,
    #                           dim_hidden=opt.dim_hid_ni,
    #                           dim_outputs=3)

    optimizers = [optim.Adam([dict(params=model.cortex.encoder_parameters(), lr=1e-4),
                              dict(params=model.cortex.aggregator_parameters(), lr=1e-3)]),
                  optim.Adam((*model.hippocampus.parameters(), *model.memory.parameters()), lr=1e-3),
                  # optim.Adam(neuralI.parameters()),
                  ]

    data_loader = prepare_data_loader()

    train(model, data_loader, optimizers, opt)


if __name__ == '__main__':
    main()
