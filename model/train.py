import os
import logging

import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm

from dataset.dataset import prepare_data_loader
from dirs.delete_data import delete_data
from dirs.dirs import MODEL_DOMAIN_DIR, MODEL_RUNS_DIR
from model import opt_parser
from model.function.flip_grad import flip_grad
from model.function.loss_fn import contrastive_loss
from model.brain_like_model import Model
from tensor_board import tb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONTROL_GROUP = False


class TrainingState:
    def __init__(self, steps, early_stop_steps):
        self.steps = steps
        self.min_loss = np.inf
        self.early_stop_steps = early_stop_steps
        self.early_stop = False
        self.total = 0
        self.inc = 0

        self.epoch = 0
        self.batch = 0
        self.early_stop = False

    def monitor(self, loss):
        self.total += 1
        if loss < self.min_loss:
            self.min_loss = loss
            self.inc = 0
        else:
            self.inc += 1
            if self.total > 10 and self.inc >= self.early_stop_steps:
                self.early_stop = True


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
    if f_old is not None \
            and not f_old == f_model \
            and os.path.dirname(f_old) == os.path.dirname(f_model):
        os.remove(f_old)


def optimize(optimizers, enc_outputs, agg_outputs, att_outputs, mem_outputs, weights):
    cortex_loss, background_loss, pre_train_loss = contrastive_loss(enc_outputs, agg_outputs, att_outputs,
                                                                    mem_outputs,
                                                                    weights)
    for opti in optimizers:
        opti.zero_grad()
    #
    if pre_train_loss > 0.0:
        # experience shows this can accelerate training
        pre_train_loss.backward(retain_graph=True)
        optimizers[0].step()
        optimizers[1].step()
        return background_loss

    if not CONTROL_GROUP:
        cortex_loss.backward(retain_graph=True)
    else:
        background_loss.backward(retain_graph=True)
    optimizers[0].step()
    optimizers[1].step()
    # hippocampus_loss = - cortex_loss
    # hippocampus_loss.backward(retain_graph=True)
    flip_grad(optimizers[2])
    optimizers[2].step()

    return background_loss


def train_batch(batch, model, optimizers, state):
    sum_loss = 0
    counter = 0

    model.reset()
    desc = '  - Frame'
    for inputs in tqdm(batch, mininterval=2, desc=desc, leave=False, total=256):
        state.steps += 1
        results = model(inputs)
        if results is None:
            continue

        (enc_outputs, agg_outputs, att_outputs, mem_outputs), attention, weights, global_attention = results
        loss = optimize(optimizers, enc_outputs, agg_outputs, att_outputs, mem_outputs, weights)
        sum_loss += loss.item()
        counter += 1

    state.monitor(sum_loss / counter)


def train(model, data_loader, optimizers, epochs, state):
    for _ in range(epochs):
        for batch in data_loader:
            train_batch(batch, model, optimizers, state)
            if state.early_stop:
                return
            state.batch += 1

        if state.early_stop:
            return
        state.epoch += 1


def main():
    opt = opt_parser.parse_opt()
    # opt = torch.load('{}/model__unit_n8_d8_@d16_@unit_d4_@mem256_@groups8_Mar10_21-00-02/opt')
    model = Model(opt).to(device)
    print(model)

    # opt.state_dict = '{}/model__unit_n8_d8_@d16_@unit_d4_@mem256_@groups8_Mar10_21-00-02/model_state_dict.15150' \
    #     .format(MODEL_DOMAIN_DIR)
    try:
        if opt.state_dict is not None:
            model.load_state_dict(torch.load(opt.state_dict, map_location=device))
        state = TrainingState(0, opt.early_stop)

        tb.creat_writer(steps_fn=lambda: state.steps, log_dir='{}/{}'.format(MODEL_RUNS_DIR, model.extra_repr()))

        optimizers = [optim.Adam(model.encoder.parameters(), lr=1e-4),
                      optim.Adam(model.aggregator.parameters(), lr=1e-3),
                      optim.Adam(model.hippocampus.parameters(), lr=1e-3),
                      ]

        file = opt.data_file + '.train'
        data_loader = prepare_data_loader(batch_size=opt.batch_size, file=file,
                                          early_cuda=opt.early_cuda)

        save(model, state.steps)
        train(model, data_loader, optimizers, opt.epochs, state)
        save(model, state.steps)
    except BaseException as e:
        delete_data(model.extra_repr())
        print('failed !!!')
        raise e

    return model.extra_repr()


if __name__ == '__main__':
    main()
