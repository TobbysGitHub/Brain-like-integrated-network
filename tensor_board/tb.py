from torch.utils.tensorboard import SummaryWriter
import numpy as np


def log(s):
    return s % int(np.sqrt(s)) == 0


def creat_writer(steps_fn, log_dir):
    global steps, writer
    writer = SummaryWriter(log_dir=log_dir)
    steps = steps_fn


def histogram(**kwargs):
    if log(steps()):
        for name, data in kwargs.items():
            if data is None:
                continue
            writer.add_histogram(name, data, steps())


def add_scalar(**kwargs):
    if log(steps()):
        for name, data in kwargs.items():
            writer.add_scalar(name, data, steps())
