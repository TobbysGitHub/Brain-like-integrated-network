from torch.utils.tensorboard import SummaryWriter

TENSOR_BOARD_STEPS = 10
writer = [None]

steps = [0]


def creat_model_writer(log_dir):
    writer[0] = SummaryWriter(log_dir=log_dir)


def histogram(**kwargs):
    if steps[0] % TENSOR_BOARD_STEPS == 0:
        for name, data in kwargs.items():
            if data is None:
                continue
            writer[0].add_histogram(name, data, steps[0])


def add_scalar(**kwargs):
    if steps[0] % TENSOR_BOARD_STEPS == 0:
        for name, data in kwargs.items():
            writer[0].add_scalar(name, data, steps[0])
