from torch.utils.tensorboard import SummaryWriter

TENSOR_BOARD_STEPS = 1
writer = SummaryWriter()

steps = [0]
best_loss = [1000]


def histogram(**kwargs):
    if steps[0] % TENSOR_BOARD_STEPS == 0:
        for name, data in kwargs.items():
            if data is None:
                continue
            writer.add_histogram(name, data, steps[0])


def add_scalar(**kwargs):
    for name, data in kwargs.items():
        writer.add_scalar(name, data, steps[0])
