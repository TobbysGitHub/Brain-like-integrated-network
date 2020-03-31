import os

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from dirs.dirs import DATA_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_frame(batch):
    """
    :param batch: 256 * batch_size * (96*96+4)
    """
    frames = []
    for i, frame in enumerate(batch):
        if early_cuda_ == 2:
            frame = frame.to(device)
        frames.append(frame)
        if len(frames) < 4:
            continue
        yield torch.cat(frames, dim=-1)  # batch_size * (4*(96*96+4))
        frames.pop(0)


class UnfoldDataSet(Dataset):
    def __init__(self, file, size=256, step=16):
        dev = device if early_cuda_ == 0 else None

        def load(dir, file_name):
            return torch.load(os.path.join(dir, file_name), map_location=dev)

        self.data = load(DATA_DIR, file).float()
        self.size = size
        self.step = step
        self.folds = (self.data.shape[1] - size) // step + 1

    def __getitem__(self, item):
        i_episode = item // self.folds
        offset = item % self.folds * self.step

        item_data = self.data[i_episode, offset:offset + self.size]

        return item_data  # 256 * -1

    def __len__(self):
        return len(self.data) * self.folds


def collate_fn(batch):
    batch = torch.stack(batch, dim=1)  # 256 * batch_size
    if early_cuda_ == 1:
        batch = batch.to(device)
    return iter_frame(batch)


def prepare_data_loader(batch_size, file, early_cuda, shuffle=True):
    global early_cuda_
    early_cuda_ = early_cuda
    data_loader = DataLoader(dataset=UnfoldDataSet(file),
                             batch_size=batch_size,
                             shuffle=shuffle,
                             collate_fn=collate_fn,
                             drop_last=True)

    return data_loader
