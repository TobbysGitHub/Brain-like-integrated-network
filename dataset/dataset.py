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


class RoundSampler(Sampler):
    def __init__(self, data_source, rounds, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.rounds = rounds
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.data_source) // self.rounds

        if self.shuffle:
            index_list = torch.stack([torch.randperm(n) * self.rounds + i for i in range(self.rounds)],
                                     dim=-1).view(-1).tolist()
        else:
            index_list = torch.stack([torch.arange(n) * self.rounds + i for i in range(self.rounds)],
                                     dim=-1).view(-1).tolist()

        return iter(index_list)

    def __len__(self):
        return len(self.data_source)


class RotatedDataSet(Dataset):
    def __init__(self, file, rotations=None):
        dev = device if early_cuda_ == 0 else None

        def load(dir, file_name):
            return torch.load(os.path.join(dir, file_name), map_location=dev)

        self.data = load(DATA_DIR, file).float()
        if not rotations:
            self.rotations = [0]
        else:
            self.rotations = rotations

    def __getitem__(self, item):
        rotation = self.rotations[item % len(self.rotations)]
        item = item // len(self.rotations)
        i_episode = item // 16
        offset = item % 16 * 16

        item_data = self.data[i_episode, offset:offset + 256]

        item_data = item_data[:, :96 * 96].contiguous()
        if not rotation == 0:
            if rotation == 1:
                item_data = item_data \
                    .view(-1, 96, 96) \
                    .transpose(1, 2) \
                    .flip(dims=(-1,)) \
                    .contiguous() \
                    .view(-1, 96 * 96)
            elif rotation == 2:
                item_data = item_data \
                    .view(-1, 96, 96) \
                    .flip(dims=(1,)) \
                    .contiguous() \
                    .view(-1, 96 * 96)
            elif rotation == 3:
                item_data = item_data \
                    .view(-1, 96, 96) \
                    .transpose(1, 2) \
                    .contiguous() \
                    .view(-1, 96 * 96)
            else:
                raise ValueError()

        return item_data  # 256 * (96*96)

    def __len__(self):
        return len(self.data) * len(self.rotations) * 16


def collate_fn(batch):
    batch = torch.stack(batch, dim=1)  # 256 * batch_size
    if early_cuda_ == 1:
        batch = batch.to(device)
    return iter_frame(batch)


def prepare_data_loader(batch_size, file, rotations, early_cuda, shuffle=True):
    global early_cuda_
    early_cuda_ = early_cuda
    dataset = RotatedDataSet(file, rotations)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             sampler=RoundSampler(data_source=dataset,
                                                  rounds=len(dataset.rotations),
                                                  shuffle=shuffle),
                             collate_fn=collate_fn,
                             drop_last=True)

    return data_loader
