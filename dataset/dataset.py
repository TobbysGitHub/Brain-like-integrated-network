import os

import torch
from torch.utils.data import DataLoader, Dataset

from dirs.dirs import DATA_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    return batch.to(device)


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

    return iter_frame(batch.to(device))


def iter_frame(batch):
    """
    :param batch: 256 * batch_size * (96*96+4)
    """
    frames = []
    for i, frame in enumerate(batch):
        frames.append(frame)
        if len(frames) < 4:
            continue
        yield torch.cat(frames, dim=-1)  # batch_size * (4*(96*96+4))
        frames.pop(0)


class DataSet(Dataset):
    def __init__(self, file):
        def load(dir, file_name):
            return torch.load(os.path.join(dir, file_name))

        self.data = load(DATA_DIR, file).float()

    def __getitem__(self, item):
        index0 = item // 16
        index1 = item % 16 * 16

        return self.data[index0, index1:index1 + 256]  # 256 * (96*96+4)

    def __len__(self):
        return len(self.data) * 16


def collate_fn(batch):
    batch = torch.stack(batch, dim=1)  # 256 * batch_size
    return iter_frame(batch.to(device))


def prepare_data_loader(batch_size, file='car-racing.64', shuffle=True):
    data_loader = DataLoader(dataset=DataSet(file),
                             batch_size=batch_size,
                             shuffle=shuffle,
                             collate_fn=collate_fn,
                             # num_workers=3,
                             drop_last=True)

    # class Wrapper:
    #     def __init__(self, data_loader):
    #         self.data_loader = data_loader
    #
    #     def __iter__(self):
    #         return iter_frame(data_loader.__iter__())

    return data_loader
