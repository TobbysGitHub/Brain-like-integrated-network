import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt

from interface.img_gen.image_gen_net import ImageGenNet
from tensor_board import tb


def show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


class MyDataSet(Dataset):
    def __init__(self):
        # prefix = 'data/car-racing-untrained-'
        prefix = 'data/car-racing-'
        self.data_img = torch.load(prefix + 'img.8192')
        self.data_attention = torch.load(prefix + 'attention.8192')
        self.data_outputs = torch.load(prefix + 'outputs.8192')

    def __getitem__(self, item):
        return self.data_img[item], self.data_attention[item], self.data_outputs[item]

    def __len__(self):
        return len(self.data_img)


def prepare_data_loader():
    return DataLoader(
        dataset=MyDataSet(),
        batch_size=64,
        shuffle=True
    )


def train_epoch(model, data_loader, optim, epoch):
    desc = 'epoch:{}'.format(epoch)
    for img, attention, _ in tqdm(data_loader, desc=desc, mininterval=2, leave=False):
        outputs = model(attention)

        loss = F.mse_loss(img, outputs.view_as(img))

        optim.zero_grad()
        loss.backward()
        optim.step()

        tb.add_scalar(loss=loss.item())
        tb.steps[0] = tb.steps[0] + 1

        if tb.steps[0] % 10 == 0:
            show(outputs[0].view(96, 96).detach().numpy())


def train(model, data_loader, optim, epochs):
    for epoch in range(epochs):
        train_epoch(model, data_loader, optim, epoch)


def main():
    net = ImageGenNet(16)
    data_loader = prepare_data_loader()
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    train(net, data_loader, optim, 4)


if __name__ == '__main__':
    main()
