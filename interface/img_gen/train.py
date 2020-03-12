import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as utils
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt

from dirs import MODEL_DOMAIN_DIR, INTERFACE_RUNS_DIR
from interface.img_gen import opt_parser
from interface.img_gen.image_gen_net import ImageGenNet
from tensor_board import tb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


class MyDataSet(Dataset):
    def __init__(self, opt):
        # prefix = 'data/car-racing-untrained-'
        file = '{}/{}/car-racing-{}.{}'.format(MODEL_DOMAIN_DIR, opt.model_repr, '{}', opt.samples)
        self.data_img = torch.load(file.format('img')).to(device)
        self.data_attention = torch.load(file.format('attention')).to(device)
        self.data_outputs = torch.load(file.format('outputs')).to(device)

    def __getitem__(self, item):
        return self.data_img[item], self.data_attention[item], self.data_outputs[item]

    def __len__(self):
        return len(self.data_img)


def prepare_data_loader(opt, batch_size=None, shuffle=True):
    if batch_size is None:
        batch_size = opt.batch_size

    return DataLoader(
        dataset=MyDataSet(opt),
        batch_size=batch_size,
        shuffle=shuffle
    )


def train_epoch(model, data_loader, optim, epoch, mode):
    assert mode in [0, 1]
    desc = '  - (Training epoch:{})   '.format(epoch)
    for img, *inputs in tqdm(data_loader, desc=desc, mininterval=2, leave=False):
        outputs = model(inputs[mode])
        loss = F.mse_loss(img, outputs.view_as(img))

        optim.zero_grad()
        loss.backward()
        optim.step()

        if mode == 0:
            tb.add_scalar(loss0=loss.item())
        else:
            tb.add_scalar(loss1=loss.item())

        tb.steps[0] = tb.steps[0] + 1

        # if tb.steps[0] % 20 == 0:
        #     show(outputs[0].view(96, 96).detach().numpy())


def train(model, data_loader, optim, epochs, mode):
    for epoch in range(epochs):
        train_epoch(model, data_loader, optim, epoch, mode)


def visualize(model, data_loader, mode, file, nrow=8):
    img, *inputs = next(data_loader.__iter__())
    img_gen = model(inputs[mode])

    num = img.shape[0]
    img, img_gen = img.view(num, -1), img_gen.view(num, -1)

    img = torch.stack([img, img_gen], dim=1).view(-1, 1, 96, 96) \
        .expand(-1, 3, -1, -1) \
        .view(num // 8, 8, 2, 3, 96, 96) \
        .transpose(2, 1) \
        .contiguous() \
        .view(-1, 3, 96, 96)

    utils.save_image(img, filename=file, nrow=nrow, normalize=True, scale_each=True)


def main():
    opt = opt_parser.parse_opt()
    opt.epochs = 1
    opt.mode = 1
    tb.creat_model_writer(log_dir='{}/{}'.format(INTERFACE_RUNS_DIR, opt.model_repr))

    model_opt = torch.load('{}/{}/opt'.format(MODEL_DOMAIN_DIR, opt.model_repr))
    if opt.mode == 0:
        dim_inputs = model_opt.dim_attention_global
    else:
        dim_inputs = sum(model_opt.num_units_regions) * model_opt.dim_unit
    gen_net = ImageGenNet(dim_inputs).to(device)

    data_loader = prepare_data_loader(opt)

    optim = torch.optim.Adam(gen_net.parameters(), lr=1e-4)

    train(gen_net, data_loader, optim, opt.epochs, opt.mode)

    with torch.no_grad():
        data_loader = prepare_data_loader(opt, batch_size=32, shuffle=False)

        visualize(gen_net, data_loader, mode=opt.mode,
                  file='{}/{}/image_gen{}.png'.format(MODEL_DOMAIN_DIR, opt.model_repr, opt.mode))


if __name__ == '__main__':
    main()
