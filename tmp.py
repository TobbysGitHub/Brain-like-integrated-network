import torch
import os
import matplotlib.pyplot as plt


def load(dir, file_name):
    return torch.load(os.path.join(dir, file_name))


data = load('data', 'car-racing.64')
pass
img = data[0][200][:96*96].view(96,96)
plt.imshow(img, cmap='gray')