import os

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from model import opt_parser
from model.cortex_network import CortexNetwork
from model.hippoCampus_network import HippoCampusNetwork

def load(dir, file_name):
    return torch.load(os.path.join(dir, file_name))

class Frames4DataSet(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(data) - 3

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.data[item: item + 4]



def prepare_data_loader():
    data = load('env/data', 'car-racing.64')
    data_set =

    pass


def main():
    opt = opt_parser.parse_opt()

    num_units_layer = [32, 16, 8]

    cortex = CortexNetwork(num_units_layer=num_units_layer, opt=opt)

    hippo_campus = HippoCampusNetwork(num_units_layer=num_units_layer,
                                      dim_inputs=cortex.dim_outputs,
                                      dim_attention_global=opt.dim_attention_global,
                                      dim_attention_unit=opt.dim_attention_unit)

    data_loader = prepare_data_loader()
