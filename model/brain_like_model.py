from datetime import datetime

import torch
from torch import nn as nn

from model.hippocampus import Hippocampus
from model.modules.aggregator import Aggregator
from model.modules.encoder import Encoder


class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_units_regions = opt.num_units_regions
        self.dim_unit = opt.dim_unit
        self.dim_attention_global = opt.dim_attention_global
        self.dim_attention_unit = opt.dim_attention_unit
        self.len_attention_memory = opt.batch_size
        self.num_attention_groups = opt.num_attention_groups

        self.dim_inputs = 4 * (96 * 96 + 4)

        self.t_intro_region = opt.t_intro_region
        self.t_inter_region = opt.t_inter_region
        self.memory = []
        self.mem_attention = None

        self.encoder = Encoder(num_units_regions=opt.num_units_regions,
                               dim_unit=opt.dim_unit,
                               dim_hidden=opt.dim_hid_enc_unit)

        self.aggregator = Aggregator(num_units_regions=opt.num_units_regions,
                                     dim_unit=opt.dim_unit,
                                     dim_hidden=opt.dim_hid_agg_unit,
                                     dim_outputs_regions=self.encoder.dim_outputs_regions,
                                     max_bptt=opt.max_bptt)

        self.hippocampus = Hippocampus(num_units_regions=opt.num_units_regions,
                                       dim_inputs=self.encoder.dim_outputs,
                                       dim_attention_global=opt.dim_attention_global,
                                       dim_attention_unit=opt.dim_attention_unit,
                                       num_attention_groups=opt.num_attention_groups,
                                       mask_p=opt.attention_mask_p)

        self.timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
        self.num_units = '_'.join([str(n) for n in self.num_units_regions])
        self.data_file = opt.data_file
        self.rotations = '_'.join([str(r) for r in opt.rotations])

    def forward(self, inputs):
        """
        :param inputs: s_b * (4*96*96+4)
        :param memory:
        :return:
        """
        warm_up = len(self.memory) < max(self.t_intro_region, self.t_inter_region)
        if warm_up:
            enc_output_list = self.encoder(inputs)  # [[s_b, n_u, d_u],]
            self.memory.append(enc_output_list)
            return None
        else:
            agg_output_list = self.aggregator(self.memory[-self.t_intro_region],
                                              self.memory[-self.t_inter_region])  # [[s_b, n_u, d_u],]
            agg_outputs = torch.cat(agg_output_list, dim=1)
            mem_outputs = torch.cat(self.memory[-1], dim=1)

            if self.mem_attention is None:
                self.mem_attention = self.hippocampus(agg_outputs)
                return None

            attention, weights, att_outputs = self.hippocampus(agg_outputs, (self.mem_attention, mem_outputs))
            self.mem_attention = attention
            enc_output_list = self.encoder(inputs, att_outputs)
            enc_outputs = torch.cat(enc_output_list, dim=1)

            self.memory.append(enc_output_list)
            self.memory.pop(0)
            return (enc_outputs, agg_outputs, att_outputs, mem_outputs), attention, weights

    def reset(self):
        self.memory.clear()
        self.mem_attention = None
        self.aggregator.reset()

    def extra_repr(self) -> str:
        return '{timestamp}_model__unit_n{num_units}_d{dim_unit}_' \
               '@d{dim_attention_global}_@unit_d{dim_attention_unit}_' \
               '@mem{len_attention_memory}_@groups{num_attention_groups}_' \
               'r{rotations}' \
            .format(**self.__dict__)
