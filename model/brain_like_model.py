from datetime import datetime

import torch
from torch import nn as nn

from model.hippocampus import Hippocampus
from model.modules.aggregator import Aggregator
from model.modules.encoder import Encoder

# cache mix mode
NONE = 1
WITH_AGGREGATOR = 2
WITH_ATTENTION = 3


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
        self.caches = []
        self.last_attention = None
        self.last_enc_outputs = None

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

        self.blinded = False
        self.counter = 0
        self.mix_mode = opt.mix_mode

    def forward(self, inputs):
        """
        :param inputs: s_b * (4*96*96+4)
        :return:
        """
        if self.warm_up(inputs):
            return None

        agg_output_list = self.aggregator(self.caches[-self.t_intro_region], self.caches[-self.t_inter_region])
        agg_outputs = torch.cat(agg_output_list, dim=1)

        mem_outputs, mem_attention = self.last_enc_outputs, self.last_attention
        attention, weights, att_outputs = self.hippocampus(agg_outputs, (mem_attention, mem_outputs))
        att_output_list = list(torch.split(att_outputs, self.num_units_regions, dim=1))

        if not self.blinded:
            enc_output_list = self.encoder(inputs)
            enc_outputs = torch.cat(enc_output_list, dim=1)

            self.last_attention = attention
            self.last_enc_outputs = enc_outputs
        elif self.opt.mix_mode == WITH_AGGREGATOR:
            enc_output_list = agg_output_list
            enc_outputs = agg_outputs
        elif self.opt.mix_mode == WITH_ATTENTION:
            enc_output_list = att_output_list
            enc_outputs = att_outputs
        else:
            raise ValueError('If is blinded, the model must has been trained on mix-mode 2 or 3')

        self.counter += 1
        if self.mix_mode == NONE or self.counter % 2 == 0:
            self.caches.append(enc_output_list)
        elif self.mix_mode == WITH_AGGREGATOR:
            self.caches.append(agg_output_list)
        elif self.mix_mode == WITH_ATTENTION:
            self.caches.append(att_output_list)
        else:
            raise ValueError('Invalid mix_mode!!')
        self.caches.pop(0)

        return (enc_outputs, agg_outputs, att_outputs, mem_outputs), attention, weights

    def warm_up(self, inputs):
        if len(self.caches) < max(self.t_intro_region, self.t_inter_region):
            enc_output_list = self.encoder(inputs)  # [[s_b, n_u, d_u],]
            self.caches.append(enc_output_list)
            return True
        elif self.last_attention is None:
            agg_output_list = self.aggregator(self.caches[-self.t_intro_region],
                                              self.caches[-self.t_inter_region])  # [[s_b, n_u, d_u],]
            agg_outputs = torch.cat(agg_output_list, dim=1)
            self.last_attention = self.hippocampus.attention(agg_outputs)
            enc_output_list = self.encoder(inputs)
            self.last_enc_outputs = torch.cat(enc_output_list, dim=1)
            self.caches.append(enc_output_list)
            self.caches.pop(0)
            return True
        return False

    def reset(self):
        self.caches.clear()
        self.last_attention = None
        self.aggregator.reset()

    def extra_repr(self) -> str:
        return '{timestamp}_model__unit_n{num_units}_d{dim_unit}_' \
               '@d{dim_attention_global}_@unit_d{dim_attention_unit}_' \
               '@mem{len_attention_memory}_@groups{num_attention_groups}_' \
               'mix{mix_mode}_{data_file}' \
            .format(**self.__dict__)
