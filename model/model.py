from torch import nn as nn

from model.cortex import Cortex
from model.hippocampus import Hippocampus
from model.memory import Memory


class Model(nn.Module):
    def __init__(self, num_units_regions, opt):
        super().__init__()
        self.cortex = Cortex(num_units_regions=num_units_regions,
                             dim_unit=opt.dim_unit,
                             dim_hid_enc_unit=opt.dim_hid_enc_unit,
                             dim_hid_agg_unit=opt.dim_hid_agg_unit,
                             max_bptt=opt.max_bptt,
                             t_inter_region=opt.t_inter_region,
                             t_intro_region=opt.t_intro_region,
                             opt=opt)

        self.hippocampus = Hippocampus(num_units_regions=num_units_regions,
                                       dim_inputs=self.cortex.dim_outputs,
                                       dim_attention_global=opt.dim_attention_global,
                                       dim_attention_unit=opt.dim_attention_unit)

        self.memory = Memory(num_units_regions=num_units_regions,
                             mask_p=opt.attention_mask_p,
                             dim_unit=opt.dim_unit,
                             memory_size=opt.memory_size,
                             delay=opt.memory_delay)
