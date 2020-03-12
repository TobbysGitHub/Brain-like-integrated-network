from datetime import datetime

from torch import nn as nn

from model.cortex import Cortex
from model.hippocampus import Hippocampus


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
        self.cortex = Cortex(num_units_regions=opt.num_units_regions,
                             dim_unit=opt.dim_unit,
                             dim_hid_enc_unit=opt.dim_hid_enc_unit,
                             dim_hid_agg_unit=opt.dim_hid_agg_unit,
                             max_bptt=opt.max_bptt,
                             t_inter_region=opt.t_inter_region,
                             t_intro_region=opt.t_intro_region,
                             opt=opt)

        self.hippocampus = Hippocampus(num_units_regions=opt.num_units_regions,
                                       dim_inputs=self.cortex.dim_outputs,
                                       dim_attention_global=opt.dim_attention_global,
                                       dim_attention_unit=opt.dim_attention_unit,
                                       num_attention_groups=opt.num_attention_groups,
                                       mask_p=opt.attention_mask_p)

        self.timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
        self.num_units = '_'.join([str(n) for n in self.num_units_regions])

    def extra_repr(self) -> str:
        return 'model__unit_n{num_units}_d{dim_unit}_' \
               '@d{dim_attention_global}_@unit_d{dim_attention_unit}_' \
               '@mem{len_attention_memory}_@groups{num_attention_groups}_{timestamp}' \
            .format(**self.__dict__)
