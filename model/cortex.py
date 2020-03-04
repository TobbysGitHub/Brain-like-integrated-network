import torch
import torch.nn as nn

from model.modules.cortex_region import CortexRegion


class Cortex(nn.Module):
    def __init__(self, num_units_regions,
                 dim_unit,
                 dim_hid_enc_unit,
                 dim_hid_agg_unit,
                 max_bptt,
                 t_inter_region,
                 t_intro_region,
                 opt):
        super().__init__()
        self.num_units = sum(num_units_regions)
        self.num_units_regions = num_units_regions
        self.dim_inputs = 4 * (96 * 96 + 3)
        self.dim_outputs = self.num_units * dim_unit

        self.linear = nn.Linear(self.dim_inputs, 256)

        def cal_dims_regions(n_u_regions, dim_unit, dim_inputs):
            dim_forward = [dim_inputs] + [dim_unit * x for x in n_u_regions[:-1]]
            dim_backward = [dim_unit * x for x in n_u_regions[1:]] + [0]
            return n_u_regions, dim_forward, dim_backward

        self.cortex_regions = nn.ModuleList()
        for n, d_f, d_b in zip(*cal_dims_regions(num_units_regions, dim_unit, 256)):
            self.cortex_regions.append(CortexRegion(num_units=n,
                                                    dim_inputs_forward=d_f,
                                                    dim_inputs_backward=d_b,
                                                    dim_unit=dim_unit,
                                                    dim_hid_enc_unit=dim_hid_enc_unit,
                                                    dim_hid_agg_unit=dim_hid_agg_unit,
                                                    max_bptt=max_bptt,
                                                    t_inter_region=t_inter_region,
                                                    t_intro_region=t_intro_region))

        self.mix = lambda x1, x2: opt.outputs_mix * x1 + (1 - opt.outputs_mix) * x2

    def forward(self, x, att_outputs=None):
        """
        :param x: s_b * d_inputs
        :param att_outputs: s_b * n_units * d_u
        :return:
        """
        if att_outputs is not None:
            with torch.no_grad():
                att_outputs = torch.split(att_outputs, self.num_units_regions, dim=1)

        enc_outputs = []
        agg_outputs = []
        agg_outputs_preview = []
        outputs = []

        outputs_region = self.linear(x)
        for i, cortex_region in enumerate(self.cortex_regions):
            enc_outputs_region, agg_outputs_region, agg_outputs_region_preview = cortex_region(outputs_region)
            enc_outputs.append(enc_outputs_region)
            agg_outputs.append(agg_outputs_region)
            agg_outputs_preview.append(agg_outputs_region_preview)

            if att_outputs is None:
                outputs_region = enc_outputs_region
            else:
                outputs_region = self.mix(enc_outputs_region, att_outputs[i])
            # 梯度隔绝
            # gradient isolation
            outputs_region = outputs_region.detach()

            outputs.append(outputs_region)

        for i, cortex_region in enumerate(self.cortex_regions[:-1]):
            cortex_region.backward(outputs[i + 1])

        def cat(tensors):
            for tensor in tensors:
                if tensor is None:
                    return None
            return torch.cat(tensors, dim=1)

        return tuple((cat(x) for x in (enc_outputs, agg_outputs, agg_outputs_preview)))

    def reset(self):
        for region in self.cortex_regions:
            region.aggregator.reset()

    def extra_repr(self) -> str:
        return 'num_units_regions:{num_units_regions}, ' \
               'dim_inputs:{dim_inputs}, ' \
               'dim_outputs:{dim_outputs}'.format(**self.__dict__)

    def encoder_parameters(self):
        yield from self.linear.parameters()
        for region in self.cortex_regions:
            yield from region.encoder.parameters()

    def aggregator_parameters(self):
        for region in self.cortex_regions:
            yield from region.aggregator.parameters()
