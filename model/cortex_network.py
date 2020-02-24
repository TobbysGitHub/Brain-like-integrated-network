import torch
import torch.nn as nn

from model import opt_parser
from model.sub_layers.cortex_layer import CortexLayer
from model.sub_layers.modules.outputs import CortexLayerOutputs, CortexNetworkOutputs


# num_units = [32, 32, 32]
class CortexNetwork(nn.Module):
    def __init__(self, num_units_layer, opt):
        super().__init__()
        self.num_units = sum(num_units_layer)
        self.num_units_layer = num_units_layer
        self.dim_unit = opt.dim_unit

        self.linear = nn.Linear(4 * 96 * 96, 256)
        self.layers = nn.ModuleList()

        def cal_dims(n_units_l, dim_unit, dim_inputs_0):
            dim_forward = [dim_inputs_0] + [dim_unit * n_units_l[:-1]]
            dim_backward = [dim_unit * n_units_l[1:]] + [0]
            return n_units_l, dim_forward, dim_backward

        for n, d_f, d_b in zip(cal_dims(num_units_layer, self.dim_unit, 256)):
            self.layers.append(CortexLayer(num_units=n,
                                           dim_inputs_forward=d_f,
                                           dim_inputs_backward=d_b,
                                           dim_unit=self.dim_unit,
                                           dim_hid_enc_unit=opt.dim_hid_enc_unit,
                                           dim_hid_agg_unit=opt.dim_hid_agg_unit,
                                           max_bptt=opt.max_bptt,
                                           t_interlayer=opt.t_interlayer,
                                           t_introlayer=opt.t_introlayer))

        self.dim_outputs = sum([l.dim_outputs for l in self.layers])

        self.opt = opt
        self.mix_fn = lambda x1, x2: self.opt.mix_outputs_ratio * x1 + (1 - self.opt.enc_att_ratio) * x2

    def forward(self, x, att_outputs=None):
        if att_outputs is None:
            att_outputs = [torch.zeros(1) for _ in range(len(self.num_units_layer))]

        enc_outputs = []
        agg_outputs = []
        agg_outputs_preview = []
        outputs = []

        output = self.linear(x)
        for i, layer in enumerate(self.layers):
            enc_output, agg_output, agg_output_preview = layer(output)
            enc_outputs.append(enc_output)
            agg_outputs.append(agg_output)
            agg_outputs_preview.append(agg_output_preview)
            with torch.no_grad():
                output = self.mix_fn(enc_output, att_outputs[i])
                outputs.append(output)

        for i, layer in enumerate(self.layers[:-1]):
            layer.backward(outputs[i + 1])

        def cat(tensors):
            for tensor in tensors:
                if tensor is None:
                    return None
            return torch.cat(tensors, dim=1)

        return tuple((cat(x) for x in (enc_outputs, agg_outputs, agg_outputs_preview)))
