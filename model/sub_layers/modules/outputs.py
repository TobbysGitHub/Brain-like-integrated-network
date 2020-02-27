from functools import reduce

import torch


class CortexLayerOutputs:
    def __init__(self, enc_outputs, agg_outputs, agg_outputs_preview):
        self.enc_outputs = enc_outputs
        self.agg_outputs = agg_outputs
        self.agg_outputs_preview = agg_outputs_preview


class CortexNetworkOutputs:
    def __init__(self, *layer_outputs_seq):
        layer_outputs: CortexLayerOutputs

        self.enc_outputs = torch.cat(
            [layer_outputs.enc_outputs for layer_outputs in layer_outputs_seq],
            dim=1)
        self.agg_outputs = torch.cat(
            [layer_outputs.agg_outputs for layer_outputs in layer_outputs_seq],
            dim=1)
        self.agg_outputs_preview = torch.cat(
            [layer_outputs.agg_outputs_preview for layer_outputs in layer_outputs_seq],
            dim=1)


class MemoryOutputs:
    def __init__(self, data, num_units_layers):
        self.num_units_layers = num_units_layers
        self.item_begin = [sum(num_units_layers[0:x]) for x in range(len(num_units_layers))]
        self.data = data

    def __getitem__(self, item):
        begin = self.item_begin[item]
        end = begin + self.num_units_layers[item]
        return self.data[:, begin:end]
