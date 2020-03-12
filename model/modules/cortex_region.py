import torch
from torch import nn
from model.modules import *
from model.modules.components.fixed_size_queue import FixedSizeQueue


class CortexRegion(nn.Module):
    def __init__(self,
                 num_units,  # 本层单元数量
                 dim_inputs_forward,  # 前馈输入维度
                 dim_inputs_backward,  # 反馈输入维度
                 dim_unit,
                 dim_hid_enc_unit,
                 dim_hid_agg_unit,
                 max_bptt,
                 t_inter_region,
                 t_intro_region):
        super().__init__()

        self.num_units = num_units
        self.dim_inputs_forward = dim_inputs_forward
        self.dim_inputs_backward = dim_inputs_backward
        self.dim_unit = dim_unit
        self.dim_outputs = num_units * dim_unit

        self.t_inter_region = t_inter_region
        self.t_intro_region = t_intro_region

        self.encoder = Encoder(num_units=num_units,
                               dim_inputs=dim_inputs_forward,
                               dim_hidden=dim_hid_enc_unit,
                               dim_unit=dim_unit)
        self.aggregator = Aggregator(num_units=num_units,
                                     dim_inputs1=self.dim_outputs,
                                     dim_inputs2=dim_inputs_backward,
                                     dim_hidden=dim_hid_agg_unit,
                                     dim_unit=dim_unit,
                                     max_bptt=max_bptt)

        self.intro_region_queue = FixedSizeQueue(t_intro_region)
        self.inter_region_queue = FixedSizeQueue(t_inter_region)

    def forward(self, inputs):
        """

        :param inputs:  # s_b * d_input_forward
        :return:
        """
        enc_outputs = self.encoder(inputs)  # s_b * n_u * d_u
        agg_outputs = self.aggregator(enc_outputs.detach(), self.inter_region_queue.peek())  # s_b * n_u * d_u
        agg_outputs = self.intro_region_queue.offer(agg_outputs)
        agg_outputs_preview = self.intro_region_queue.peek()
        return enc_outputs, agg_outputs, agg_outputs_preview

    def backward(self, inputs_backward):
        self.inter_region_queue.offer(inputs_backward)

    def extra_repr(self) -> str:
        return 'num_units:{num_units}, ' \
               'dim_unit:{dim_unit}, ' \
               'dim_inputs_forward:{dim_inputs_forward}, ' \
               'dim_inputs_backward:{dim_inputs_backward}, ' \
               'dim_outputs:{dim_outputs}, ' \
               't_intro_region:{t_intro_region}, ' \
               't_inter_region:{t_inter_region}'.format(**self.__dict__)


def main():
    pass


if __name__ == '__main__':
    main()
