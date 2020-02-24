import torch
from torch import nn
from model import opt_parser
from model.sub_layers import *
from model.utils.fixed_size_queue import FixedSizeQueue


class CortexLayer(nn.Module):
    def __init__(self,
                 num_units,  # 本层单元数量
                 dim_inputs_forward,  # 前馈输入维度
                 dim_inputs_backward,  # 反馈输入维度
                 dim_unit,
                 dim_hid_enc_unit,
                 dim_hid_agg_unit,
                 max_bptt,
                 t_interlayer,
                 t_introlayer):
        super().__init__()

        self.num_units = num_units
        self.dim_inputs_forward = dim_inputs_forward
        self.dim_inputs_backward = dim_inputs_backward
        self.dim_outputs_unit = dim_unit
        self.dim_outputs = num_units * dim_unit

        self.t_interlayer = t_interlayer
        self.t_introlayer = t_introlayer

        self.enc_layer = EncodeLayer(num_units=num_units,
                                     dim_inputs=dim_inputs_forward,
                                     dim_hidden=dim_hid_enc_unit,
                                     dim_unit=dim_unit)
        self.agg_layer = AggregateLayer(num_units=num_units,
                                        dim_input1=self.dim_outputs,
                                        dim_input2=dim_inputs_backward,
                                        dim_hidden=dim_hid_agg_unit,
                                        dim_unit=dim_unit,
                                        max_bptt=max_bptt)

        self.a_q = FixedSizeQueue(t_introlayer)
        self.b_q = FixedSizeQueue(t_interlayer)

    def forward(self, inputs):
        """

        :param inputs:  # s_b * d_input_forward
        :return:
        """
        enc_outputs = self.enc_layer(inputs)  # s_b * n_u * d_o_u
        agg_outputs = self.agg_layer(enc_outputs, self.b_q.peek())  # s_b * n_u * d_o_u
        agg_outputs = self.a_q.offer(agg_outputs)
        agg_outputs_preview = self.a_q.peek()
        return enc_outputs, agg_outputs, agg_outputs_preview

    def backward(self, inputs_backward):
        self.b_q.offer(inputs_backward)

    def store(self, attention_seq, outputs_seq):
        self.memory.store(attention_seq, outputs_seq)

    def extra_repr(self) -> str:
        return 'num_units:{num_units}, ' \
               'dim_inputs_forward:{dim_inputs_forward}, ' \
               'dim_inputs_backward:{dim_inputs_backward}, ' \
               'dim_outputs_unit:{dim_outputs_unit}, ' \
               'dim_outputs:{dim_outputs}, ' \
               't_introlayer:{t_introlayer}, ' \
               't_interlayer:{t_interlayer}'.format(**self.__dict__)


def main():
    num_units = 8
    d_f = 32
    d_b = 43
    opt = opt_parser.parse_opt()
    l = CortexLayer(num_units, d_f, d_b, opt)

    x = torch.rand(opt.batch_size, d_f)
    att = torch.rand(opt.batch_size, opt.dim_attention)
    x_b = torch.rand(opt.batch_size, d_b)

    for i in range(12):
        y = l(x, att)
        l.backward(x_b)
    reward = torch.rand(opt.batch_size)
    y[3](reward)

    pass


if __name__ == '__main__':
    main()
