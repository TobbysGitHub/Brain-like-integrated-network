import torch
from torch import nn
from model import opt_parser
from model.sub_layers.loss_layer import ContrastiveLossLayer
from model.sub_layers import *
from model.utils.loop_queue import LoopQueue


class CortexLayer(nn.Module):
    def __init__(self,
                 num_units,  # 本层单元数量
                 dim_inputs_forward,  # 前馈输入维度
                 dim_inputs_backward,  # 反馈输入维度
                 opt):
        super().__init__()

        self.num_units = num_units
        self.dim_inputs_forward = dim_inputs_forward
        self.dim_inputs_backward = dim_inputs_backward
        self.dim_outputs_unit = opt.dim_outputs_unit
        self.dim_outputs = num_units * opt.dim_outputs_unit

        self.t_interlayer = opt.t_interlayer
        self.t_introlayer = opt.t_introlayer

        self.enc_layer = EncodeLayer(num_units, dim_inputs_forward, opt)
        self.agg_layer = AggregateLayer(num_units, self.dim_outputs, dim_inputs_backward, opt)
        self.att_layer = AttentionLayer(num_units, opt)
        self.loss_layer = ContrastiveLossLayer(num_units)

        self.a_q = LoopQueue(opt.t_introlayer)
        self.i_q = LoopQueue(opt.t_interlayer)

    def forward(self, inputs, attention, cal_loss=True):
        """

        :param inputs:  # s_b * d_input
        :param attention:       # s_b * d_att
        :param cal_loss:
        :return:
        """
        enc_outputs = self.enc_layer(inputs)
        agg_outputs = self.agg_layer(enc_outputs, self.i_q.peek())
        agg_outputs = self.a_q.offer(agg_outputs)

        att_outputs, negative_samples, weights, rewards, refresh_handler = \
            self.att_layer(attention, enc_outputs)

        if cal_loss:
            loss = self.loss_layer(enc_outputs, agg_outputs, negative_samples, weights)
        else:
            loss = None

        agg_outputs_preview = self.a_q.peek()
        return enc_outputs, att_outputs, rewards, agg_outputs_preview, refresh_handler, loss

    def backward(self, inputs_backward):
        self.i_q.offer(inputs_backward)

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
