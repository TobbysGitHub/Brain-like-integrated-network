import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=32,
                        help='size of batch')

    parser.add_argument('--dim_outputs_unit', default=8,
                        help='dim of the output of each unit')

    parser.add_argument('--dim_hid_enc_unit', default=32,
                        help='number of hidden nodes inside each unit`s encode network')

    parser.add_argument('--dim_hid_agg_unit', default=21,
                        help='number of hidden nodes inside each unit`s aggregate network')

    parser.add_argument('--max_bptt_agg', default=20,
                        help='the max steps of back-propagation-through-time in GRU of the aggregate network')

    parser.add_argument('--t_introlayer', default=4,
                        help='the delay of aggregation output')

    parser.add_argument('--t_interlayer', default=4,
                        help='the delay of backward inputs')

    parser.add_argument('--dim_attention', default=32,
                        help='the dim of attention of query and key in each unit')

    parser.add_argument('--dim_attention_unit', default=8,
                        help='the dim of attention of query and key in each unit')

    parser.add_argument('--memory_capacity', default=64,
                        help='the negative samples in each unit')

    parser.add_argument('--attention_mask_p', default=0.382,
                        help='random mask some memory to avoid over-fitting')

    parser.add_argument('--max_bptt_mem', default=200,
                        help='the max steps of back-propagation-through-time through memory')

    parser.add_argument('--memory_fresh_rate', default=0.1,
                        help='the max ratio of memory update if attention weight=1')

    parser.add_argument('--reward_gamma', default=0.88,
                        help='decay rate when the reward`s influence back propagate on memory through time')

    parser.add_argument('--reward_fresh_rate', default=0.01,
                        help='the max ratio of reward update if attention weight=1')

    opt = parser.parse_args()

    return opt
