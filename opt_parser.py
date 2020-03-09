import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--state_dict',
                        help='model state_dict')

    parser.add_argument('--epochs', default=4,
                        help='epochs')

    parser.add_argument('--batch_size', default=16,
                        help='size of batch')

    parser.add_argument('--dim_unit', default=8,
                        help='dim of the output of each unit')

    parser.add_argument('--dim_hid_enc_unit', default=32,
                        help='number of hidden nodes inside each unit`s encode network')

    parser.add_argument('--dim_hid_agg_unit', default=32,
                        help='number of hidden nodes inside each unit`s aggregate network')

    parser.add_argument('--max_bptt', default=8,
                        help='the max steps of back-propagation-through-time in GRU of the aggregate network')

    parser.add_argument('--t_intro_region', default=4,
                        help='the delay of aggregation output')

    parser.add_argument('--t_inter_region', default=4,
                        help='the delay of backward inputs')
    # 32
    parser.add_argument('--dim_attention_global', default=16,
                        help='the dim of attention of query and key in each unit')
    # 8
    parser.add_argument('--dim_attention_unit', default=4,
                        help='the dim of attention of query and key in each unit')

    parser.add_argument('--memory_size', default=256,
                        help='the negative samples in each unit')

    parser.add_argument('--memory_delay', default=10,
                        help='the delay to be sent to memory')

    parser.add_argument('--attention_mask_p', default=0.,
                        help='random mask some memory to avoid over-fitting')

    parser.add_argument('--outputs_mix', default=0.5,
                        help='the mix ratio between att_outputs and enc_outputs')

    parser.add_argument('--mem_interval', default=4,
                        help='the memory load interval')

    # parser.add_argument('--reward_gamma', default=0.88,
    #                     help='decay rate when the reward`s influence back propagate on memory through time')
    #
    # parser.add_argument('--reward_fresh_rate', default=0.01,
    #                     help='the max ratio of reward update if attention weight=1')

    opt = parser.parse_args()

    return opt
