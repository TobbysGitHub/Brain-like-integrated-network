import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--state_dict',
                        help='model state_dict')

    parser.add_argument('--early_stop', type=int, default=1e8)

    parser.add_argument('--data_file', type=str, default='car-racing.64',
                        help='which data to train on')

    parser.add_argument('--early_cuda', type=int, default=0,
                        help='0, 1, 2,to prevent out of cuda mem')

    parser.add_argument('--epochs', default=20, type=int,
                        help='epochs')

    parser.add_argument('--batch_size', default=256, type=int,
                        help='size of batch')

    parser.add_argument('--num_units_regions', nargs='+', type=int, default=[8],
                        help='list of num of units in diff regions')

    parser.add_argument('--dim_unit', default=2, type=int,
                        help='dim of the output of each unit')

    parser.add_argument('--dim_hid_enc_unit', default=32, type=int,
                        help='number of hidden nodes inside each unit`s encode network')

    parser.add_argument('--dim_hid_agg_unit', default=32, type=int,
                        help='number of hidden nodes inside each unit`s aggregate network')

    parser.add_argument('--max_bptt', default=8, type=int,
                        help='the max steps of back-propagation-through-time in GRU of the aggregate network')

    parser.add_argument('--t_intro_region', default=4, type=int,
                        help='the delay of aggregation output')

    parser.add_argument('--t_inter_region', default=4, type=int,
                        help='the delay of backward inputs')

    parser.add_argument('--dim_attention_global', default=16, type=int,
                        help='the dim of attention of query and key in each unit')

    parser.add_argument('--dim_attention_unit', default=2, type=int,
                        help='the dim of attention of query and key in each unit')

    parser.add_argument('--num_attention_groups', default=8, type=int,
                        help='the num of groups when apply soft_max attention')

    parser.add_argument('--attention_mask_p', default=0., type=float,
                        help='random mask some memory to avoid over-fitting')

    parser.add_argument('--outputs_mix', default=0.0, type=float,
                        help='the mix ratio between att_outputs and enc_outputs')

    # parser.add_argument('--reward_gamma', default=0.88,
    #                     help='decay rate when the reward`s influence back propagate on memory through time')
    #
    # parser.add_argument('--reward_fresh_rate', default=0.01,
    #                     help='the max ratio of reward update if attention weight=1')

    opt = parser.parse_args()

    return opt
