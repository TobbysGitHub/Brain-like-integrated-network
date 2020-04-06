from argparse import ArgumentParser


def parse_opt():
    parser = ArgumentParser()

    parser.add_argument('--model_domain',
                        default='model__unit_n8_d8_@d16_@unit_d4_@mem256_@groups8_Mar22_08-26-04',
                        help='The dir where brain-like-network stored')
    parser.add_argument('--mode', type=int, default=2,
                        help='0 and 1 is invalid'
                             '2 to train on agg_outputs, 3 to train on att_outputs')

    return parser.parse_args()
