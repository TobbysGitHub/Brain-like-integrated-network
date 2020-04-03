from argparse import ArgumentParser


def parse_opt():
    parser = ArgumentParser()

    parser.add_argument('--model_repr',
                        default='model__unit_n8_d8_@d16_@unit_d4_@mem256_@groups8_Mar22_08-26-04',
                        help='The dir where brain-like-network stored')
    parser.add_argument('--mode', type=int, default=2,
                        help='O to train on attention, 1 to train on outputs of units')

    return parser.parse_args()
