from argparse import ArgumentParser


def parse_opt():
    parser = ArgumentParser()

    parser.add_argument('--model_domain',
                        default='model__unit_n8_d8_@d16_@unit_d4_@mem256_@groups8_Mar22_08-26-04',
                        help='The dir where brain-like-network stored')

    return parser.parse_args()
