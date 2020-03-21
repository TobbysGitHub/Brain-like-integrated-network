from argparse import ArgumentParser


def parse_opt():
    parser = ArgumentParser()

    parser.add_argument('--model_repr',
                        default='model__unit_n8_d8_@d16_@unit_d4_@mem256_@groups8_Mar19_19-49-20',
                        help='The dir where brain-like-network stored')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')

    parser.add_argument('--mode', type=int, default=0,
                        help='O to train on attention, 1 to train on outputs of units')

    parser.add_argument('--epochs', type=int, default=8,
                        help='epochs')

    return parser.parse_args()
