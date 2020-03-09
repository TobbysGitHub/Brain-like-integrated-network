import torch
from tqdm import tqdm

import opt_parser
from model.Model import Model
from model.train import prepare_data_loader

samples = 0


def gen_data(model, data_loader):
    global samples
    data_img, data_outputs, data_attention = [], [], []
    for batch in data_loader:
        agg_outputs_preview, memories = None, None
        for index, inputs in tqdm(enumerate(batch), mininterval=2, leave=True):
            if agg_outputs_preview is not None:
                attention_global = model.hippocampus.model[0](
                    agg_outputs_preview.view(-1, model.hippocampus.dim_inputs))
                if memories is not None:
                    attention, weights, att_outputs = model.hippocampus(agg_outputs_preview, memories)
                else:
                    attention = model.hippocampus(agg_outputs_preview)
                    weights, att_outputs = None, None
            else:
                attention, weights, att_outputs = (None,) * 3

            enc_outputs, agg_outputs, agg_outputs_preview = model.cortex(inputs, att_outputs)
            if not index % 10 == 0 or attention is None:
                continue

            samples += len(attention)
            data_img.append(inputs[:, -96 * 96 - 3:-3])
            data_outputs.append(enc_outputs)
            data_attention.append(attention_global)
            # if samples >= 256:
            if samples >= 1024 * 8:
                return tuple(torch.cat(x, dim=0) for x in (data_img, data_outputs, data_attention))


def main():
    global samples
    opt = opt_parser.parse_opt()
    num_units_regions = [8]

    # model = Model(num_units_regions, opt)
    # model.load_state_dict(torch.load('model_state/steps_2650'))
    #
    # data_loader = prepare_data_loader()
    #
    # with torch.no_grad():
    #     data = gen_data(model, data_loader)
    #
    # torch.save(data[0], f='data/car-racing-img.{}'.format(samples))
    # torch.save(data[1], f='data/car-racing-outputs.{}'.format(samples))
    # torch.save(data[2], f='data/car-racing-attention.{}'.format(samples))

    model_untrained = Model(num_units_regions, opt)

    data_loader = prepare_data_loader()

    with torch.no_grad():
        data_untrained = gen_data(model_untrained, data_loader)

    torch.save(data_untrained[0], f='data/car-racing-img-untrained.{}'.format(samples))
    torch.save(data_untrained[1], f='data/car-racing-outputs-untrained.{}'.format(samples))
    torch.save(data_untrained[2], f='data/car-racing-attention-untrained.{}'.format(samples))


if __name__ == '__main__':
    main()
