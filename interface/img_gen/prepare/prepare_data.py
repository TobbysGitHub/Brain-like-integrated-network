import torch
from tqdm import tqdm

from dirs import MODEL_DOMAIN_DIR
from interface.img_gen import opt_parser
from model.brain_like_model import Model
from model.train import prepare_data_loader, iter_frame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_data(model, data_loader, samples):
    data_img, data_outputs, data_attention = [], [], []
    for batch in data_loader:
        batch = batch.to(device)
        agg_outputs_preview, memories = None, None
        for index, inputs in tqdm(enumerate(iter_frame(batch)), mininterval=2, leave=True):
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
            if index < 60 or not index % 10 == 0 or attention is None:
                continue

            samples -= len(attention)
            data_img.append(inputs[:, -96 * 96 - 4:-4])
            data_outputs.append(enc_outputs)
            data_attention.append(attention_global)
            if samples <= 0:
                return tuple(torch.cat(x, dim=0) for x in (data_img, data_outputs, data_attention))


def main():
    opt = opt_parser.parse_opt()

    model_opt = torch.load('{}/{}/opt'.format(MODEL_DOMAIN_DIR, opt.model_repr))
    samples = int(opt.samples)
    # samples = 16

    model = Model(model_opt).to(device)
    model.load_state_dict(torch.load(model_opt.state_dict, map_location=device))

    data_loader = prepare_data_loader(model_opt.batch_size, shuffle=False)

    with torch.no_grad():
        data = gen_data(model, data_loader, samples)

    torch.save(data[0], f='{}/{}/car-racing-img.{}'.format(MODEL_DOMAIN_DIR, opt.model_repr, samples))
    torch.save(data[1], f='{}/{}/car-racing-outputs.{}'.format(MODEL_DOMAIN_DIR, opt.model_repr, samples))
    torch.save(data[2], f='{}/{}/car-racing-attention.{}'.format(MODEL_DOMAIN_DIR, opt.model_repr, samples))


if __name__ == '__main__':
    main()
