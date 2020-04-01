import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_batch(model, data_loader, batch_size, epoch=None):
    for batch_frames in data_loader:
        model.reset()
        if epoch is not None:
            desc = 'epoch:{}'.format(epoch)
        else:
            desc = ''
        for index, inputs in enumerate(tqdm(batch_frames, total=256, mininterval=2, leave=False, desc=desc)):
            assert len(inputs) >= batch_size

            with torch.no_grad():
                results = model(inputs)
                if results is None:
                    continue
                if index < 24 or not index % 10 == 0:
                    continue
                inputs_size = inputs.shape[0]
                (enc_outputs, agg_outputs, *_), *_ = results

                attention = model.hippocampus.model[0](agg_outputs.view(inputs_size, -1))

            imgs, enc_outputs, attentions = (torch.split(x, batch_size) for x in
                                         (inputs[:, -96 * 96:], enc_outputs, attention))

            for img, attention, enc_output in zip(imgs, attentions, enc_outputs):
                if len(img) == batch_size:
                    yield img, attention, enc_output