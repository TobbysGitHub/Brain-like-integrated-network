import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_batch(model, data_loader, batch_size):
    for batch_frames in data_loader:
        model.reset()

        for index, inputs in enumerate(tqdm(batch_frames, total=256, mininterval=2, leave=False, desc='  - Frame')):
            assert len(inputs) >= batch_size

            with torch.no_grad():
                results = model(inputs)
                if results is None:
                    continue
                if index < 24 or not index % 31 == 0:
                    continue
                (enc_outputs, agg_outputs, att_outputs, _), _, _, global_attention = results

            imgs, global_attentions, enc_outputs, agg_outputs, att_outputs = (torch.split(x, batch_size) for x in
                                                                              (inputs[:, -96 * 96:], global_attention,
                                                                               enc_outputs,
                                                                               agg_outputs, att_outputs))

            for img, attention, enc_output, agg_output, att_output in \
                    zip(imgs, global_attentions, enc_outputs, agg_outputs, att_outputs):
                if len(img) == batch_size:
                    yield img, attention, enc_output, agg_output, att_output
