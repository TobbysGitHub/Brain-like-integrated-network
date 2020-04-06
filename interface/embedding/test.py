from dataset.dataset import prepare_data_loader
from dirs.dirs import *
from interface.casual_test import opt_parser
from interface.img_gen.image_gen_net import ImageGenNet
from model.brain_like_model import *
from tensor_board import tb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def creat_model(model_opt):
    model = Model(model_opt).to(device)
    model.load_state_dict(torch.load(model_opt.state_dict, map_location=device))
    return model


def creat_net(model, model_domain, mode):
    dim_inputs = model.encoder.dim_outputs
    net = ImageGenNet(dim_inputs).to(device)
    state_dict_dir = '{}/{}/mode{}/gen_net_state_dict'.format(MODEL_DOMAIN_DIR, model_domain, mode)
    net.load_state_dict(torch.load(state_dict_dir, map_location=device))
    return net


def prepare_tensorboard(opt):
    tb.creat_writer(steps_fn=None,
                    log_dir='{}/{}'.format(EMBEDDING_RUNS_DIR, opt.model_domain))


def predict(model, model_opt):
    file = model_opt.data_file + '.eval'
    data_loader = prepare_data_loader(batch_size=model_opt.batch_size, file=file,
                                      early_cuda=model_opt.early_cuda,
                                      shuffle=False)
    file = model_opt.data_file + '.eval.direction'
    label_loader = prepare_data_loader(batch_size=model_opt.batch_size, file=file,
                                       early_cuda=0,
                                       shuffle=False)

    count = 2048
    enc_outputs_list = []
    attention_list = []
    labels_list = []
    for batch in zip(data_loader, label_loader):
        model.reset()
        for n, (inputs, labels) in enumerate(zip(*batch)):
            result = model(inputs)
            if n > 20 and n % 31 == 0 :
                labels = labels.view(model_opt.batch_size, 4)[:, -1]
                (enc_outputs, *_), attention, weights = result
                enc_outputs_list.append(enc_outputs.view(model_opt.batch_size, -1))
                attention_list.append(attention.view(model_opt.batch_size, -1))
                labels_list.append(labels)
                count -= model_opt.batch_size
            if count <= 0:
                break
        if count <= 0:
            break

    labels = torch.cat(labels_list, dim=0)
    enc_outputs = torch.cat(enc_outputs_list, dim=0)
    attention = torch.cat(attention_list, dim=0)
    tb.writer.add_embedding(mat=enc_outputs, metadata=labels, tag='enc_outputs')
    tb.writer.add_embedding(mat=attention, metadata=labels, tag='attention')


def main():
    opt = opt_parser.parse_opt()

    model_opt = torch.load('{}/{}/opt'.format(MODEL_DOMAIN_DIR, opt.model_domain))
    model = creat_model(model_opt)

    prepare_tensorboard(opt)

    with torch.no_grad():
        predict(model, model_opt=model_opt)


if __name__ == '__main__':
    import sys

    sys.argv = ['', '--model_domain', 'Apr06_16-01-59_model__unit_n8_d2_@d16_@unit_d2_@mem256_@groups8_mix2_cubic_96']
    main()
