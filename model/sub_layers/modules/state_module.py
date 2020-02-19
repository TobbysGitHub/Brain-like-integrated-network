import os
from collections import OrderedDict

import torch
import torch.nn as nn


class StatefulModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_ordered_dict = OrderedDict()

    def register_state(self, name, param):
        self.state_ordered_dict[name] = param
        return param

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        destination = super().state_dict(destination, prefix, keep_vars)
        for name, param in self.state_ordered_dict.items():
            destination[prefix + name] = param
        return destination

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                      missing_keys, unexpected_keys, error_msgs)
        for name, param in self.state_ordered_dict.items():
            key = prefix + name
            param_input = state_dict[key].data
            param.copy_(param_input)

    def _apply(self, fn):
        super()._apply(fn)

        for name, param in self.state_ordered_dict.items():
            self.__setattr__(name, fn(param))

        return self


def main():
    sm = StatefulModule()
    sm.tensor = torch.ones(2, 2)
    sm.register_state('tensor', sm.tensor)

    state = sm.state_dict()

    torch.save(state, f='./data')
    state = torch.load(f='./data')
    state['tensor'] += 1
    sm.load_state_dict(state)
    print(sm.tensor)
    os.remove('./data')
    pass


if __name__ == '__main__':
    main()
