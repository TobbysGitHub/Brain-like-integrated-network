from torch import nn


class HippoCampusNetwork(nn.Module):
    def __init__(self, num_units, dim_inputs, dim_attention_global, dim_attention_unit):
        super().__init__()
        self.num_units = num_units
        self.dim_inputs = dim_inputs
        self.dim_attention_global = dim_attention_global
        self.dim_attention_unit = dim_attention_unit

        self.model = nn.Sequential(
            nn.Linear(in_features=self.dim_inputs, out_features=dim_attention_global, bias=False),
            nn.Linear(in_features=dim_attention_global, out_features=self.num_units * self.dim_attention_unit,
                      bias=False)
        )

    def forward(self, x):
        x = x.view(-1, self.dim_inputs)
        # project to units
        attention = self.model(x)

        attention = attention.view(-1, self.num_units, self.dim_attention_unit)

        return attention

    def extra_repr(self) -> str:
        return 'num_units:{num_units}, ' \
               'dim_inputs:{dim_inputs}, ' \
               'dim_attention_global:{dim_attention_global}, ' \
               'dim_attention_unit:{dim_attention_unit}'.format(**self.__dict__)


def main():
    pass


if __name__ == '__main__':
    main()
