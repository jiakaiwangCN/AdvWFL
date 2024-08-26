import torch


class MixtureNorm1d(torch.nn.Module):

    def __init__(self, Out_channel, data_types):
        super(MixtureNorm1d, self).__init__()
        norms = [torch.nn.BatchNorm1d(Out_channel) for i in range(len(data_types))]

        self.norm_dic = {
            data_types[i]: norms[i] for i in range(len(data_types))
        }

    def forward(self, x, data_type='normal'):
        norm = self.norm_dic[data_type]
        if x.device != list(norm.parameters())[0].device:
            norm.to(x.device)
        return norm(x)
