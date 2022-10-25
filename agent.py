from torch import nn
import torch as th
from torch.nn.init import xavier_uniform_


class Opti(nn.Module):  # takes an embedder and a network and runs the embedder on the car obs before passing to the network
    def __init__(self, embedder: nn.Module, net: nn.Module):
        super().__init__()
        self.embedder = embedder
        self.net = net
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, inp: tuple):
        main, cars = inp
        # shaped_cars = th.reshape(cars, (len(main), 5, len(cars[0])))
        out = th.max(self.embedder(cars), -2)[0]
        result = self.net(th.cat((main, out), dim=1))
        return result
