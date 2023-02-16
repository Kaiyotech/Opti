from torch import nn
import torch as th
import numpy as np
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.distributions import Categorical
from rocket_learn.agent.policy import Policy
from typing import Optional, List, Tuple


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


class OptiSelector(nn.Module):  # takes an embedder and a network and runs the embedder on the car obs before passing to the network
    # then outputs a tuple of output with action size, 1
    def __init__(self, embedder: nn.Module, net: nn.Module, shape: Tuple[int, ...]):
        super().__init__()
        self.embedder = embedder
        self.net = net
        self._reset_parameters()
        self.shape = shape

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, inp: tuple):
        main, cars = inp
        out = th.max(self.embedder(cars), -2)[0]
        result = self.net(th.cat((main, out), dim=1))
        result = result.split(self.shape, 1)
        return result


class MaskIndices(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def forward(self, x):
        return x[..., ~self.indices]


class MultiDiscretePolicy(Policy):

    def __init__(self, net: nn.Module, shape: Tuple[int, ...] = (3,) * 5 + (2,) * 3, deterministic=False):
        return NotImplemented
        super().__init__(deterministic)
        self.net = net
        self.shape = shape

    def forward(self, obs):
        logits = self.net(obs)
        return logits

    def get_action_distribution(self, obs):
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        elif isinstance(obs, tuple):
            obs = tuple(o if isinstance(o, th.Tensor) else th.from_numpy(o).float() for o in obs)

        logits = self(obs)

        if isinstance(logits, th.Tensor):
            logits = (logits,)

        max_shape = max(self.shape)
        logits = th.stack(
            [
                l
                if l.shape[-1] == max_shape
                else F.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf"))
                for l in logits
            ],
            dim=1
        )

        return Categorical(logits=logits)

    def sample_action(
        self,
        distribution: Categorical,
        deterministic=None
    ):
        if deterministic is None:
            deterministic = self.deterministic
        if deterministic:
            action_indices = th.argmax(distribution.logits, dim=-1)
        else:
            action_indices = distribution.sample()

        return action_indices

    def log_prob(self, distribution: Categorical, selected_action):
        log_prob = distribution.log_prob(selected_action).sum(dim=-1)
        return log_prob

    def entropy(self, distribution: Categorical, selected_action):
        entropy = distribution.entropy().sum(dim=-1)
        return entropy

    def env_compatible(self, action):
        if isinstance(action, th.Tensor):
            action = action.numpy()
        return action
