import os
import numpy as np
import torch
import torch.nn.functional as F
from rocket_learn.agent.pretrained_policy import HardcodedAgent
from CoyoteObs import CoyoteObsBuilder
from CoyoteParser import CoyoteAction
from rlgym.utils.gamestates import GameState
#import copy


class GP(HardcodedAgent):
    def __init__(self, model_string):
        cur_dir = os.path.dirname(__file__)
        self.actor = torch.jit.load(os.path.join(cur_dir, '..\\..\\submodels', model_string))
        self.obs_builder = CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, embed_players=True)
        self.previous_action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        self._lookup_table = CoyoteAction.make_lookup_table()

    def act(self, state: GameState, player_index: int):
        player = state.players[player_index]

        obs = self.obs_builder.build_obs(player, state, self.previous_action)
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        elif isinstance(obs, tuple):
            obs = tuple(o if isinstance(o, torch.Tensor) else torch.from_numpy(o).float() for o in obs)

        out = self.actor(obs)

        if isinstance(out, torch.Tensor):
            out = (out,)

        #out = (out,)
        max_shape = max(o.shape[-1] for o in out)
        logits = torch.stack(
            [
                l
                if l.shape[-1] == max_shape
                else F.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf"))
                for l in out
            ],
            dim=1
        )

        actions = np.argmax(logits, axis=-1)
        parsed = self._lookup_table[actions.numpy().item()]

        self.previous_action = parsed
        return parsed
