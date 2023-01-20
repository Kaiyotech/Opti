import os
import numpy as np
import torch
import torch.nn.functional as F
from rocket_learn.agent.pretrained_policy import HardcodedAgent
from pretrained_agents.KBB.KBBObs import AdvancedObsPadder
from rlgym.utils.gamestates import GameState
#import copy


class KBB(HardcodedAgent):
    def __init__(self, model_string):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.actor = torch.jit.load(os.path.join(cur_dir, model_string))
        self.obs_builder = AdvancedObsPadder(team_size=3, expanding=True)
        self.previous_action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        self._lookup_table = self.make_lookup_table()

    @staticmethod
    def make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        actions = np.array(actions)
        return actions

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
