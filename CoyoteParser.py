from typing import Any

import gym.spaces
import numpy as np
from gym.spaces import Discrete
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.gamestates import GameState
from submodels.submodel_agent import SubAgent
from CoyoteObs import CoyoteObsBuilder

class CoyoteAction(ActionParser):
    def __init__(self):
        super().__init__()
        self._lookup_table = self.make_lookup_table()

    @staticmethod
    def make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 0.5, 1):
            for steer in (-1, -0.5, 0, 0.5, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, -0.75, -0.5, 0, 0.5, 0.75, 1):
            for yaw in (-1, -0.75, -0.5, 0, 0.5, 0.75, 1):
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
        # append stall
        actions.append([0, 1, 0, 0, -1, 1, 0, 0])
        actions = np.array(actions)
        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        # hacky pass through to allow multiple types of agent actions while still parsing nectos

        # strip out fillers, pass through 8sets, get look up table values, recombine
        parsed_actions = []
        for action in actions:
            # test
            # parsed_actions.append([0, 0, 0, 0, 0, 0, 0, 0])
            # continue
            # support reconstruction
            if action.size != 8:
                if action.shape == 0:
                    action = np.expand_dims(action, axis=0)
                # to allow different action spaces, pad out short ones (assume later unpadding in parser)
                action = np.pad(action.astype('float64'), (0, 8 - action.size), 'constant', constant_values=np.NAN)

            if np.isnan(action).any():  # it's been padded, delete to go back to original
                stripped_action = (action[~np.isnan(action)]).squeeze().astype('int')
                parsed_actions.append(self._lookup_table[stripped_action])
            else:
                parsed_actions.append(action)

        return np.asarray(parsed_actions)


class SelectorParser(ActionParser):
    def __init__(self, num_models):
        super().__init__()
        self.num_models = num_models
        self._lookup_table = self.make_lookup_table(self.num_models)
        self.models = [(SubAgent("kickoff_1.jit"), CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3)),
                       (SubAgent("kickoff_1.jit"), CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3)),
                       (SubAgent("kickoff_1.jit"), CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3)),
                       (SubAgent("kickoff_1.jit"), CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3)),
                       (SubAgent("kickoff_1.jit"), CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3)),
                       (SubAgent("kickoff_1.jit"), CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3)),
                       (SubAgent("kickoff_1.jit"), CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3)),
                       (SubAgent("kickoff_1.jit"), CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3)),
                       ]
        self.prev_action = None
        self.prev_model = None


    @staticmethod
    def make_lookup_table(num_models):
        actions = []
        for index in range(8):
            chosen = [0] * num_models
            chosen[index] = 1
            actions.append(chosen)
        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        # TODO: test all this shit
        if len(actions) != len(self.prev_model):
            self.prev_action = np.asarray([[None] * 8] * len(actions))
            self.prev_model = [None] * len(actions)
        parsed_actions = []
        for i, action in enumerate(actions):
            if self.prev_model[i] != action:
                self.prev_action[i] = None
            player = state.players[i]
            obs = self.models[action][1].build_obs(player, state, self.prev_action[i])
            parse_action = self.models[action][0].act(obs)
            self.prev_action[i] = np.asarray(parse_action)
            parsed_actions.append(parse_action)
        return np.asarray(parsed_actions)


if __name__ == '__main__':
    ap = CoyoteAction()
    print(ap.get_action_space())
