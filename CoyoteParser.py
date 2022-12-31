import copy
from typing import Any

import gym.spaces
import numpy as np
from gym.spaces import Discrete
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.gamestates import GameState
from CoyoteObs import CoyoteObsBuilder


class CoyoteAction(ActionParser):
    def __init__(self, version=None):
        super().__init__()
        self._lookup_table = self.make_lookup_table(version)

    @staticmethod
    def make_lookup_table(version):
        actions = []
        if version is None or version == "Normal":
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

        elif version == "flip_reset":
            # Ground
            for throttle in (-1, 0, 1):
                for steer in (-1, 0, 1):
                    for boost in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, 0])
            # Aerial
            for pitch in (-1, 0, 1):
                for yaw in (-1, 0, 1):
                    for roll in (-1, 0, 1):
                        for jump in (0, 1):
                            for boost in (0, 1):
                                if jump == 1 and yaw != 0 or roll != 0 or pitch != 0:  # no flips necessary here
                                    continue
                                if pitch == roll == jump == 0:  # Duplicate with ground
                                    continue
                                # Enable handbrake for potential wavedashes
                                actions.append([boost, yaw, pitch, yaw, roll, jump, boost, 0])
            # append stall
            # actions.append([0, 1, 0, 0, -1, 1, 0, 0])
            actions = np.array(actions)

        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._lookup_table))

    @staticmethod
    def get_model_action_space() -> int:
        return 1

    def get_model_action_size(self) -> int:
        return len(self._lookup_table)

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


def override_ball(player, state, position_index) -> np.ndarray:
    # takes the player and ball and returns a new ball position based on the position index
    # 0 is straight in front, 1500 units away, 1 is diagonal front left, 7 is diagonal front right
    angle_rad = position_index * np.pi / 4
    fwd = player.car_data.forward()[:2]  # unit vector in forward direction just xy
    rot_fwd = np.asarray([fwd[0] * np.cos(angle_rad) - fwd[1] * np.sin(angle_rad),
                          fwd[0] * np.sin(angle_rad) + fwd[1] * np.cos(angle_rad)])
    forward_point = (1500 * rot_fwd) + player.car_data.position[:2]  # distance of 1500 in rotated direction
    return np.asarray([forward_point[0], forward_point[1], 0])


class SelectorParser(ActionParser):
    def __init__(self):
        from submodels.submodel_agent import SubAgent
        super().__init__()

        self.models = [(SubAgent("kickoff_1_jit.pt"), CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3)),
                       (SubAgent("kickoff_2_jit.pt"), CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3)),
                       (SubAgent("gp_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, embed_players=True)),
                       (SubAgent("aerial_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False)),
                       (SubAgent("flick_1_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, embed_players=True)),
                       (SubAgent("flick_2_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, embed_players=True)),
                       (SubAgent("flipreset_1_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False)),
                       (SubAgent("flipreset_2_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False)),
                       (SubAgent("flipreset_3_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False)),
                       (SubAgent("pinch_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False)),
                       ]
        self._lookup_table = self.make_lookup_table(len(self.models))
        # self.prev_action = None
        # self.prev_model = None
        self.prev_actions = np.asarray([[0] * 8] * 8)

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

    @staticmethod
    def get_model_action_space() -> int:
        return 1

    def get_model_action_size(self) -> int:
        return len(self.models)

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:

        for models in self.models:
            models[1].pre_step(state)

        parsed_actions = []
        for i, action in enumerate(actions):
            # if self.prev_model[i] != action:
            #     self.prev_action[i] = None
            action = int(action)  # change ndarray [0.] to 0
            player = state.players[i]
            # override ball for recovery
            actual_ball = copy.deepcopy(state.ball.position)
            if action >= 10:
                state.ball.position = override_ball(player, state, action - 10)
            obs = self.models[action][1].build_obs(player, state, self.prev_actions[i])
            parse_action = self.models[action][0].act(obs)[0]
            # self.prev_action[i] = np.asarray(parse_action)
            self.prev_actions[i] = parse_action
            parsed_actions.append(parse_action)
            state.ball.position = copy.deepcopy(actual_ball)
        return np.asarray(parsed_actions)  # , np.asarray(actions)

    # necessary because of the stateful obs
    def reset(self, initial_state: GameState):
        for model in self.models:
            model[1].reset(initial_state)


if __name__ == '__main__':
    ap = CoyoteAction()
    print(ap.get_action_space())
