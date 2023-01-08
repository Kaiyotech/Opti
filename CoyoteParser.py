import copy
from typing import Any

import gym.spaces
import numpy as np
from gym.spaces import Discrete
from rlgym.utils import math
from rlgym.utils.gamestates import PhysicsObject
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.gamestates import GameState
from CoyoteObs import CoyoteObsBuilder
from selection_listener import SelectionListener


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
                            actions.append(
                                [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
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
                                handbrake = jump == 1 and (
                                        pitch != 0 or yaw != 0 or roll != 0)
                                actions.append(
                                    [boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
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
                        actions.append(
                            [throttle or boost, steer, 0, steer, 0, 0, boost, 0])
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
                                actions.append(
                                    [boost, yaw, pitch, yaw, roll, jump, boost, 0])
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
                action = np.pad(action.astype(
                    'float64'), (0, 8 - action.size), 'constant', constant_values=np.NAN)

            if np.isnan(action).any():  # it's been padded, delete to go back to original
                stripped_action = (
                    action[~np.isnan(action)]).squeeze().astype('int')
                parsed_actions.append(self._lookup_table[stripped_action])
            else:
                parsed_actions.append(action)

        return np.asarray(parsed_actions)


def override_state(player, state, position_index) -> GameState:
    # takes the player and state and returns a new state based on the position index
    # which mocks specific values such as ball position, nearest opponent position, etc.
    # for use with the recovery model's observer builder
    retstate = copy.deepcopy(state)
    assert 10 <= position_index <= 21

    if player.team_num == 1:
        inverted = True
        player_car = player.inverted_car_data
        ball = state.inverted_ball
    else:
        inverted = False
        player_car = player.car_data
        ball = state.ball

    oppo_car = [
        (p.inverted_car_data if inverted else p.car_data) for p in state.players if
        p.team_num != player.team_num]
    oppo_car.sort(key=lambda c: np.linalg.norm(
        c.position - player_car.position))

    # Ball position first
    ball_pos = np.asarray([0, 0, 0])
    # 0 is straight in front, 1500 units away, 1 is diagonal front left, 7 is diagonal front right
    if position_index < 18:
        position_index = position_index - 10
        angle_rad = position_index * np.pi / 4
        fwd = player_car.forward()[:2]  # vector in forward direction just xy
        if abs(fwd[0]) < 0.01 and abs(fwd[1]) < 0.01:
            fwd = player_car.up()[:2]
        fwd = fwd / np.linalg.norm(fwd)  # make unit
        rot_fwd = np.asarray([fwd[0] * np.cos(angle_rad) - fwd[1] * np.sin(angle_rad),
                              fwd[0] * np.sin(angle_rad) + fwd[1] * np.cos(angle_rad)])
        # distance of 1500 in rotated direction
        forward_point = (1500 * rot_fwd) + player_car.position[:2]
        forward_point[0] = np.clip(forward_point[0], -4096, 4096)
        forward_point[1] = np.clip(forward_point[1], -5120, 5120)
        ball_pos = np.asarray([forward_point[0], forward_point[1], 40])
    elif position_index < 20:  # 18 and 19 are back left and back right boost
        if position_index == 18:
            ball_pos = np.asarray([3072, -4096, 40])
        elif position_index == 19:

            ball_pos = np.asarray([-3072, -4096, 40])

    elif position_index == 20:  # 20 is closest opponent
        ball_pos = oppo_car[0].position
    elif position_index == 21:  # 21 is back post entry, approx 1000, 4800
        x_pos = 1000
        if ball.position[0] >= 0:
            x_pos = -1000
        ball_pos = np.asarray([x_pos, -4800, 40])
    retstate.ball.position = ball_pos
    retstate.inverted_ball.position = ball_pos

    # Ball velocity next
    retstate.ball.linear_velocity = np.zeros(3)
    retstate.inverted_ball.linear_velocity = np.zeros(3)
    retstate.ball.angular_velocity = np.zeros(3)
    retstate.inverted_ball.angular_velocity = np.zeros(3)

    # Nearest player next
    player_car_ball_pos_vec = ball_pos[:2] - player_car.position[:2]
    player_car_ball_pos_vec /= np.linalg.norm(player_car_ball_pos_vec)
    # oppo_pos is 400 uu behind player
    oppo_pos = player_car.position[:2] - 400 * player_car_ball_pos_vec
    # Octane elevation at rest is 17.01uu
    oppo_pos = np.asarray([oppo_pos[0], oppo_pos[1], 17.01])
    oppo_yaw = np.arctan2(
        player_car_ball_pos_vec[1], player_car_ball_pos_vec[0])
    oppo_rot_cy = np.cos(oppo_yaw)
    oppo_rot_sy = np.sin(oppo_yaw)
    oppo_rot = np.array(((oppo_rot_cy, -oppo_rot_sy, 0),
                         (oppo_rot_sy, oppo_rot_cy, 0), (0, 0, 1)))
    # oppo_vel is max driving speed without boosting in direction of ball
    oppo_vel = [1410 * player_car_ball_pos_vec[0], 1410 * player_car_ball_pos_vec[1], 0]
    new_oppo_car_data = PhysicsObject(
        position=oppo_pos, quaternion=math.rotation_to_quaternion(oppo_rot), linear_velocity=oppo_vel)
    oppo_car_idx = len(state.players) // 2
    retstate.players[oppo_car_idx].car_data = new_oppo_car_data
    retstate.players[oppo_car_idx].inverted_car_data = new_oppo_car_data
    # make other opponents so they are definitely farther away and the obs only takes closest
    # and dummies the rest
    for i in range(oppo_car_idx + 1, len(state.players)):
        oppo_pos = player_car.position[:2] - 2500 * player_car_ball_pos_vec
        oppo_pos = np.asarray([oppo_pos[0], oppo_pos[1], 17.01 * i])
        retstate.players[i].car_data.position = oppo_pos
        retstate.players[i].inverted_car_data.position = oppo_pos
    return retstate


def override_abs_state(player, state, position_index) -> GameState:
    # takes the player and state and returns a new state based on the position index
    # which mocks specific values such as ball position, nearest opponent position, etc.
    # for use with the recovery model's observer builder
    retstate = copy.deepcopy(state)
    assert 10 <= position_index <= 21

    if player.team_num == 1:
        inverted = True
        player_car = player.inverted_car_data
        ball = state.inverted_ball
    else:
        inverted = False
        player_car = player.car_data
        ball = state.ball

    oppo_car = [
        (p.inverted_car_data if inverted else p.car_data) for p in state.players if
        p.team_num != player.team_num]
    oppo_car.sort(key=lambda c: np.linalg.norm(
        c.position - player_car.position))

    # Ball position first
    ball_pos = np.asarray([0, 0, 0])
    # 1500 uu away, 0 straight +y, 1 +x+y, 4 -y, 7 -x+y
    if position_index < 18:
        position_index = position_index - 10
        angle_rad = position_index * np.pi / 4
        fwd = np.asarray([0, 1])
        fwd = fwd / np.linalg.norm(fwd)  # make unit
        rot_fwd = np.asarray([fwd[0] * np.cos(angle_rad) - fwd[1] * np.sin(angle_rad),
                              fwd[0] * np.sin(angle_rad) + fwd[1] * np.cos(angle_rad)])
        # distance of 1500 in rotated direction
        forward_point = (1500 * rot_fwd) + player_car.position[:2]
        forward_point[0] = np.clip(forward_point[0], -4096, 4096)
        forward_point[1] = np.clip(forward_point[1], -5120, 5120)
        ball_pos = np.asarray([forward_point[0], forward_point[1], 40])
    elif position_index < 20:  # 18 and 19 are back left and back right boost
        if position_index == 18:
            ball_pos = np.asarray([3072, -4096, 40])
        elif position_index == 19:

            ball_pos = np.asarray([-3072, -4096, 40])

    elif position_index == 20:  # 20 is closest opponent
        ball_pos = oppo_car[0].position
    elif position_index == 21:  # 21 is back post entry, approx 1000, 4800
        x_pos = 1000
        if ball.position[0] >= 0:
            x_pos = -1000
        ball_pos = np.asarray([x_pos, -4800, 40])
    retstate.ball.position = ball_pos
    retstate.inverted_ball.position = ball_pos

    # Ball velocity next
    retstate.ball.linear_velocity = np.zeros(3)
    retstate.inverted_ball.linear_velocity = np.zeros(3)
    retstate.ball.angular_velocity = np.zeros(3)
    retstate.inverted_ball.angular_velocity = np.zeros(3)

    # Nearest player next
    player_car_ball_pos_vec = ball_pos[:2] - player_car.position[:2]
    player_car_ball_pos_vec /= np.linalg.norm(player_car_ball_pos_vec)
    # oppo_pos is 400 uu behind player
    oppo_pos = player_car.position[:2] - 400 * player_car_ball_pos_vec
    # Octane elevation at rest is 17.01uu
    oppo_pos = np.asarray([oppo_pos[0], oppo_pos[1], 17.01])
    oppo_yaw = np.arctan2(
        player_car_ball_pos_vec[1], player_car_ball_pos_vec[0])
    oppo_rot_cy = np.cos(oppo_yaw)
    oppo_rot_sy = np.sin(oppo_yaw)
    oppo_rot = np.array(((oppo_rot_cy, -oppo_rot_sy, 0),
                         (oppo_rot_sy, oppo_rot_cy, 0), (0, 0, 1)))
    # oppo_vel is max driving speed without boosting in direction of ball
    oppo_vel = [1410 * player_car_ball_pos_vec[0], 1410 * player_car_ball_pos_vec[1], 0]
    new_oppo_car_data = PhysicsObject(
        position=oppo_pos, quaternion=math.rotation_to_quaternion(oppo_rot), linear_velocity=oppo_vel)
    oppo_car_idx = len(state.players) // 2
    retstate.players[oppo_car_idx].car_data = new_oppo_car_data
    retstate.players[oppo_car_idx].inverted_car_data = new_oppo_car_data
    # make other opponents so they are definitely farther away and the obs only takes closest
    # and dummies the rest
    for i in range(oppo_car_idx + 1, len(state.players)):
        oppo_pos = player_car.position[:2] - 2500 * player_car_ball_pos_vec
        oppo_pos = np.asarray([oppo_pos[0], oppo_pos[1], 17.01 * i])
        retstate.players[i].car_data.position = oppo_pos
        retstate.players[i].inverted_car_data.position = oppo_pos
    return retstate

class SelectorParser(ActionParser):
    def __init__(self):
        from submodels.submodel_agent import SubAgent
        from Constants_selector import SUB_MODEL_NAMES
        self.sub_model_names = [
            name.replace("recover", "rush").replace("_", " ").title()
            for name in SUB_MODEL_NAMES
        ]
        self.selection_listener = None
        super().__init__()

        self.models = [(SubAgent("kickoff_1_jit.pt"), CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3)),
                       (SubAgent("kickoff_2_jit.pt"), CoyoteObsBuilder(
                           expanding=True, tick_skip=4, team_size=3)),
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
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                         only_closest_opp=True)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                         only_closest_opp=True)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                         only_closest_opp=True)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                         only_closest_opp=True)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                         only_closest_opp=True)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                         only_closest_opp=True)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                         only_closest_opp=True)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                         only_closest_opp=True)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                         only_closest_opp=True)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                         only_closest_opp=True)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                         only_closest_opp=True)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                         only_closest_opp=True)),
                       (SubAgent("recovery_jit.pt"),
                        CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                         only_closest_opp=True)),
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
            # override state for recovery

            newstate = state

            if 10 <= action <= 21:
                newstate = override_abs_state(player, state, action)

            obs = self.models[action][1].build_obs(
                player, newstate, self.prev_actions[i])
            parse_action = self.models[action][0].act(obs)[0]
            if self.selection_listener is not None:
                self.selection_listener.on_selection(self.sub_model_names[action], parse_action)
            # self.prev_action[i] = np.asarray(parse_action)
            self.prev_actions[i] = parse_action
            parsed_actions.append(parse_action)

        return np.asarray(parsed_actions)  # , np.asarray(actions)

    # necessary because of the stateful obs
    def reset(self, initial_state: GameState):
        for model in self.models:
            model[1].reset(initial_state)

    def register_selection_listener(self, listener: SelectionListener):
        self.selection_listener = listener


if __name__ == '__main__':
    ap = CoyoteAction()
    print(ap.get_action_space())
