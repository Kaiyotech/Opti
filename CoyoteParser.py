from rlgym.utils.gamestates import GameState, PlayerData, PhysicsObject
from rlgym.utils.action_parsers import ActionParser
import copy
from typing import Any

import gym.spaces
import numpy as np
from gym.spaces import Discrete

from rlgym.utils import math

from CoyoteObs import CoyoteObsBuilder, CoyoteObsBuilder_Legacy
from selection_listener import SelectionListener


class CoyoteAction(ActionParser):
    def __init__(self, version=None):
        super().__init__()
        self._lookup_table = self.make_lookup_table(version)
        # # TODO: remove this
        # self.angle = 0
        # self.counter = 0

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

        elif version == "test_dodge":
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
            for pitch in (-0.75, -0.75, -0.75, -0.75, -0.75, -0.75, -0.75):
                for yaw in (0, 0, 0, 0, 0, 0, 0):
                    for roll in (0, 0, 0):
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

        elif version == "test_setter":
            # Ground
            for throttle in (-1, 0, 0.5, 1):
                for steer in (-1, -0.5, 0, 0.5, 1):
                    for boost in (0, 1):
                        for handbrake in (0, 1):
                            if boost == 1 and throttle != 1:
                                continue
                            actions.append(
                                [1, 0, 0, 0, 0, 0, 0, 0])
            # Aerial
            for pitch in (-0.85, -0.84, -0.83, 0, 0.83, 0.84, 0.85):
                for yaw in (0, 0, 0, 0, 0, 0, 0):
                    for roll in (0, 0, 0):
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
                                    [1, 0, 0, 0, 0, 0, 0, 0])
            # append stall
            actions.append([1, 0, 0, 0, 0, 0, 0, 0])
            actions = np.array(actions)

        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._lookup_table))

    @staticmethod
    def get_model_action_space() -> int:
        return 1

    def get_model_action_size(self) -> int:
        return len(self._lookup_table)

    def parse_actions(self, actions: Any, state: GameState, zero_boost: bool = False) -> np.ndarray:

        # hacky pass through to allow multiple types of agent actions while still parsing nectos

        # strip out fillers, pass through 8sets, get look up table values, recombine
        parsed_actions = []
        for action in actions:
            # test
            # parsed_actions.append([, 0, 0, 0, 0, 0, 1, 0])
            # continue
            # support reconstruction
            # if action.size != 8:
            #     if action.shape == 0:
            #         action = np.expand_dims(action, axis=0)
            #     # to allow different action spaces, pad out short ones (assume later unpadding in parser)
            #     action = np.pad(action.astype(
            #         'float64'), (0, 8 - action.size), 'constant', constant_values=np.NAN)

            if np.isnan(action).any():  # it's been padded, delete to go back to original
                stripped_action = (
                    action[~np.isnan(action)]).squeeze().astype('int')

                done_action = copy.deepcopy(self._lookup_table[stripped_action])
                if zero_boost:
                    done_action[6] = 0
                parsed_actions.append(done_action)
            elif action.shape[0] == 1:
                action = copy.deepcopy(self._lookup_table[action[0].astype('int')])
                if zero_boost:
                    action[6] = 0
                parsed_actions.append(action)
            else:
                parsed_actions.append(action)
        # # TODO: remove this
        # self.counter += 1
        # if self.counter % 2:
        #     return np.array(
        #         [np.array([0., 0., 0, 0., 0., 0., 0., 0.]), np.array([0., 0., 0, 0., 0., 0., 0., 0.])])
        # else:
        #     return np.array([np.array([0., 0., self.angle, 0., 0., 1., 0., 0.]), np.array([0., 0., self.angle, 0., 0., 1., 0., 0.])])
        return np.asarray(parsed_actions)


def mirror_commands(actions):
    # [throttle, steer, pitch, yaw, roll, jump, boost, handbrake])
    actions[1] = -actions[1]
    actions[3] = -actions[3]
    actions[4] = -actions[4]


def mirror_state_over_y(player, state) -> GameState:
    retstate = copy_state(state)
    retstate.ball = state.ball
    retstate.inverted_ball = state.inverted_ball

    # mirror ball and all cars across the y-axis
    retstate.ball.position[0] = -retstate.ball.position[0]
    retstate.ball.linear_velocity[0] = -retstate.ball.linear_velocity[0]
    retstate.inverted_ball.position[0] = -retstate.inverted_ball.position[0]
    retstate.inverted_ball.linear_velocity[0] = -retstate.inverted_ball.linear_velocity[0]
    for car in retstate.players:
        car_position = np.asarray([-car.car_data.position[0], car.car_data.position[1], car.car_data.position[2]])
        car_linear_velocity = np.asarray([-car.car_data.linear_velocity[0], car.car_data.linear_velocity[1],
                                          car.car_data.linear_velocity[2]])
        car_angular_velocity = np.asarray([-car.car_data.angular_velocity[0], -car.car_data.angular_velocity[1],
                                           car.car_data.angular_velocity[2]])
        car_rotation = math.quat_to_euler(car.car_data.quaternion)
        car_rotation = np.asarray([car_rotation[0], -car_rotation[1], car_rotation[2]])
        new_car_data = PhysicsObject(
            position=car_position, quaternion=math.rotation_to_quaternion(math.euler_to_rotation(car_rotation)),
            linear_velocity=car_linear_velocity, angular_velocity=car_angular_velocity)
        car.car_data = new_car_data
        car.inverted_car_data = new_car_data
    return retstate


def override_state(player, state, position_index) -> GameState:
    return NotImplemented
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
        fwd = fwd / (np.linalg.norm(fwd) + 1e-8)  # make unit
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
    player_car_ball_pos_vec /= (np.linalg.norm(player_car_ball_pos_vec) + 1e-8)
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


def copy_state(state) -> GameState:
    # quickly copy state
    retstate = GameState()
    retstate.ball.quaternion = state.ball.quaternion.copy()
    retstate.inverted_ball.quaternion = state.inverted_ball.quaternion.copy()

    retstate.blue_score = state.blue_score
    retstate.orange_score = state.orange_score

    retstate.boost_pads = state.boost_pads
    retstate.inverted_boost_pads = state.inverted_boost_pads
    retstate.game_type = state.game_type
    retstate.last_touch = state.last_touch

    for i in range(0, len(state.players)):
        player = PlayerData()
        player.ball_touched = state.players[i].ball_touched
        player.boost_amount = state.players[i].boost_amount
        player.boost_pickups = state.players[i].boost_pickups

        player.car_data.angular_velocity = state.players[i].car_data.angular_velocity.copy()
        player.car_data.linear_velocity = state.players[i].car_data.linear_velocity.copy()
        player.car_data.position = state.players[i].car_data.position.copy()
        player.car_data.quaternion = state.players[i].car_data.quaternion.copy()

        player.inverted_car_data.angular_velocity = state.players[i].inverted_car_data.angular_velocity.copy()
        player.inverted_car_data.linear_velocity = state.players[i].inverted_car_data.linear_velocity.copy()
        player.inverted_car_data.position = state.players[i].inverted_car_data.position.copy()
        player.inverted_car_data.quaternion = state.players[i].inverted_car_data.quaternion.copy()

        player.car_id = state.players[i].car_id
        player.has_flip = state.players[i].has_flip
        player.has_jump = state.players[i].has_jump
        player.is_demoed = state.players[i].is_demoed
        player.match_demolishes = state.players[i].match_demolishes
        player.match_goals = state.players[i].match_goals
        player.match_saves = state.players[i].match_saves
        player.match_shots = state.players[i].match_shots
        player.on_ground = state.players[i].on_ground
        player.team_num = state.players[i].team_num

        retstate.players.append(player)

    return retstate


def override_abs_state(player, state, position_index, ball_position: np.ndarray = None) -> GameState:
    # takes the player and state and returns a new state based on the position index
    # which mocks specific values such as ball position, nearest opponent position, etc.
    # for use with the recovery model's observer builder
    # retstate = copy.deepcopy(state)
    team_size = len(state.players) // 2
    if player.team_num == 0:
        oppo_start = team_size
        oppo_stop = team_size * 2
    else:
        oppo_start = 0
        oppo_stop = team_size

    retstate = copy_state(state)
    assert 10 <= position_index <= 28

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

    recovery_distance = 3000
    # 21, 24 are actual ball, just override player
    if ball_position is None and (position_index != 21 and position_index != 24):
        # Ball position first
        ball_pos = np.asarray([0, 0, 0])
        # 2000 uu away, 0 straight +y, 1 +x+y, 4 -y, 7 -x+y
        if position_index < 18:
            position_index = position_index - 10
            angle_rad = position_index * np.pi / 4
            fwd = np.asarray([0, 1])
            fwd = fwd / (np.linalg.norm(fwd) + 1e-8)  # make unit
            rot_fwd = np.asarray([fwd[0] * np.cos(angle_rad) - fwd[1] * np.sin(angle_rad),
                                  fwd[0] * np.sin(angle_rad) + fwd[1] * np.cos(angle_rad)])
            # distance of 2000 in rotated direction
            forward_point = (recovery_distance * rot_fwd) + player_car.position[:2]
            forward_point[0] = np.clip(forward_point[0], -4096, 4096)
            forward_point[1] = np.clip(forward_point[1], -5120, 5120)
            ball_pos = np.asarray([forward_point[0], forward_point[1], 94])
        elif position_index < 20:  # 18 and 19 are back left and back right boost
            if position_index == 18:
                ball_pos = np.asarray([3072, -4096, 94])
            elif position_index == 19:
                ball_pos = np.asarray([-3072, -4096, 94])

        # elif position_index == 20:  # 20 is closest opponent
        #     ball_pos = oppo_car[0].position
        #     ball_pos[2] = 94
        elif position_index == 20:  # 20 is back post entry, approx 1000, 4800
            x_pos = 1000
            if ball.position[0] >= 0:
                x_pos = -1000
            ball_pos = np.asarray([x_pos, -4800, 94])
        elif position_index == 22:  # 22 is behind for half-flip, relative
            angle_rad = np.pi
            fwd = player_car.forward()[:2]  # vector in forward direction just xy
            if abs(fwd[0]) < 0.01 and abs(fwd[1]) < 0.01:
                fwd = player_car.up()[:2]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-8)  # make unit
            rot_fwd = np.asarray([fwd[0] * np.cos(angle_rad) - fwd[1] * np.sin(angle_rad),
                                  fwd[0] * np.sin(angle_rad) + fwd[1] * np.cos(angle_rad)])
            # distance of 2500 in rotated direction
            forward_point = (2500 * rot_fwd) + player_car.position[:2]
            forward_point[0] = np.clip(forward_point[0], -4096, 4096)
            forward_point[1] = np.clip(forward_point[1], -5120, 5120)
            ball_pos = np.asarray([forward_point[0], forward_point[1], 94])
        elif position_index == 23:  # 23 is forward, relative
            angle_rad = 0
            fwd = player_car.forward()[:2]  # vector in forward direction just xy
            if abs(fwd[0]) < 0.01 and abs(fwd[1]) < 0.01:
                fwd = player_car.up()[:2]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-8)  # make unit
            rot_fwd = np.asarray([fwd[0] * np.cos(angle_rad) - fwd[1] * np.sin(angle_rad),
                                  fwd[0] * np.sin(angle_rad) + fwd[1] * np.cos(angle_rad)])
            # distance of 2300 in rotated direction
            forward_point = (2300 * rot_fwd) + player_car.position[:2]
            forward_point[0] = np.clip(forward_point[0], -4096, 4096)
            forward_point[1] = np.clip(forward_point[1], -5120, 5120)
            ball_pos = np.asarray([forward_point[0], forward_point[1], 94])
        elif position_index == 25:  # same z
            fwd = player_car.forward()[1]  # vector in forward direction just y
            if abs(fwd) == 0:
                fwd = player_car.forward()[0]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-8)  # make unit (just get sign basically)
            # distance of 1700
            y_pos = (1700 * fwd) + player_car.position[1]
            y_pos = np.clip(y_pos, -3900, 3900)
            ball_pos = np.asarray([player_car.position[0], y_pos, player_car.position[2]])
        elif position_index == 26:  # up 45
            # space until ceiling
            z_space = max(1, 1700 - player_car.position[2])
            length = z_space / 0.707
            fwd = player_car.forward()[1]  # vector in forward direction just y
            if abs(fwd) == 0:
                fwd = player_car.forward()[0]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-8)  # make unit (just get sign basically)
            # distance of length to keep 45 degrees until ceiling/ground
            y_pos = (0.707 * length * fwd) + player_car.position[1]
            y_pos = np.clip(y_pos, -3900, 3900)
            z_pos = (0.707 * length) + player_car.position[2]
            z_pos = np.clip(z_pos, 300, 1750)
            ball_pos = np.asarray([player_car.position[0], y_pos, z_pos])
        elif position_index == 27:  # down 45
            # space until ground
            z_space = max(1, player_car.position[2] - 300)
            length = z_space / 0.707
            fwd = player_car.forward()[1]  # vector in forward direction just y
            if abs(fwd) == 0:
                fwd = player_car.forward()[0]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-8)  # make unit (just get sign basically)
            # distance of length to keep 45 degrees until ceiling/ground
            y_pos = (0.707 * length * fwd) + player_car.position[1]
            y_pos = np.clip(y_pos, -3900, 3900)
            z_pos = player_car.position[2] - (0.707 * length)
            z_pos = np.clip(z_pos, 300, 1750)
            ball_pos = np.asarray([player_car.position[0], y_pos, z_pos])
        elif position_index == 28:  # back boost this side
            x_pos = 3072 if player_car.position[0] >= 0 else -3072
            ball_pos = np.asarray([x_pos, -4096, 40])

    elif position_index == 21 or position_index == 24:
        ball_pos = state.inverted_ball.position if inverted else state.ball.position

    # override with passed in ball position
    else:
        ball_pos = ball_position

    retstate.ball.position = ball_pos
    retstate.inverted_ball.position = ball_pos

    # Ball velocity next
    if position_index != 21 and position_index != 24:
        retstate.ball.linear_velocity = np.zeros(3)
        retstate.inverted_ball.linear_velocity = np.zeros(3)
        retstate.ball.angular_velocity = np.zeros(3)
        retstate.inverted_ball.angular_velocity = np.zeros(3)
    # elif position_index == 20:
    #     retstate.ball.linear_velocity = oppo_car[0].linear_velocity
    #     retstate.inverted_ball.linear_velocity = oppo_car[0].linear_velocity
    #     retstate.ball.angular_velocity = oppo_car[0].angular_velocity
    #     retstate.inverted_ball.angular_velocity = oppo_car[0].angular_velocity
    else:
        retstate.ball = state.ball
        retstate.inverted_ball = state.inverted_ball

    # Nearest player next
    player_car_ball_pos_vec = ball_pos[:2] - player_car.position[:2]
    player_car_ball_pos_vec /= (np.linalg.norm(player_car_ball_pos_vec) + 1e-8)
    # oppo_pos is 200 uu behind player
    oppo_pos = player_car.position[:2] - 200 * player_car_ball_pos_vec
    # ready to wavedash, so slightly up
    oppo_pos = np.asarray([oppo_pos[0], oppo_pos[1], 100])
    oppo_yaw = np.arctan2(
        player_car_ball_pos_vec[1], player_car_ball_pos_vec[0])
    oppo_rot_cy = np.cos(oppo_yaw)
    oppo_rot_sy = np.sin(oppo_yaw)
    oppo_rot = np.array(((oppo_rot_cy, -oppo_rot_sy, 0),
                         (oppo_rot_sy, oppo_rot_cy, 0), (0, 0, 1)))
    # oppo_vel is same vel as player
    oppo_vel = player_car.linear_velocity
    new_oppo_car_data = PhysicsObject(
        position=oppo_pos, quaternion=math.rotation_to_quaternion(oppo_rot), linear_velocity=oppo_vel)
    retstate.players[oppo_start].car_data = new_oppo_car_data
    retstate.players[oppo_start].inverted_car_data = new_oppo_car_data
    retstate.players[oppo_start].boost_amount = 1  # give the opponent full boost
    # make other opponents so they are definitely farther away and the obs only takes closest
    # and dummies the rest
    for i in range(oppo_start + 1, oppo_stop):
        oppo_pos = player_car.position[:2] - 5000 * player_car_ball_pos_vec
        oppo_pos = np.asarray([oppo_pos[0], oppo_pos[1], 17.01 * i])
        retstate.players[i].car_data.position = oppo_pos
        retstate.players[i].inverted_car_data.position = oppo_pos
    return retstate


class SelectorParser(ActionParser):
    def __init__(self, obs_info=None):
        # if simulator:
        #     from rlgym_sim.utils.gamestates import GameState
        # else:
        #     from rlgym.utils.gamestates import GameState

        from submodels.submodel_agent import SubAgent
        from Constants_selector import SUB_MODEL_NAMES
        self.sub_model_names = [
            name.replace("recover", "rush").replace("_", " ").title()
            for name in SUB_MODEL_NAMES
        ]
        self.selection_listener = None
        self.ball_position = np.zeros([6, 3])
        # self.obs_output = obs_output
        self.obs_info = obs_info
        self.force_selector_choice = None
        super().__init__()

        self.models = [
            (SubAgent("kickoff_1_jit.pt"), CoyoteObsBuilder_Legacy(expanding=True, tick_skip=4, team_size=3)),
            (SubAgent("kickoff_2_jit.pt"), CoyoteObsBuilder_Legacy(
                expanding=True, tick_skip=4, team_size=3)),
            (SubAgent("gp_jit.pt"),
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, embed_players=True)),
            (SubAgent("aerial_jit.pt"),
             CoyoteObsBuilder_Legacy(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                     mask_aerial_opp=True)),
            (SubAgent("flick_1_jit.pt"),
             CoyoteObsBuilder_Legacy(expanding=True, tick_skip=4, team_size=3, embed_players=True)),
            (SubAgent("flick_2_jit.pt"),
             CoyoteObsBuilder_Legacy(expanding=True, tick_skip=4, team_size=3, embed_players=True)),
            (SubAgent("flipreset_1_jit.pt"),
             CoyoteObsBuilder_Legacy(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                     mask_aerial_opp=True)),
            (SubAgent("flipreset_2_jit.pt"),
             CoyoteObsBuilder_Legacy(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                     mask_aerial_opp=True)),
            (SubAgent("flipreset_3_jit.pt"),
             CoyoteObsBuilder_Legacy(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                                     mask_aerial_opp=True)),
            (SubAgent("pinch_jit.pt"),
             CoyoteObsBuilder_Legacy(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False)),
            (SubAgent("recovery_jit.pt"),  # 10 straight +y
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("recovery_jit.pt"),  # +x +y
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("recovery_jit.pt"),  # direction
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("recovery_jit.pt"),  # direction
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("recovery_jit.pt"),  # direction
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("recovery_jit.pt"),  # 15  direction
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("recovery_jit.pt"),  # direction
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("recovery_jit.pt"),  # 17 direction end (-x +y)
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("recovery_jit.pt"),  # 18 back left boost
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("recovery_jit.pt"),  # 19 back right boost
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("recovery_jit.pt"),  # 20, used to be closest opponent, now back post entry
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("recovery_jit.pt"),  # 21, actual ball
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("halfflip_jit.pt"),  # 22 behind, freeze
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("halfflip_jit.pt"),  # 23 forward (for the speedflip), freeze, relative
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("halfflip_jit.pt"),  # 24 actual ball
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("walldash_jit.pt"),  # 25 same z
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=1, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("walldash_jit.pt"),  # 26 up 45
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=1, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("walldash_jit.pt"),  # 27 down 45
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=1, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("walldash_jit.pt"),  # 28 back boost this side
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=1, extra_boost_info=False,
                              only_closest_opp=True, add_fliptime=True, add_airtime=True, add_jumptime=True,
                              add_handbrake=True, add_boosttime=True)),
            (SubAgent("demo_jit.pt"),  # demo
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True)),
            (SubAgent("dtap_jit.pt"),  # doubletap
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              mask_aerial_opp=True, doubletap_indicator=True)),
            (SubAgent("wall_jit.pt"),  # 31 ball to wall
             CoyoteObsBuilder(expanding=True, tick_skip=4, team_size=3, extra_boost_info=False,
                              only_closest_opp=True)),
            # 32 is turn left with throttle, 33 is straight with throttle, 34 is right
        ]
        self._lookup_table = self.make_lookup_table(len(self.models))
        # self.prev_action = None
        # self.prev_model = None
        self.prev_actions = np.asarray([[0] * 8] * 8)
        self.prev_model_actions = np.asarray([-1] * 6)

    @staticmethod
    def make_lookup_table(num_models):
        actions = []
        for index in range(8):
            chosen = [0] * num_models
            chosen[index] = 1
            actions.append(chosen)
        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._lookup_table), 1)

    @staticmethod
    def get_model_action_space() -> int:
        return 1

    def get_model_action_size(self) -> int:
        return len(self.models) + 3  # plus 3 for the left/straight/right

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        # for models in self.models:
        #     models[1].pre_step(state)
        self.obs_info.pre_step(state)

        parsed_actions = []
        for i, action in enumerate(actions):
            mirrored = False
            # if self.prev_model[i] != action:
            #     self.prev_action[i] = None
            action = int(action[0])  # change ndarray [0.] to 0
            # TODO remove this testing
            action = 31
            zero_boost = bool(action >= self.get_model_action_size())  # boost action 1 means no boost usage
            if action >= self.get_model_action_size():
                action -= self.get_model_action_size()

            # run timers for stateful obs for flips and such
            player = state.players[i]
            self.obs_info.step(player, state, self.prev_actions[i])

            if 32 <= action <= 34:  # simple steer goes here
                # [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
                steer = action - 33
                parse_action = np.asarray([1, steer, 0, steer, 0, 0, not zero_boost, 0])
            else:
                # override states

                newstate = state
                # 31 is wall, which gets mirrored if blue x negative or orange x positive for car
                if action == 31:
                    if (player.team_num == 0 and player.car_data.position[0] < 0) or \
                            (player.team_num == 1 and player.car_data.position[0] > 0):
                        newstate = mirror_state_over_y(player, state)
                        mirrored = True

                # 21, 24 are actual ball, just override player
                if 10 <= action <= 28:
                    newstate = override_abs_state(player, state, action)

                if 10 <= action <= 28:
                    # if reaching the "ball" or ball soon, allow a new choice by selector
                    check_radius = 300
                    if action in [11, 13, 15, 17]:  # these are the 45 degree ones, need bigger radius to reach
                        check_radius = 800
                    self.force_selector_choice[i] = check_terminal_selector(newstate, player, check_radius=check_radius)

                if 22 <= action <= 23:  # freeze
                    if self.prev_model_actions[i] == action:  # action didn't change
                        newstate = override_abs_state(player, state, action, self.ball_position[i])
                    else:  # action submodel changed or reaching the objective
                        newstate = override_abs_state(player, state, action)
                        self.ball_position[i] = state.ball.position  # save new position

                obs = self.models[action][1].build_obs(
                    player, newstate, self.prev_actions[i], obs_info=self.obs_info, zero_boost=zero_boost, n_override=i)
                parse_action = self.models[action][0].act(obs, zero_boost=zero_boost)[0]
                if mirrored:
                    mirror_commands(parse_action)

            if self.selection_listener is not None and i == 0:  # only call for first player
                self.selection_listener.on_selection(self.sub_model_names[action], parse_action)
            # self.prev_action[i] = np.asarray(parse_action)
            self.prev_actions[i] = parse_action
            self.prev_model_actions[i] = action
            parsed_actions.append(parse_action)

        return np.asarray(parsed_actions)  # , np.asarray(actions)

    # necessary because of the stateful obs
    def reset(self, initial_state: GameState):
        # add this back if one of them has an action stacker I guess, but no submodels do so :)
        # for model in self.models:
        #     model[1].reset(initial_state)
        self.obs_info.reset(initial_state)
        self.prev_model_actions = np.asarray([-1] * 6)

    def register_selection_listener(self, listener: SelectionListener):
        self.selection_listener = listener


def check_terminal_selector(state: GameState, player: PlayerData, check_radius=300) -> bool:
    if np.linalg.norm(player.car_data.position - state.ball.position) < check_radius:
        return True
    return False


if __name__ == '__main__':
    ap = CoyoteAction()
    print(ap.get_action_space())
