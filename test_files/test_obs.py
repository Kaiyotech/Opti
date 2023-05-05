import math
import numpy as np
from typing import Any, List
from rlgym_sim.utils import common_values
from rlgym_sim.utils.gamestates import PlayerData, GameState
from rlgym_sim.utils.obs_builders import ObsBuilder
from collections import deque

from gym import Space
from gym.spaces import Tuple, Box


def print_state(state: GameState):
    print("State:")
    print(f"Score is {state.blue_score} - {state.orange_score}")
    print(f"Ball Info:")
    print(f"ang_vel: {state.ball.angular_velocity}")
    print(f"lin_vel: {state.ball.linear_velocity}")
    print(f"position: {state.ball.position}")
    print(f"qaut: {state.ball.quaternion}")
    print(f"Inv Ball Info:")
    print(f"ang_vel: {state.inverted_ball.angular_velocity}")
    print(f"lin_vel: {state.inverted_ball.linear_velocity}")
    print(f"position: {state.inverted_ball.position}")
    print(f"qaut: {state.inverted_ball.quaternion}")
    print(f"Boost_pads: {state.boost_pads}")
    print(f"inv_boost_pads: {state.inverted_boost_pads}")
    print("Player Info:")
    for player in state.players:
        print(f"car_data:")
        print(f"ang_vel: {player.car_data.angular_velocity}")
        print(f"lin_vel: {player.car_data.linear_velocity}")
        print(f"pos: {player.car_data.position}")
        print(f"quat: {player.car_data.quaternion}")
        print(f"inv_car_data:")
        print(f"ang_vel: {player.inverted_car_data.angular_velocity}")
        print(f"lin_vel: {player.inverted_car_data.linear_velocity}")
        print(f"pos: {player.inverted_car_data.position}")
        print(f"quat: {player.inverted_car_data.quaternion}")
        print(f"boost: {player.boost_amount}")
        print(f"boost pickups: {player.boost_pickups}")
        print(f"Is Demoed: {player.is_demoed}")
        print(f"on ground: {player.on_ground}")
        print(f"ball touched: {player.ball_touched}")
        print(f"has_jump: {player.has_jump}")
    print()


class TestObs(ObsBuilder):
    def __init__(self, pos_coef=1 / 2300, ang_coef=1 / math.pi, lin_vel_coef=1 / 2300, ang_vel_coef=1 / math.pi):
        """
        :param pos_coef: Position normalization coefficient
        :param ang_coef: Rotation angle normalization coefficient
        :param lin_vel_coef: Linear velocity normalization coefficient
        :param ang_vel_coef: Angular velocity normalization coefficient
        """
        super().__init__()
        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef
        self.states = deque(maxlen=50)
        self.steps = 0
        self.actions = deque(maxlen=50)


    def reset(self, initial_state: GameState):
        self.states.clear()
        self.actions.clear()
        self.steps = 0

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray,
                  previous_model_action: np.ndarray,
                  ) -> Any:
        self.steps += 1
        self.states.appendleft(state)
        self.actions.appendleft(previous_action)
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [ball.position * self.POS_COEF,
               ball.linear_velocity * self.LIN_VEL_COEF,
               ball.angular_velocity * self.ANG_VEL_COEF,
               previous_action,
               pads]

        self._add_player_to_obs(obs, player, inverted)

        allies = []
        enemies = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            self._add_player_to_obs(team_obs, other, inverted)

        obs.extend(allies)
        obs.extend(enemies)
        # obs = np.asarray(obs)
        # obs = obs.flatten()
        # obs = np.expand_dims(np.fromiter(obs, dtype=np.float32, count=len(obs)), 0)
        obs = np.concatenate(obs)
        # if np.isnan(obs).any():
        #     # self.actions.reverse()
        #     # self.states.reverse()
        #     print(f"There is a nan in the obs. Printing states")
        #     input()
            # print(f" {self.steps} since reset")
            # i = 0
            # for each_state in self.states:
            #     print(i)
            #     i += 1
            #     print_state(each_state)
            # print("printing actions")
            # i = 0
            # for each_action in self.actions:
            #     print(i)
            #     i += 1
            #     print(each_action)
            # exit()
        return obs

    def _add_player_to_obs(self, obs: List, player: PlayerData, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        obs.extend([
            player_car.position * self.POS_COEF,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity * self.LIN_VEL_COEF,
            player_car.angular_velocity * self.ANG_VEL_COEF,
            np.asarray([player.boost_amount,
            int(player.on_ground),
            int(player.has_flip),
            int(player.is_demoed)])])

        return player_car

    def get_obs_space(self) -> Space:
        players = 2
        car_size = 19
        player_size = 51 + 19
        return Box(-np.inf, np.inf, (1, (players * car_size) + player_size))
