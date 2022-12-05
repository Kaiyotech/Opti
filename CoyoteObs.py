import math
import random

import numpy as np
from typing import Any, List

from gym import Space
from gym.spaces import Tuple, Box
from rlgym.gym import Gym
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.common_values import BOOST_LOCATIONS
from rocket_learn.utils.scoreboard import Scoreboard
from collections.abc import Iterable


# inspiration from Raptor (Impossibum) and Necto (Rolv/Soren)
class CoyoteObsBuilder(ObsBuilder):
    def __init__(self, tick_skip=8, team_size=3, expanding: bool = True, extra_boost_info: bool = True,
                 embed_players=False, stack_size=0, action_parser=None, env: Gym = None, infinite_boost_odds=0,
                 only_closest_opp=False,
                 ):
        super().__init__()
        self.expanding = expanding
        self.only_closest_opp = only_closest_opp
        self.extra_boost_info = extra_boost_info
        self.POS_STD = 2300
        self.VEL_STD = 2300
        self.ANG_STD = 5.5
        self.BOOST_TIMER_STD = 10
        self.DEMO_TIMER_STD = 3
        self.dummy_player = [0] * 35
        self.boost_locations = np.array(BOOST_LOCATIONS)
        self.inverted_boost_locations = self.boost_locations[::-1]
        self.boost_timers = np.zeros(self.boost_locations.shape[0])
        self.inverted_boost_timers = np.zeros(self.boost_locations.shape[0])
        self.boosts_availability = np.zeros(self.boost_locations.shape[0])
        self.inverted_boosts_availability = np.zeros(self.boost_locations.shape[0])
        self.boost_objs = []
        self.inverted_boost_objs = []
        self.state = None
        self.time_interval = tick_skip / 120
        self.demo_timers = None
        self.num_players = team_size * 2
        self.generic_obs = None
        self.blue_obs = None
        self.orange_obs = None
        self.embed_players = embed_players
        self.default_action = np.zeros(8)
        self.stack_size = stack_size
        self.action_stacks = {}
        self.action_size = self.default_action.shape[0]
        self.action_parser = action_parser
        self.env = env
        self.infinite_boost_odds = infinite_boost_odds
        self.infinite_boost_episode = False

    def reset(self, initial_state: GameState):
        self.state = None
        self.boost_timers = np.zeros(self.boost_locations.shape[0])
        self.inverted_boost_timers = np.zeros(self.boost_locations.shape[0])
        self.demo_timers = np.zeros(max(p.car_id for p in initial_state.players) + 1)
        self.blue_obs = []
        self.orange_obs = []

        self.action_stacks = {}
        if self.stack_size != 0:
            for p in initial_state.players:
                self.action_stacks[p.car_id] = np.concatenate([self.default_action] * self.stack_size)

        if self.action_parser is not None:
            self.action_parser.reset(initial_state)

        if self.env is not None:
            if random.random() <= self.infinite_boost_odds:
                self.env.update_settings(boost_consumption=0)
                self.infinite_boost_episode = True
            else:
                self.env.update_settings(boost_consumption=1)
                self.infinite_boost_episode = False

    def pre_step(self, state: GameState):
        self.state = state
        # create player/team agnostic items (do these even exist?)
        self._update_timers(state)
        # create team specific things
        self.blue_obs = self.boost_timers / self.BOOST_TIMER_STD
        self.orange_obs = self.inverted_boost_timers / self.BOOST_TIMER_STD

        if self.env is not None:
            if self.infinite_boost_episode:
                for player in state.players:
                    player.boost_amount = 2
            else:
                for player in state.players:
                    player.boost_amount /= 1

    def _update_timers(self, state: GameState):
        current_boosts = state.boost_pads
        boost_locs = self.boost_locations
        demo_states = [[p.car_id, p.is_demoed] for p in state.players]

        for i in range(len(current_boosts)):
            if current_boosts[i] == self.boosts_availability[i]:
                if self.boosts_availability[i] == 0:
                    self.boost_timers[i] = max(0, self.boost_timers[i] - self.time_interval)
            else:
                if self.boosts_availability[i] == 0:
                    self.boosts_availability[i] = 1
                    self.boost_timers[i] = 0
                else:
                    self.boosts_availability[i] = 0
                    if boost_locs[i][2] == 73:
                        self.boost_timers[i] = 10.0
                    else:
                        self.boost_timers[i] = 4.0
        self.boosts_availability = current_boosts
        self.inverted_boost_timers = self.boost_timers[::-1]
        self.inverted_boosts_availability = self.boosts_availability[::-1]

        for cid, dm in demo_states:
            if dm == True:  # Demoed
                prev_timer = self.demo_timers[cid]
                if prev_timer > 0:
                    self.demo_timers[cid] = max(0, prev_timer - self.time_interval)
                else:
                    self.demo_timers[cid] = 3
            else:  # Not demoed
                self.demo_timers[cid] = 0

    def create_ball_packet(self, ball: PhysicsObject):
        p = [
            ball.position[0] / self.POS_STD, ball.position[1] / self.POS_STD, ball.position[2] / self.POS_STD,
            ball.linear_velocity[0] / self.VEL_STD, ball.linear_velocity[1] / self.VEL_STD, ball.linear_velocity[2] / self.VEL_STD,
            ball.angular_velocity[0] / self.ANG_STD, ball.angular_velocity[1] / self.ANG_STD, ball.angular_velocity[2] / self.ANG_STD,
            math.sqrt(ball.linear_velocity[0] ** 2 + ball.linear_velocity[1] ** 2 + ball.linear_velocity[2] ** 2)/2300,
            int(ball.position[2] <= 100),
        ]
        return p

    def create_player_packet(self, player: PlayerData, car: PhysicsObject, ball: PhysicsObject, prev_act: np.ndarray):
        pos_diff = ball.position - car.position
        vel_diff = ball.linear_velocity - car.linear_velocity
        fwd = car.forward()
        up = car.up()
        p = [
            car.position[0] / self.POS_STD, car.position[1] / self.POS_STD, car.position[2] / self.POS_STD,
            car.linear_velocity[0] / self.VEL_STD, car.linear_velocity[1] / self.VEL_STD, car.linear_velocity[2] / self.VEL_STD,
            car.angular_velocity[0] / self.ANG_STD, car.angular_velocity[1] / self.ANG_STD, car.angular_velocity[2] / self.ANG_STD,
            pos_diff[0] / self.POS_STD, pos_diff[1] / self.POS_STD, pos_diff[2] / self.POS_STD,
            vel_diff[0] / self.VEL_STD, vel_diff[1] / self.VEL_STD, vel_diff[2] / self.VEL_STD,
            fwd[0], fwd[1], fwd[2],
            up[0], up[1], up[2],
            math.sqrt(car.linear_velocity[0] ** 2 + car.linear_velocity[1] ** 2 + car.linear_velocity[2] ** 2)/2300,
            player.boost_amount,
            int(player.on_ground),
            int(player.has_flip),
            int(player.is_demoed),
            int(player.has_jump),
            self.demo_timers[player.car_id] / self.DEMO_TIMER_STD,
            prev_act[0], prev_act[1], prev_act[2], prev_act[3], prev_act[4], prev_act[5], prev_act[6], prev_act[7],
        ]
        if self.stack_size != 0:
            self.add_action_to_stack(prev_act, player.car_id)
            p.extend(list(self.action_stacks[player.car_id]))

        return p

    def create_car_packet(self, player_car: PhysicsObject, car: PhysicsObject,
                          _car: PlayerData, ball: PhysicsObject, teammate: bool):
        diff = car.position - player_car.position
        magnitude = math.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
        car_play_vel = car.linear_velocity - player_car.linear_velocity
        pos_diff = ball.position - car.position
        ball_car_vel = ball.linear_velocity - car.linear_velocity
        fwd = car.forward()
        up = car.up()
        p = [
                car.position[0] / self.POS_STD, car.position[1] / self.POS_STD, car.position[2] / self.POS_STD,
                car.linear_velocity[0] / self.VEL_STD, car.linear_velocity[1] / self.VEL_STD, car.linear_velocity[2] / self.VEL_STD,
                car.angular_velocity[0] / self.ANG_STD, car.angular_velocity[1] / self.ANG_STD, car.angular_velocity[2] / self.ANG_STD,
                diff[0] / self.POS_STD, diff[1] / self.POS_STD, diff[2] / self.POS_STD,
                car_play_vel[0] / self.VEL_STD, car_play_vel[1] / self.VEL_STD, car_play_vel[2] / self.VEL_STD,
                pos_diff[0] / self.POS_STD, pos_diff[1] / self.POS_STD, pos_diff[2] / self.POS_STD,
                ball_car_vel[0] / self.VEL_STD, ball_car_vel[1] / self.VEL_STD, ball_car_vel[2] / self.VEL_STD,
                fwd[0], fwd[1], fwd[2],
                up[0], up[1], up[2],
                _car.boost_amount,
                int(_car.on_ground),
                int(_car.has_flip),
                int(_car.is_demoed),
                int(_car.has_jump),
                magnitude/self.POS_STD,
                teammate,
                self.demo_timers[_car.car_id] / self.DEMO_TIMER_STD,
            ]
        return p

    def create_boost_packet(self, player_car: PhysicsObject, boost_index: int, inverted: bool):
        # for each boost give the direction, distance, and availability of boost
        boost_avail_list = self.inverted_boosts_availability if inverted else self.boosts_availability
        location = self.inverted_boost_locations[boost_index] if inverted else self.boost_locations[boost_index]
        dist = location - player_car.position
        mag = math.sqrt(dist[0] ** 2 + dist[1] ** 2 + dist[2] ** 2) / self.POS_STD
        val = 0 if not bool(boost_avail_list[boost_index]) else (1.0 if location[2] == 73.0 else 0.12)
        p = [
            dist[0] / self.POS_STD, dist[1] / self.POS_STD, dist[2] / self.POS_STD,
            val,
            mag
        ]
        return p

    def add_boosts_to_obs(self, obs, player_car: PhysicsObject, inverted: bool):
        for i in range(self.boost_locations.shape[0]):
            obs.extend(self.create_boost_packet(player_car, i, inverted))

    def add_players_to_obs(self, obs: List, state: GameState, player: PlayerData, ball: PhysicsObject,
                           prev_act: np.ndarray, inverted: bool):

        player_data = self.create_player_packet(player, player.inverted_car_data
                                                if inverted else player.car_data, ball, prev_act)
        a_max = 2
        o_max = 3
        a_count = 0
        o_count = 0
        allies = []
        opponents = []
        closest = 0
        if self.only_closest_opp:
            tmp_oppo = [p for p in state.players if p.team_num != player.team_num]
            tmp_oppo.sort(key=lambda p: np.linalg.norm(p.car_data.position - player.car_data.position))
            closest = tmp_oppo[0].car_id

        for p in state.players:
            if p.car_id == player.car_id:
                continue

            if p.team_num == player.team_num and a_count < a_max:
                a_count += 1
            elif o_count < o_max:
                o_count += 1
            else:
                continue

            if p.team_num == player.team_num and not self.only_closest_opp:
                allies.append(self.create_car_packet(player.inverted_car_data if inverted else player.car_data,
                              p.inverted_car_data if inverted else p.car_data, p, ball, p.team_num == player.team_num))
            elif not self.only_closest_opp or closest == p.car_id:
                opponents.append(self.create_car_packet(player.inverted_car_data if inverted else player.car_data,
                                                        p.inverted_car_data if inverted else p.car_data, p, ball,
                                                        p.team_num == player.team_num))
            else:
                continue

        for _ in range(a_max - a_count):
            allies.append(self.dummy_player)

        for _ in range(o_max - o_count):
            opponents.append(self.dummy_player)

        if not self.embed_players:
            random.shuffle(allies)
            random.shuffle(opponents)

        obs.extend(allies)
        obs.extend(opponents)

        return player_data

    def add_action_to_stack(self, new_action: np.ndarray, car_id: int):
        stack = self.action_stacks[car_id]
        stack[self.action_size:] = stack[:-self.action_size]
        stack[:self.action_size] = new_action

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:

        if player.team_num == 1:
            inverted = True
            ball = state.inverted_ball
        else:
            inverted = False
            ball = state.ball

        obs = []
        players_data = []
        player_dat = self.add_players_to_obs(players_data, state, player, ball, previous_action, inverted)
        obs.extend(player_dat)
        obs.extend(self.create_ball_packet(ball))
        if not self.embed_players:
            for p in players_data:
                obs.extend(p)
        # this adds boost timers and direction/distance to all boosts
        # unnecessary if only doing aerial stuff
        if self.extra_boost_info:
            if inverted:
                obs.extend(self.orange_obs)
            else:
                obs.extend(self.blue_obs)
            self.add_boosts_to_obs(obs, player.inverted_car_data if inverted else player.car_data, inverted)
        if self.expanding and not self.embed_players:
            return np.expand_dims(obs, 0)
        elif self.expanding and self.embed_players:
            return np.expand_dims(obs, 0), np.expand_dims(players_data, 0)
        elif not self.expanding and not self.embed_players:
            return obs
        else:
            return obs, players_data

    def get_obs_space(self) -> Space:
        players = self.num_players-1 or 5
        car_size = len(self.dummy_player)
        player_size = 251 + (self.stack_size * self.action_size)
        return Tuple((
            Box(-np.inf, np.inf, (1, player_size)),
            Box(-np.inf, np.inf, (1, players, car_size)),
        ))


class CoyoteStackedObsBuilder(CoyoteObsBuilder):
    def __init__(self, stack_size: int = 5, action_parser=None
                 ):
        return NotImplemented
        super().__init__()
        self.action_parser = action_parser
        self.stack_size = stack_size

    def reset(self, initial_state: GameState):
        super().reset(initial_state)
        self.action_stacks = {}
        for p in initial_state.players:
            self.action_stacks[p.car_id] = np.concatenate([self.default_action] * self.stack_size)
        self.action_parser.reset(initial_state)




if __name__ == "__main__":
    print("nope")
