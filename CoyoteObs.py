import math
import random

import numpy as np
from typing import Any, List
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.common_values import BOOST_LOCATIONS
from itertools import chain

# TODO profile


# inspiration from Raptor (Impossibum) and Necto (Rolv/Soren)
class CoyoteObsBuilder(ObsBuilder):
    def __init__(self, tick_skip=8, team_size=3, expanding: bool = True):
        super().__init__()
        self.expanding = expanding
        self.POS_STD = 2300
        self.VEL_STD = 2300
        self.ANG_STD = 5.5
        self.BOOST_TIMER_STD = 10
        self.DEMO_TIMER_STD = 3
        self.dummy_player = [0] * 34
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

    def reset(self, initial_state: GameState):
        self.state = None
        self.boost_timers = np.zeros(self.boost_locations.shape[0])
        self.inverted_boost_timers = np.zeros(self.boost_locations.shape[0])
        self.demo_timers = np.zeros(max(p.car_id for p in initial_state.players))
        self.blue_obs = []
        self.orange_obs = []

    def pre_step(self, state: GameState):
        self.state = state
        # create player/team agnostic items (do these even exist?)
        self._update_timers(state)
        # create team specific things
        self.blue_obs.extend(self.boost_timers / self.BOOST_TIMER_STD)
        self.orange_obs.extend(self.inverted_boost_timers / self.BOOST_TIMER_STD)

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
            if dm == 1:  # Demoed
                prev_timer = self.demo_timers[cid]
                if prev_timer > 0:
                    self.demo_timers[cid] = max(0, prev_timer - self.time_interval)
                else:
                    self.demo_timers[cid] = 3
            else:  # Not demoed
                self.demo_timers[cid] = 0

    def create_ball_packet(self, ball: PhysicsObject):
        p = [
            ball.position / self.POS_STD,
            ball.linear_velocity / self.VEL_STD,
            ball.angular_velocity / self.ANG_STD,
            [
                math.sqrt(sum([x * x for x in ball.linear_velocity]))/2300,
                int(ball.position[2] <= 100)
            ],
        ]
        return list(chain(*p))

    def create_player_packet(self, player: PlayerData, car: PhysicsObject, ball: PhysicsObject, prev_act: np.ndarray):
        p = [
            car.position / self.POS_STD,
            car.linear_velocity / self.VEL_STD,
            car.angular_velocity / self.ANG_STD,
            (ball.position - car.position) / self.POS_STD,
            (ball.linear_velocity - car.linear_velocity) / self.VEL_STD,
            car.forward(),
            car.up(),
            [
                np.linalg.norm(car.linear_velocity)/2300,
                player.boost_amount,
                int(player.on_ground),
                int(player.has_flip),
                int(player.is_demoed),
            ],
            self.demo_timers[player.car_id] / self.DEMO_TIMER_STD,
            prev_act,
        ]
        return list(chain(*p))

    def create_car_packet(self, player_car: PhysicsObject, car: PhysicsObject,
                          _car: PlayerData, ball: PhysicsObject, teammate: bool):
        diff = car.position - player_car.position
        magnitude = np.linalg.norm(diff)
        p = [
                car.position / self.POS_STD,
                car.linear_velocity / self.VEL_STD,
                car.angular_velocity / self.ANG_STD,
                diff / self.POS_STD,
                (car.linear_velocity - player_car.linear_velocity) / self.VEL_STD,
                (ball.position - car.position) / self.POS_STD,
                (ball.linear_velocity - car.linear_velocity) / self.VEL_STD,
                car.forward(),
                car.up(),
                [_car.boost_amount,
                    int(_car.on_ground),
                    int(_car.has_flip),
                    int(_car.is_demoed),
                    magnitude/self.POS_STD],
                teammate,
                self.demo_timers[_car.car_id] / self.DEMO_TIMER_STD,
            ]
        return list(chain(*p))

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

        for p in state.players:
            if p.car_id == player.car_id:
                continue

            if p.team_num == player.team_num and a_count < a_max:
                a_count += 1
            elif p.team_num == 1 and o_count < o_max:
                o_count += 1
            else:
                continue

            if p.team_num == player.team_num:
                allies.append(self.create_car_packet(player.inverted_car_data if inverted else player.car_data,
                              p.inverted_car_data if inverted else p.car_data, p, ball, p.team_num == player.team_num))
            else:
                opponents.append(self.create_car_packet(player.inverted_car_data if inverted else player.car_data,
                                                        p.inverted_car_data if inverted else p.car_data, p, ball,
                                                        p.team_num == player.team_num))

        for _ in range(a_max - a_count):
            allies.append(self.dummy_player)

        for _ in range(o_max - o_count):
            opponents.append(self.dummy_player)

        random.shuffle(allies)
        random.shuffle(opponents)

        obs.extend(allies)
        obs.extend(opponents)

        return player_data

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
        for p in players_data:
            obs.extend(p)
        if inverted:
            obs.extend(self.orange_obs)
        else:
            obs.extend(self.blue_obs)
        if self.expanding:
            return np.expand_dims(obs, 0)
        return obs


if __name__ == "__main__":
    print("nope")
