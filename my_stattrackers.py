import numpy as np

from rocket_learn.utils.gamestate_encoding import StateConstants
from rocket_learn.utils.stat_trackers.stat_tracker import StatTracker
from rlgym.utils.common_values import CEILING_Z, BACK_WALL_Y, SIDE_WALL_X


# class ExitVelocity(StatTracker):
#     def __init__(self):
#         super().__init__("avg_exit_vel_top_20_perc")
#         self.count = 0
#         self.total_speed = 0.0
#         self.exit_vels = None
#
#     def reset(self):
#         self.count = 0
#         self.total_speed = 0.0
#         self.exit_vels = []
#
#     def update(self, gamestates: np.ndarray, mask: np.ndarray):
#         ball_speeds = gamestates[:, StateConstants.BALL_LINEAR_VELOCITY]
#         touched = gamestates[:, StateConstants.BALL_TOUCHED]
#         xs = ball_speeds[:, 0]
#         ys = ball_speeds[:, 1]
#         zs = ball_speeds[:, 2]
#         speeds = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
#         self.total_speed += np.sum(speeds)
#         self.count += speeds.size
#
#     def get_stat(self):
#         return self.total_speed / (self.count or 1)


class GoalSpeedTop5perc(StatTracker):
    def __init__(self):
        super().__init__("avg_goal_speed_top_5_percent")
        self.count = 0
        self.goal_speeds = None

    def reset(self):
        self.count = 0
        self.goal_speeds = []

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        if gamestates.ndim > 1 and len(gamestates) > 1:
            end = gamestates[-2]
            goal_speed = end[StateConstants.BALL_LINEAR_VELOCITY]
            goal_speed = np.linalg.norm(goal_speed)

            self.goal_speeds.append(goal_speed / 27.78)  # convert to km/h
            self.count += 1

    def get_stat(self):
        self.goal_speeds.sort()
        top_5 = int(-1 * self.count / 20) or -1
        self.goal_speeds = self.goal_speeds[top_5:]
        total_speed = sum(self.goal_speeds)
        return total_speed / ((-1 * top_5) or 1)


class FlipReset(StatTracker):
    # slice it up into has_jump and the on_ground
    # do if diff if it goes from 0 to 1 has-flip, and that with not on-ground
    # count those 1s

    def __init__(self):
        super().__init__("Flip_reset")
        self.count = 0
        self.flip_reset_count = 0

    def reset(self):
        self.count = 0
        self.flip_reset_count = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]
        num_players = len(players[0]) // 39
        has_jumps = players[:, StateConstants.HAS_JUMP]
        # on_grounds = players[:, StateConstants.ON_GROUND]
        players_x = players[:, StateConstants.CAR_POS_X]
        players_y = players[:, StateConstants.CAR_POS_Y]
        players_z = players[:, StateConstants.CAR_POS_Z]
        for i in range(num_players):
            has_jumps_player = has_jumps[:, i]
            changes = np.where(has_jumps_player[:1] < has_jumps_player[:-1], True, False)
            player_x = players_x[:, i]
            player_y = players_y[:, i]
            player_z = players_z[:, i]
            on_grounds_player = np.where((player_z < 300) | (player_z > CEILING_Z - 300) |
                                         ((-SIDE_WALL_X + 700) > player_x) |
                                         ((SIDE_WALL_X - 700) > player_x) |
                                         ((-BACK_WALL_Y + 700) > player_y) |
                                         ((BACK_WALL_Y - 700) > player_y), True, False)
            self.flip_reset_count += (~on_grounds_player[1:] & changes).sum()

        self.count += has_jumps.size

    def get_stat(self):
        return self.flip_reset_count / (self.count or 1)


class ActionGroupingTracker:
    # check if actions were used at the appropriate time

    def __init__(self, aerial_indices, ground_indices, defend_indices, wall_indices):
        self.name = "Action Grouping Tracker"
        self.count = 0
        self.flip_reset_count = 0

    def reset(self):
        self.count = 0
        self.flip_reset_count = 0

    def update(self, gamestates: np.ndarray, mask: np.ndarray):
        players = gamestates[:, StateConstants.PLAYERS]
        num_players = len(players[0]) // 39
        has_jumps = players[:, StateConstants.HAS_JUMP]
        # on_grounds = players[:, StateConstants.ON_GROUND]
        players_x = players[:, StateConstants.CAR_POS_X]
        players_y = players[:, StateConstants.CAR_POS_Y]
        players_z = players[:, StateConstants.CAR_POS_Z]
        for i in range(num_players):
            has_jumps_player = has_jumps[:, i]
            changes = np.where(has_jumps_player[:1] < has_jumps_player[:-1], True, False)
            player_x = players_x[:, i]
            player_y = players_y[:, i]
            player_z = players_z[:, i]
            on_grounds_player = np.where((player_z < 300) | (player_z > CEILING_Z - 300) |
                                         ((-SIDE_WALL_X + 700) > player_x) |
                                         ((SIDE_WALL_X - 700) > player_x) |
                                         ((-BACK_WALL_Y + 700) > player_y) |
                                         ((BACK_WALL_Y - 700) > player_y), True, False)
            self.flip_reset_count += (~on_grounds_player[1:] & changes).sum()

        self.count += has_jumps.size

    def get_stat(self):
        return self.flip_reset_count / (self.count or 1)