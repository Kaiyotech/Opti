from rlgym.utils.common_values import BLUE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM, BALL_MAX_SPEED, \
    CAR_MAX_SPEED, BALL_RADIUS, GOAL_HEIGHT, CEILING_Z, BACK_NET_Y, BACK_WALL_Y, SIDE_WALL_X
import numpy as np
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.math import cosine_similarity
from Constants_kickoff import FRAME_SKIP

from numpy.linalg import norm

from typing import Tuple, List
from rlgym.utils.common_values import BOOST_LOCATIONS

def _closest_to_ball(state: GameState) -> Tuple[int, int]:
    # returns [blue_closest, orange_closest]
    length = len(state.players)
    dist_list: List[float] = [100_000] * length
    blue_closest = -1
    orange_closest = -1
    for i, player in enumerate(state.players):
        dist = np.linalg.norm(player.car_data.position - state.ball.position)
        dist_list[i] = dist
        if state.players[i].team_num == BLUE_TEAM and blue_closest == -1:
            blue_closest = i
        elif state.players[i].team_num == ORANGE_TEAM and orange_closest == -1:
            orange_closest = i
        elif state.players[i].team_num == BLUE_TEAM and dist <= dist_list[blue_closest]:
            if dist == dist_list[blue_closest]:
                if state.players[i].car_data.position[0] > state.players[blue_closest].car_data.position[0]:
                    blue_closest = i
                    continue
            else:
                blue_closest = i
                continue
        elif state.players[i].team_num == ORANGE_TEAM and dist <= dist_list[orange_closest]:
            if dist == dist_list[orange_closest]:
                if state.players[i].car_data.position[0] < state.players[orange_closest].car_data.position[0]:
                    orange_closest = i
                    continue
            else:
                orange_closest = i
                continue
    return blue_closest, orange_closest


class ZeroSumReward(RewardFunction):
    # framework for zerosum comes from Nexto code (Rolv and Soren)
    # (https://github.com/Rolv-Arild/Necto/blob/master/training/reward.py)
    def __init__(
            self,
            goal_w=0,  # go to 10 after working
            concede_w=0,
            velocity_pb_w=0,  # 0.01,
            velocity_bg_w=0,  # 0.05,
            touch_grass_w=0,  # -0.005,
            acel_ball_w=0,  # 1.5,
            boost_gain_w=0,  # 1.5,
            boost_spend_w=1.5,  # 1.5 is default
            punish_boost=False,  # punish once they start wasting and understand the game a bit
            jump_touch_w=0,  # 3,
            cons_air_touches_w=0,  # 6,
            wall_touch_w=0,  # 0.25,
            demo_w=0,  # 3,  # 6,
            got_demoed_w=0,  # -3,  # -6,
            kickoff_w=0,  # 0.1,
            double_tap_w=0,
            aerial_goal_w=0,
            flip_reset_w=0,
            inc_flip_reset_w=0,
            flip_reset_goal_w=0,
            flip_reset_help_w=0,
            has_flip_reset_vbg_w=0,
            quick_flip_reset_w=0,
            quick_flip_reset_norm_sec=1,
            punish_low_touch_w=0,
            punish_ceiling_pinch_w=0,
            ball_opp_half_w=0,
            kickoff_special_touch_ground_w=0,
            kickoff_final_boost_w=0,
            kickoff_vpb_after_0_w=0,
            dribble_w=0,
            forward_ctrl_w=0,
            exit_velocity_w=0,
            exit_vel_angle_w=0,
            req_reset_exit_vel=False,
            punish_car_ceiling_w=0,
            punish_action_change_w=0,
            decay_punish_action_change_w=0,
            goal_speed_exp=1,  # fix this eventually
            min_goal_speed_rewarded_kph=0,
            touch_height_exp=1,
            tick_skip=FRAME_SKIP,
            team_spirit=0,  # increase as they learn
            zero_sum=True,
            prevent_chain_reset=False,
            touch_ball_w=0,
            boost_remain_touch_w=0,
            supersonic_bonus_vpb_w=0,
            turtle_w=0,
            zero_touch_grass_if_ss=False,
            final_reward_ball_dist_w=0,
            final_reward_boost_w=0,
            punish_directional_changes=False,
            punish_bad_spacing_w=0,
    ):
        self.punish_bad_spacing_w = punish_bad_spacing_w
        self.forward_ctrl_w = forward_ctrl_w
        self.final_reward_boost_w = final_reward_boost_w
        self.punish_directional_changes = punish_directional_changes
        self.decay_punish_action_change_w = decay_punish_action_change_w
        self.final_reward_ball_dist_w = final_reward_ball_dist_w
        self.zero_touch_grass_if_ss = zero_touch_grass_if_ss
        self.turtle_w = turtle_w
        self.supersonic_bonus_vpb_w = supersonic_bonus_vpb_w
        self.boost_remain_touch_w = boost_remain_touch_w
        self.touch_ball_w = touch_ball_w
        self.end_object_tracker = 0
        self.min_goal_speed_rewarded = min_goal_speed_rewarded_kph * 27.78  # to uu
        self.exit_vel_angle_w = exit_vel_angle_w
        self.quick_flip_reset_w = quick_flip_reset_w
        self.req_reset_exit_vel = req_reset_exit_vel
        self.cons_resets = 0
        self.goal_w = goal_w
        self.concede_w = concede_w
        if zero_sum:
            self.concede_w = 0
        self.velocity_pb_w = velocity_pb_w
        self.velocity_bg_w = velocity_bg_w * (tick_skip / 8)
        self.touch_grass_w = touch_grass_w
        self.acel_ball_w = acel_ball_w
        self.boost_gain_w = boost_gain_w
        if punish_boost:
            self.boost_spend_w = boost_spend_w * self.boost_gain_w * ((33.3334 / (120 / tick_skip)) * 0.01)
        else:
            self.boost_spend_w = 0
        self.jump_touch_w = jump_touch_w
        self.cons_air_touches_w = cons_air_touches_w
        self.wall_touch_w = wall_touch_w
        self.demo_w = demo_w
        self.got_demoed_w = got_demoed_w
        if zero_sum:
            self.got_demoed_w = 0
        self.kickoff_w = kickoff_w * (tick_skip / 8)
        self.double_tap_w = double_tap_w
        self.aerial_goal_w = aerial_goal_w
        self.flip_reset_w = flip_reset_w
        self.inc_flip_reset_w = inc_flip_reset_w
        self.flip_reset_goal_w = flip_reset_goal_w
        self.flip_reset_help_w = flip_reset_help_w
        self.has_flip_reset_vbg_w = has_flip_reset_vbg_w
        self.prevent_chain_reset = prevent_chain_reset
        self.punish_low_touch_w = punish_low_touch_w
        self.punish_ceiling_pinch_w = punish_ceiling_pinch_w
        self.ball_opp_half_w = ball_opp_half_w
        self.kickoff_special_touch_ground_w = kickoff_special_touch_ground_w
        self.kickoff_final_boost_w = kickoff_final_boost_w
        self.kickoff_vpb_after_0_w = kickoff_vpb_after_0_w
        self.dribble_w = dribble_w
        self.exit_velocity_w = exit_velocity_w
        self.punish_car_ceiling_w = punish_car_ceiling_w
        self.punish_action_change_w = punish_action_change_w
        self.previous_action = None
        self.goal_speed_exp = goal_speed_exp
        self.touch_height_exp = touch_height_exp
        self.rewards = None
        self.current_state = None
        self.last_state = None
        self.touch_timeout = 8 * 120 // tick_skip  # 120 ticks at 8 tick skip is 8 seconds
        self.kickoff_timeout = 5 * 120 // tick_skip
        self.quick_flip_reset_norm_steps = quick_flip_reset_norm_sec * 120 // tick_skip
        self.kickoff_timer = 0
        self.closest_reset_blue = -1
        self.closest_reset_orange = -1
        self.blue_touch_timer = self.touch_timeout + 1
        self.orange_touch_timer = self.touch_timeout + 1
        self.blue_toucher = None
        self.orange_toucher = None
        self.team_spirit = team_spirit
        self.n = 0
        self.cons_touches = 0
        self.zero_sum = zero_sum
        self.num_touches = []
        # for double tap
        self.backboard_bounce = False
        self.floor_bounce = False
        self.got_reset = []
        # for aerial goal
        self.blue_touch_height = -1
        self.orange_touch_height = -1
        self.last_touched_frame = [-1] * 6
        self.reset_timer = -100000
        self.flip_reset_delay_steps = 0.25 * (120 // tick_skip)
        self.last_touch_time = -100_000
        self.exit_vel_arm_time_steps = 0.15 * (120 // tick_skip)
        self.exit_rewarded = [False] * 6
        self.last_touch_car = None
        self.launch_angle_car = [None] * 6
        self.exit_vel_save = [None] * 6
        self.big_boosts = [BOOST_LOCATIONS[i] for i in [3, 4, 15, 18, 29, 30]]
        self.big_boost = np.asarray(self.big_boosts)
        self.big_boost[:, -1] = 18  # fix the boost height
        self.last_action_change = None

    def pre_step(self, state: GameState):
        if state != self.current_state:
            self.last_state = self.current_state
            self.current_state = state
            self.n = 0
            self.blue_touch_timer += 1
            self.orange_touch_timer += 1
            self.kickoff_timer += 1
            # for double tap
            if state.ball.position[2] < BALL_RADIUS * 2 and state.last_touch == -1:
                self.floor_bounce = True
            elif 0.55 * self.last_state.ball.linear_velocity[1] < state.ball.linear_velocity[1] > 0.65 * \
                    self.last_state.ball.linear_velocity[1] and \
                    abs(state.ball.position[1]) > 4900 and state.ball.position[2] > 500:
                self.backboard_bounce = True
            # for aerial
            # player who last touched is now on the ground, don't allow jumping back up to continue "dribble"
            if self.blue_touch_timer < self.orange_touch_timer and state.players[self.blue_toucher].on_ground:
                self.cons_touches = 0
            elif self.orange_touch_timer < self.blue_touch_timer and state.players[self.orange_toucher].on_ground:
                self.cons_touches = 0
        # Calculate rewards
        player_rewards = np.zeros(len(state.players))
        player_self_rewards = np.zeros(len(state.players))
        normed_last_ball_vel = norm(self.last_state.ball.linear_velocity)
        norm_ball_vel = norm(state.ball.linear_velocity)
        xy_norm_ball_vel = norm(state.ball.linear_velocity[:-1])
        for i, player in enumerate(state.players):
            last = self.last_state.players[i]

            if player.ball_touched:
                player_rewards[i] += self.touch_ball_w
                player_rewards[i] += player.boost_amount * self.boost_remain_touch_w
                self.last_touch_time = self.kickoff_timer
                self.exit_rewarded[i] = False
                self.last_touch_car = i
                if player.team_num == BLUE_TEAM:
                    # new blue toucher for aerial touches (or kickoff touch)
                    if self.blue_toucher != i or self.orange_touch_timer <= self.blue_touch_timer:
                        self.cons_touches = 0
                        player_rewards += player.boost_amount * self.kickoff_final_boost_w
                    self.blue_toucher = i
                    self.blue_touch_timer = 0
                    self.blue_touch_height = player.car_data.position[2]
                else:
                    if self.orange_toucher != i or self.blue_touch_timer <= self.orange_touch_timer:
                        self.cons_touches = 0
                        player_rewards += player.boost_amount * self.kickoff_final_boost_w
                    self.orange_toucher = i
                    self.orange_touch_timer = 0
                    self.orange_touch_height = player.car_data.position[2]

                # acel_ball
                vel_difference = abs(np.linalg.norm(self.last_state.ball.linear_velocity -
                                                    self.current_state.ball.linear_velocity))
                player_rewards[i] += self.acel_ball_w * vel_difference / 4600.0

                # jumptouch
                min_height = 120
                max_height = CEILING_Z - BALL_RADIUS
                rnge = max_height - min_height
                if not player.on_ground and state.ball.position[2] > min_height:
                    # x_from_wall = min(SIDE_WALL_X - BALL_RADIUS - abs(state.ball.position[0]), 20)
                    # wall_multiplier = x_from_wall / 20
                    player_rewards[i] += self.jump_touch_w * (
                            (state.ball.position[2] ** self.touch_height_exp) - min_height) / rnge

                # wall touch
                min_height = 500
                if player.on_ground and state.ball.position[2] > min_height:
                    player_self_rewards[i] += self.wall_touch_w * (state.ball.position[2] - min_height) / rnge

                # ground/kuxir/team pinch training
                if state.ball.position[2] < 250:
                    player_self_rewards[i] += self.punish_low_touch_w

                # anti-ceiling pinch
                if state.ball.position[2] > CEILING_Z - 2 * BALL_RADIUS:
                    player_self_rewards += self.punish_ceiling_pinch_w

                # cons air touches, max reward of 5, normalized to 1, initial reward 1.4, only starts at second touch
                if state.ball.position[2] > 140 and not player.on_ground:
                    self.cons_touches += 1
                    if self.cons_touches > 1:
                        player_rewards[i] += self.cons_air_touches_w * min((1.4 ** self.cons_touches), 5) / 5
                else:
                    self.cons_touches = 0

                # dribble
                if state.ball.position[2] > 120 and player.on_ground:
                    player_rewards[i] += self.dribble_w

            # not touched
            else:
                if last.ball_touched:
                    self.launch_angle_car[i] = last.car_data.forward()[:-1] / np.linalg.norm(last.car_data.forward()[:-1] + 1e-8)
                    self.exit_vel_save[i] = state.ball.linear_velocity[:-1] / np.linalg.norm(state.ball.linear_velocity[:-1] + 1e-8)
                if self.kickoff_timer - self.last_touch_time > self.exit_vel_arm_time_steps and not self.exit_rewarded[i] and self.last_touch_car == i:
                    self.exit_rewarded[i] = True
                    # rewards 1 for a 120 kph flick (3332 uu/s), 11 for a 6000 uu/s (max speed)
                    req_reset = 1
                    if self.req_reset_exit_vel:
                        if self.got_reset[i]:
                            req_reset = 1
                        else:
                            req_reset = 0
                    ang_mult = 1
                    if self.exit_vel_angle_w != 0:
                        # 0.785 is 45

                        dot_product = np.dot(self.launch_angle_car[i], self.exit_vel_save[i])
                        angle = min(np.arccos(dot_product), 0.785) / 0.785
                        if np.isnan(angle):  # rare enough to just avoid in the data
                            angle = 0
                        ang_mult = self.exit_velocity_w * max(angle, 0.1)  #  0.1 is a small mult to still reward 0 angle
                    vel_mult = self.exit_velocity_w * 0.5 * ((xy_norm_ball_vel ** 5) / (3332 ** 5) + ((xy_norm_ball_vel ** 2) / (3332 ** 2)))
                    player_rewards[i] += vel_mult * req_reset * ang_mult

            # ball got too low, don't credit bounces
            if self.cons_touches > 0 and state.ball.position[2] <= 140:
                self.cons_touches = 0

            # vel bg
            if self.blue_toucher is not None or self.orange_toucher is not None:
                if player.team_num == BLUE_TEAM:
                    objective = np.array(ORANGE_GOAL_BACK)
                else:
                    objective = np.array(BLUE_GOAL_BACK)
                vel = state.ball.linear_velocity
                pos_diff = objective - state.ball.position
                norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
                norm_vel = vel / BALL_MAX_SPEED
                vel_bg_reward = float(np.dot(norm_pos_diff, norm_vel))
                player_rewards[i] += self.velocity_bg_w * vel_bg_reward
                if self.got_reset[i] and player.has_jump and not player.on_ground:
                    player_rewards[i] += self.has_flip_reset_vbg_w * vel_bg_reward

            # distance ball from halfway (for kickoffs)
            # 1 at max oppo wall, 0 at midfield, -1 at our wall
            if player.team_num == BLUE_TEAM:
                objective = BACK_WALL_Y - BALL_RADIUS
            else:
                objective = -BACK_WALL_Y + BALL_RADIUS
            player_rewards[i] += self.ball_opp_half_w * (1 + (state.ball.position[1] - objective) / objective)

            # boost
            # don't punish or reward boost when above  approx single jump height
            if player.car_data.position[2] < 2 * BALL_RADIUS:
                boost_diff = player.boost_amount - last.boost_amount
                if boost_diff > 0:
                    player_rewards[i] += self.boost_gain_w * boost_diff
                else:
                    player_rewards[i] += self.boost_spend_w * boost_diff

            # touch_grass
            if player.on_ground and player.car_data.position[2] < 25:
                if not self.zero_touch_grass_if_ss:
                    player_self_rewards[i] += self.touch_grass_w
                elif not np.linalg.norm(np.dot(player.car_data.linear_velocity, player.car_data.forward())) >= 2200:
                    player_self_rewards[i] += self.touch_grass_w
                if self.closest_reset_blue == i or self.closest_reset_orange == i:
                    player_self_rewards[i] += self.kickoff_special_touch_ground_w

            # turtle
            if abs(player.car_data.roll()) > 3 and player.car_data.position[2] < 55 and \
                np.linalg.norm(player.car_data.angular_velocity) < 5 and abs(last.car_data.roll()) > 3 and \
                    last.car_data.position[2] < 55:
                player_self_rewards[i] += self.turtle_w

            # spacing punishment
            mid = len(player_rewards) // 2
            if mid > 1:  # skip in 1v1
                if i < mid:
                    start = 0
                    stop = mid
                else:
                    start = mid
                    stop = mid * 2
                for _i in range(start, stop):
                    if _i == i:
                        continue
                    dist = np.linalg.norm(player.car_data.position - state.players[_i].car_data.position)
                    player_self_rewards[i] += self.punish_bad_spacing_w * max(((700 - dist) / 700), 0.0)

            # touch ceiling
            if player.on_ground and player.car_data.position[2] > CEILING_Z - 20:
                player_self_rewards[i] += self.punish_car_ceiling_w

            # demo
            if player.is_demoed and not last.is_demoed:
                player_rewards[i] += self.got_demoed_w
            if player.match_demolishes > last.match_demolishes:
                player_rewards[i] += self.demo_w

            # vel pb
            vel = player.car_data.linear_velocity
            pos_diff = state.ball.position - player.car_data.position
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / CAR_MAX_SPEED
            supersonic = False
            if self.supersonic_bonus_vpb_w != 0 and np.linalg.norm(vel) >= 2190:
                supersonic = True
            speed_rew = float(np.dot(norm_pos_diff, norm_vel))
            player_rewards[i] += self.velocity_pb_w * speed_rew
            player_rewards[i] += supersonic * self.supersonic_bonus_vpb_w * speed_rew
            if state.ball.position[0] != 0 and state.ball.position[1] != 0:
                player_rewards[i] += self.kickoff_vpb_after_0_w * speed_rew

            # flip reset helper
            if self.flip_reset_help_w != 0:
                upness = cosine_similarity(
                    np.asarray([0, 0, CEILING_Z - player.car_data.position[2]]),
                    -player.car_data.up())  # bottom of car points to ceiling
                from_wall_ratio = min(1, abs(state.ball.position[0]) / 1300)
                height_ratio = min(1, state.ball.position[2] / 1700)
                bottom_ball_ratio = 2 * cosine_similarity(
                    state.ball.position - player.car_data.position, -player.car_data.up())
                if player.team_num == BLUE_TEAM:
                    objective = np.array(ORANGE_GOAL_BACK)
                else:
                    objective = np.array(BLUE_GOAL_BACK)
                align_ratio = cosine_similarity(objective - player.car_data.position, player.car_data.forward())
                pos_diff = state.ball.position - player.car_data.position
                pos_diff[2] *= 2  # make the z axis twice as important
                norm_pos_diff = np.linalg.norm(pos_diff)
                flip_rew = bottom_ball_ratio * from_wall_ratio * height_ratio * align_ratio * \
                           np.clip(-1, 1, 40 * upness / (norm_pos_diff + 1))
                player_self_rewards[i] += self.flip_reset_help_w * flip_rew

            # kickoff reward
            if state.ball.position[0] == 0 and state.ball.position[1] == 0 and \
                    (self.closest_reset_blue == i or self.closest_reset_orange == i) and \
                    self.kickoff_timer < self.kickoff_timeout:
                player_self_rewards[i] += self.kickoff_w * -1

            # flip reset
            if not last.has_jump and player.has_jump and player.car_data.position[2] > 200 and not player.on_ground:
                if not self.got_reset[i]:  # first reset of episode
                    #  1 reward for
                    player_rewards[i] += self.quick_flip_reset_w * self.quick_flip_reset_norm_steps / self.kickoff_timer
                self.got_reset[i] = True
                # player_rewards[i] += self.flip_reset_w * np.clip(cosine_similarity(
                #     np.asarray([0, 0, CEILING_Z - player.car_data.position[2]]), -player.car_data.up()), 0.1, 1)
                if self.kickoff_timer - self.reset_timer > self.flip_reset_delay_steps:
                    player_rewards[i] += self.flip_reset_w
                    self.cons_resets += 1
                    if self.cons_resets > 1:
                        player_rewards[i] += self.inc_flip_reset_w * min((1.4 ** self.cons_resets), 6) / 6
                if self.prevent_chain_reset:
                    self.reset_timer = self.kickoff_timer
            if player.on_ground:
                #  self.got_reset[i] = False
                self.cons_resets = 0
                self.reset_timer = -100000

        mid = len(player_rewards) // 2

        # Handle goals with no scorer for critic consistency,
        # random state could send ball straight into goal
        if self.blue_touch_timer < self.touch_timeout or self.orange_touch_timer < self.touch_timeout:
            d_blue = state.blue_score - self.last_state.blue_score
            d_orange = state.orange_score - self.last_state.orange_score
            if d_blue > 0:
                goal_speed = normed_last_ball_vel ** self.goal_speed_exp
                if normed_last_ball_vel < self.min_goal_speed_rewarded:
                    goal_speed = 0
                goal_reward = self.goal_w * (goal_speed / (CAR_MAX_SPEED * 1.25))
                if self.blue_touch_timer < self.touch_timeout:
                    player_rewards[self.blue_toucher] += (1 - self.team_spirit) * goal_reward
                    if self.got_reset[self.blue_toucher]:
                        player_rewards[self.blue_toucher] += self.flip_reset_goal_w * (goal_speed / (CAR_MAX_SPEED * 1.25))
                    if self.backboard_bounce and not self.floor_bounce:
                        player_rewards[self.blue_toucher] += self.double_tap_w
                    if self.blue_touch_height > GOAL_HEIGHT:
                        player_rewards[self.blue_toucher] += self.aerial_goal_w * (goal_speed / (CAR_MAX_SPEED * 1.25))
                    player_rewards[:mid] += self.team_spirit * goal_reward
                elif self.orange_touch_timer < self.touch_timeout and self.zero_sum:
                    player_rewards[mid:] -= goal_reward

                if self.orange_touch_timer < self.touch_timeout or self.blue_touch_timer < self.touch_timeout:
                    player_rewards[mid:] += self.concede_w

            if d_orange > 0:
                goal_speed = normed_last_ball_vel ** self.goal_speed_exp
                if normed_last_ball_vel < self.min_goal_speed_rewarded:
                    goal_speed = 0
                goal_reward = self.goal_w * (goal_speed / (CAR_MAX_SPEED * 1.25))
                if self.orange_touch_timer < self.touch_timeout:
                    player_rewards[self.orange_toucher] += (1 - self.team_spirit) * goal_reward
                    if self.got_reset[self.orange_toucher]:
                        player_rewards[self.orange_toucher] += self.flip_reset_goal_w * (goal_speed / (CAR_MAX_SPEED * 1.25))
                    if self.backboard_bounce and not self.floor_bounce:
                        player_rewards[self.orange_toucher] += self.double_tap_w
                    if self.orange_touch_height > GOAL_HEIGHT:
                        player_rewards[self.orange_toucher] += self.aerial_goal_w * (goal_speed / (CAR_MAX_SPEED * 1.25))
                    player_rewards[mid:] += self.team_spirit * goal_reward

                elif self.blue_touch_timer < self.touch_timeout and self.zero_sum:
                    player_rewards[:mid] -= goal_reward

                if self.orange_touch_timer < self.touch_timeout or self.blue_touch_timer < self.touch_timeout:
                    player_rewards[:mid] += self.concede_w

        # zero mean
        if self.zero_sum:
            orange_mean = np.mean(player_rewards[mid:])
            blue_mean = np.mean(player_rewards[:mid])
            player_rewards[:mid] -= orange_mean
            player_rewards[mid:] -= blue_mean

        self.last_state = state
        self.rewards = player_rewards + player_self_rewards
        self.last_touched_frame = [x + 1 for x in self.last_touched_frame]

    def reset(self, initial_state: GameState):
        self.n = 0
        self.last_state = None
        self.rewards = None
        self.current_state = initial_state
        self.blue_toucher = None
        self.orange_toucher = None
        self.blue_touch_timer = self.touch_timeout + 1
        self.orange_touch_timer = self.touch_timeout + 1
        self.cons_touches = 0
        self.cons_resets = 0
        self.kickoff_timer = 0
        self.closest_reset_blue, self.closest_reset_orange = _closest_to_ball(initial_state)
        self.backboard_bounce = False
        self.floor_bounce = False
        self.got_reset = [False] * len(initial_state.players)
        self.num_touches = [0] * len(initial_state.players)
        self.blue_touch_height = -1
        self.orange_touch_height = -1
        self.reset_timer = -100000
        self.last_touch_time = -100_000
        self.exit_rewarded = [False] * 6
        self.last_touch_car = None
        self.launch_angle_car = [None] * 6
        self.exit_vel_save = [None] * 6
        self.previous_action = np.asarray([-1] * len(initial_state.players))
        self.last_action_change = np.asarray([0] * len(initial_state.players))
        self.end_object_tracker += 1
        if self.end_object_tracker == 7:
            self.end_object_tracker = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, previous_model_action: np.ndarray) -> float:
        rew = self.rewards[self.n]

        if self.previous_action[self.n] != -1 and self.previous_action[self.n] != previous_model_action[0] and not \
                (10 <= previous_model_action[0] < 18 and 10 <= self.previous_action[self.n] < 18 and not
                 self.punish_directional_changes):
            rew += self.punish_action_change_w
            step_since_change = self.kickoff_timer - self.last_action_change[self.n]
            rew += self.decay_punish_action_change_w * max(0, (1 - ((step_since_change - 1) ** 1.75) / (14 ** 1.75)))
            self.last_action_change[self.n] = self.kickoff_timer

        self.previous_action[self.n] = previous_model_action[0]
        # forward on stick
        if previous_action[2] == 1 and previous_action[3] == 0 and not player.on_ground:
            rew += self.forward_ctrl_w
        self.n += 1
        return float(rew)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, previous_model_action: np.ndarray) -> float:
        reg_reward = self.get_reward(player, state, previous_action, previous_model_action)
        # if dist is 0, reward is 1
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        dist_rew = float(np.exp(-1 * dist / CAR_MAX_SPEED)) * self.final_reward_ball_dist_w
        boost_rew = float(player.boost_amount) * self.final_reward_boost_w
        return reg_reward + dist_rew + boost_rew
