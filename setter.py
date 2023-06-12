from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlgym.utils.state_setters import StateWrapper

from rocket_learn.utils.dynamic_gamemode_setter import DynamicGMSetter

from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter

from mybots_statesets import GroundAirDribble, WallDribble, HalfFlip, Walldash, Wavedash, \
    Curvedash, Chaindash, RandomEvenRecovery, RecoverySetter, LixSetter


class CoyoteSetter(DynamicGMSetter):
    def __init__(self, mode, end_object_choice=None, simulator=False, dtap_dict=None):
        if simulator:
            from rlgym_sim.utils.state_setters import DefaultState
            from rlgym_sim.utils.state_setters.random_state import RandomState
        else:
            from rlgym.utils.state_setters import DefaultState
            from rlgym.utils.state_setters.random_state import RandomState
        self.setters = []  # [1v1, 2v2, 3v3]
        replays = ["replays/ssl_1v1.npy", "replays/ssl_2v2.npy", "replays/ssl_3v3.npy"]
        aerial_replays = ["replays/aerial_1v1.npy", "replays/aerial_2v2.npy", "replays/aerial_3v3.npy"]
        flip_reset_replays = ["replays/flip_reset_1v1.npy", "replays/flip_reset_2v2.npy",
                              "replays/flip_reset_3v3.npy"]
        wall_flip_reset_replays = ["replays/wall_flip_reset_1v1.npy", "replays/wall_flip_reset_2v2.npy",
                                   "replays/wall_flip_reset_3v3.npy"]
        ground_flip_reset_replays = ["replays/ground_flip_reset_1v1.npy", "replays/ground_flip_reset_2v2.npy",
                                     "replays/ground_flip_reset_3v3.npy"]
        low_flip_reset_replays = ["replays/low_flip_reset_1v1.npy", "replays/low_flip_reset_2v2.npy",
                                  "replays/low_flip_reset_3v3.npy"]
        kickoff_replays = ["replays/kickoff_1v1.npy", "replays/kickoff_2v2.npy", "replays/kickoff_3v3.npy"]
        ceiling_replays = ["replays/ceiling_1v1.npy", "replays/ceiling_2v2.npy", "replays/ceiling_3v3.npy"]
        air_dribble_replays = ["replays/air_dribble_1v1.npy", "replays/air_dribble_2v2.npy",
                               "replays/air_dribble_3v3.npy"]
        team_pinch_replays = ["replays/pinch_1v1.npy", "replays/team_pinch_2v2.npy", "replays/team_pinch_3v3.npy"]
        full_pinch_replays = ["replays/pinch_1v1.npy", "replays/pinch_2v2.npy", "replays/pinch_3v3.npy"]
        pinch_replays = ["replays/full_pinch_1v1.npy", "replays/full_pinch_2v2.npy", "replays/full_pinch_3v3.npy"]
        double_tap_replays = ["replays/double_tap_1v1.npy", "replays/double_tap_2v2.npy", "replays/double_tap_3v3.npy",
                              "replays/double_tap_1v0.npy"]
        easy_double_tap_replays = ["replays/easy_double_tap_1v1.npy", "replays/easy_double_tap_1v1.npy",
                                   "replays/easy_double_tap_1v1.npy", "replays/easy_double_tap_1v0.npy"]
        ground_dribble_replays = ["replays/ground_dribble_1v1.npy", "replays/ground_dribble_2v2.npy",
                                  "replays/ground_dribble_3v3.npy"]
        demo_replays = ["replays/demo_1v1.npy", "replays/demo_2v2.npy", "replays/demo_3v3.npy"]
        low_recovery_replays = ["replays/low_recovery_1v1.npy", "replays/low_recovery_1v1.npy",
                                "replays/low_recovery_1v1.npy"]
        high_recovery_replays = ["replays/high_recovery_1v1.npy", "replays/high_recovery_1v1.npy",
                                 "replays/high_recovery_1v1.npy"]
        self.end_object_choice = end_object_choice
        self.end_object_tracker = [0]
        if end_object_choice is not None and end_object_choice == "random":
            self.end_object_tracker = [0]
        elif end_object_choice is not None:
            self.end_object_tracker = [int(self.end_object_choice)]

        if mode is None or mode == "normal":
            for i in range(3):
                self.setters.append(
                    WeightedSampleSetter(
                        (
                            DefaultState(),
                            AugmentSetter(ReplaySetter(replays[i], random_boost=True), True, False, False),
                            AugmentSetter(ReplaySetter(aerial_replays[i], random_boost=True), True, False, False),
                            AugmentSetter(GroundAirDribble(), True, False, False),
                            AugmentSetter(WallDribble(), True, False, False),
                            AugmentSetter(RandomState(cars_on_ground=True), True, False, False),
                            AugmentSetter(RandomState(cars_on_ground=False), True, False, False),
                            # Wavedash(zero_boost_weight=1, zero_ball_vel_weight=0.2) if i == 1 else
                            AugmentSetter(ReplaySetter(replays[i], random_boost=True), True, False, False),
                            # Curvedash(zero_boost_weight=1, zero_ball_vel_weight=0.2) if i == 1 else
                            AugmentSetter(ReplaySetter(replays[i], random_boost=True), True, False, False),
                        ),
                        # (0.05, 0.50, 0.20, 0.20, 0.025, 0.025)
                        (0.05, 0.2, 0.1, 0.1, 0.55, 0, 0, 0, 0)
                    )
                )
        elif mode == "selector":
            for i in range(3):
                self.setters.append(
                    WeightedSampleSetter(
                        (
                            DefaultState(),
                            AugmentSetter(ReplaySetter(kickoff_replays[i]), True, False, False),
                            AugmentSetter(ReplaySetter(replays[i], random_boost=True), True, False, False),
                            AugmentSetter(ReplaySetter(aerial_replays[i], random_boost=True), True, False, False),
                            AugmentSetter(GroundAirDribble(), True, False, False),
                            AugmentSetter(RandomState(cars_on_ground=True), True, False, False),
                            AugmentSetter(RandomState(cars_on_ground=False), True, False, False),
                            AugmentSetter(
                                ReplaySetter(flip_reset_replays[i], defender_front_goal_weight=0, random_boost=True),
                                True, False, False),
                            AugmentSetter(WallDribble(), True, False, False),
                            AugmentSetter(ReplaySetter(wall_flip_reset_replays[i], defender_front_goal_weight=0,
                                                       random_boost=True), True, False, False),
                            AugmentSetter(WallDribble(speed_min=1700, speed_max=1900, max_rand_z=300), True, False,
                                          False),
                            AugmentSetter(ReplaySetter(low_flip_reset_replays[i], defender_front_goal_weight=0,
                                                       random_boost=True), True, False, False),
                            AugmentSetter(ReplaySetter(ceiling_replays[i], random_boost=True), True, False, False),
                            AugmentSetter(ReplaySetter(air_dribble_replays[i], random_boost=True), True, False, False),
                            AugmentSetter(ReplaySetter(pinch_replays[i], random_boost=True), True, False, False),
                            AugmentSetter(ReplaySetter(team_pinch_replays[i], random_boost=True), True, False, False),
                            AugmentSetter(ReplaySetter(full_pinch_replays[i], random_boost=True), True, False, False),
                            AugmentSetter(ReplaySetter(ground_dribble_replays[i],
                                                       random_boost=True,
                                                       remove_defender_weight=0,
                                                       ), True, False, False
                                          ),
                            AugmentSetter(ReplaySetter(double_tap_replays[0], defender_front_goal_weight=0,
                                                       random_boost=True, dtap_dict=dtap_dict,
                                                       initial_state_dict=(0, 0, 0), expand_shrink_cars=True),
                                          True, False, False),
                            AugmentSetter(ReplaySetter(easy_double_tap_replays[0], defender_front_goal_weight=0,
                                                       random_boost=True, dtap_dict=dtap_dict,
                                                       initial_state_dict=(1, 0, 0), expand_shrink_cars=True),
                                          True, False, False),
                            # AugmentSetter(ReplaySetter(low_recovery_replays[i], random_boost=False,
                            #                            zero_ball_weight=0.8,
                            #                            zero_car_weight=0.2,
                            #                            rotate_car_weight=0.2,
                            #                            backward_car_weight=0.15,
                            #                            vel_div_weight=0.2,
                            #                            ), False, True, False),
                            # AugmentSetter(ReplaySetter(high_recovery_replays[i], random_boost=False,
                            #                            zero_ball_weight=0.8,
                            #                            zero_car_weight=0.2,
                            #                            rotate_car_weight=0.2,
                            #                            backward_car_weight=0.15,
                            #                            vel_div_weight=0.2,
                            #                            ), False, True, False),
                            # HalfFlip(),
                        ),
                        (0.08, 0.02, 0.06, 0.04,  # default, ko_repl, repl, aerial
                         0.04, 0, 0, 0.12,  # ground-air, rand grouwnd, rand air, flip reset
                         0.12, 0.12, 0.09, 0,  # wall dribble, wall flip reset, fast-low wall, low flip reset
                         0.06, 0.09, 0.04, 0.02,  # ceiling, air dribble, pinch, team pinch
                         0.02, 0, 0.04, 0.04)  # full pinch, ground dribble, double-tap, easy_doubletap
                        # # test
                        # (0, 0, 0, 0,  # default, ko_repl, repl, aerial
                        #  0.25, 0, 0, 0,  # ground-air, rand ground, rand air, flip reset
                        #  0.125, 0, 0.125, 0,  # wall dribble, wall flip reset, fast-low wall, low flip reset
                        #  0, 0, 0, 0,  # ceiling, air dribble, pinch, team pinch
                        #  0, 0, 0.25, 0.25)  # full pinch, ground dribble, double-tap, easy_doubletap
                    )
                )
        elif mode == "kickoff":
            for i in range(3):
                self.setters.append(
                    WeightedSampleSetter(
                        (
                            AugmentSetter(ReplaySetter(kickoff_replays[i]), True, False, False),
                            DefaultState(),
                        ),
                        (0.3, 0.7)
                    )
                )
        elif mode == "aerial":
            for i in range(3):
                self.setters.append(
                    WeightedSampleSetter(
                        (
                            AugmentSetter(
                                ReplaySetter(aerial_replays[i], random_boost=True, remove_defender_weight=0.1), True,
                                False, False),
                            AugmentSetter(
                                ReplaySetter(flip_reset_replays[i], random_boost=True, remove_defender_weight=0.1),
                                True, False, False),
                            AugmentSetter(
                                ReplaySetter(ceiling_replays[i], random_boost=True, remove_defender_weight=0.1), True,
                                False, False),
                            AugmentSetter(
                                ReplaySetter(air_dribble_replays[i], random_boost=True, remove_defender_weight=0.1),
                                True, False, False),
                            AugmentSetter(WallDribble(), True, False, False),
                            AugmentSetter(RandomState(cars_on_ground=False), True, False, False),
                            AugmentSetter(
                                ReplaySetter(double_tap_replays[i], random_boost=True, remove_defender_weight=0.1),
                                True, False, False),
                            AugmentSetter(GroundAirDribble(), True, False, False),
                        ),
                        (0.05, 0.20, 0.14, 0.15, 0.15, 0, 0.21, 0.1)
                    )
                )
        elif mode == "pinch":
            for i in range(3):
                self.setters.append(
                    WeightedSampleSetter(
                        (
                            AugmentSetter(ReplaySetter(pinch_replays[i], random_boost=True), True, False, False),
                            AugmentSetter(ReplaySetter(full_pinch_replays[i], random_boost=True), True, False, False),
                            AugmentSetter(ReplaySetter(team_pinch_replays[i], random_boost=True), True, False, False),
                            AugmentSetter(WallDribble(), True, False, False),
                        ),
                        (0.5, 0.15, 0.35, 0)
                    )
                )
        # elif mode == "flick":
        #     for i in range(3):
        #         self.setters.append(
        #             AugmentSetter(FlickSetter(), True, False, False)
        #         )
        elif mode == "flick":
            for i in range(3):
                self.setters.append(
                    AugmentSetter(ReplaySetter(ground_dribble_replays[i],
                                               random_boost=True,
                                               remove_defender_weight=0.25,
                                               ), True, False, False
                                  )
                )
        elif mode == "flip_reset":
            for i in range(3):
                self.setters.append(
                    WeightedSampleSetter(
                        (
                            AugmentSetter(
                                ReplaySetter(flip_reset_replays[i], defender_front_goal_weight=0.25, random_boost=True),
                                True, False, False),
                            AugmentSetter(WallDribble(), True, False, False),
                            AugmentSetter(WallDribble(speed_min=1450, speed_max=1700, max_rand_z=300), True, False,
                                          False),
                            AugmentSetter(ReplaySetter(wall_flip_reset_replays[i], defender_front_goal_weight=0.25,
                                                       random_boost=True), True, False, False),
                            AugmentSetter(WallDribble(speed_min=1700, speed_max=1900, max_rand_z=300), True, False,
                                          False),
                            AugmentSetter(ReplaySetter(ground_flip_reset_replays[i], defender_front_goal_weight=0.25,
                                                       random_boost=True), True, False, False),
                            AugmentSetter(ReplaySetter(low_flip_reset_replays[i], defender_front_goal_weight=0.25,
                                                       random_boost=True), True, False, False),
                        ),
                        # (0, 0, 0, 0.5, 0, 0.5)
                        # (0, 0, 0, 1, 0, 0, 0)
                        (0.5, 0.05, 0.05, 0.35, 0.05, 0, 0)
                    )
                )

        elif mode == "demo":
            for i in range(3):
                self.setters.append(
                    AugmentSetter(ReplaySetter(demo_replays[i]), True, False, False)
                )

        elif mode == "recovery":
            for i in range(3):
                self.setters.append(
                    WeightedSampleSetter(
                        (HalfFlip(zero_boost_weight=0.7, zero_ball_vel_weight=0.8),
                         Curvedash(zero_boost_weight=0.7, zero_ball_vel_weight=0.8),
                         RandomEvenRecovery(zero_boost_weight=0.7, zero_ball_vel_weight=0.8),
                         Chaindash(zero_boost_weight=0.7, zero_ball_vel_weight=0.8),
                         Walldash(zero_boost_weight=0.7, zero_ball_vel_weight=0.8),
                         Wavedash(zero_boost_weight=0.7, zero_ball_vel_weight=0.8),
                         RecoverySetter(zero_boost_weight=0.7, zero_ball_vel_weight=0.8)
                         ),
                        (0.1, 0.1, 0.25, 0.25, 0, 0.15, 0.15)
                    )
                )

        elif mode == "doubletap":
            for i in range(4):
                self.setters.append(
                    WeightedSampleSetter(
                        (AugmentSetter(ReplaySetter(double_tap_replays[i], defender_front_goal_weight=0,
                                                   random_boost=True, dtap_dict=dtap_dict, initial_state_dict=(0, 0, 0)),
                                       True, False, False),
                         AugmentSetter(ReplaySetter(easy_double_tap_replays[i], defender_front_goal_weight=0,
                                                    random_boost=True, dtap_dict=dtap_dict, initial_state_dict=(1, 0, 0)),
                                       True, False, False),
                         ), (0.8, 0.2))
                )

        elif mode == "recovery_ball":
            for i in range(3):
                self.setters.append(
                    WeightedSampleSetter(
                        (HalfFlip(zero_boost_weight=0.7, zero_ball_vel_weight=0, ball_zero_z=True, ball_vel_mult=3),
                         Curvedash(zero_boost_weight=0.7, zero_ball_vel_weight=0, ball_zero_z=True, ball_vel_mult=2),
                         RandomEvenRecovery(zero_boost_weight=0.7, zero_ball_vel_weight=0, ball_zero_z=True,
                                            ball_vel_mult=3),
                         Chaindash(zero_boost_weight=0.7, zero_ball_vel_weight=0, ball_zero_z=True, ball_vel_mult=3),
                         Walldash(zero_boost_weight=0.7, zero_ball_vel_weight=0, ball_zero_z=True, ball_vel_mult=3),
                         Wavedash(zero_boost_weight=0.7, zero_ball_vel_weight=0, ball_zero_z=True, ball_vel_mult=3),
                         RecoverySetter(zero_boost_weight=0.7, zero_ball_vel_weight=0, ball_zero_z=True,
                                        ball_vel_mult=3)
                         ),
                        (0, 0.075, 0.2, 0.4, 0.075, 0.1, 0.15)
                    )
                )
                # self.setters.append(
                #     WeightedSampleSetter(
                #         (
                #             AugmentSetter(ReplaySetter(low_recovery_replays[i], random_boost=False,
                #                                        zero_ball_weight=0.8,
                #                                        zero_car_weight=0.2,
                #                                        rotate_car_weight=0.2,
                #                                        backward_car_weight=0.2,
                #                                        vel_div_weight=0.1,
                #                                        special_loc_weight=0.1,
                #                                        zero_boost_weight=0.2,
                #                                        ), False, True, False),
                #             AugmentSetter(ReplaySetter(high_recovery_replays[i], random_boost=False,
                #                                        zero_ball_weight=0.8,
                #                                        zero_car_weight=0.2,
                #                                        rotate_car_weight=0.2,
                #                                        backward_car_weight=0.2,
                #                                        vel_div_weight=0.1,
                #                                        special_loc_weight=0.1,
                #                                        zero_boost_weight=0.2,
                #                                        ), False, True, False),
                #             HalfFlip(),
                #         ),
                #         (0.75, 0.15, 0.1)
                #     )
                # )

    def reset(self, state_wrapper: StateWrapper):
        # if self.end_object_choice is not None and self.end_object_choice == "random":
        #     self.end_object_tracker[0] += 1
        #     if self.end_object_tracker[0] == 7:
        #         self.end_object_tracker[0] = 0
        index = 3 if len(state_wrapper.cars) == 1 else (len(state_wrapper.cars) // 2) - 1
        self.setters[index].reset(state_wrapper)
