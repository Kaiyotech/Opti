from rocket_learn.utils.dynamic_gamemode_setter import DynamicGMSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.state_setters import DefaultState
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter
from rlgym.utils.state_setters.random_state import RandomState
from mybots_statesets import GroundAirDribble, WallDribble, FlickSetter


class CoyoteSetter(DynamicGMSetter):
    def __init__(self, mode):
        self.setters = []  # [1v1, 2v2, 3v3]
        replays = ["replays/ssl_1v1.npy", "replays/ssl_2v2.npy", "replays/ssl_3v3.npy"]
        aerial_replays = ["replays/aerial_1v1.npy", "replays/aerial_2v2.npy", "replays/aerial_3v3.npy"]
        flip_reset_replays = ["replays/flip_reset_1v1.npy", "replays/flip_reset_2v2.npy",
                              "replays/flip_reset_3v3.npy"]
        wall_flip_reset_replays = ["replays/wall_flip_reset_1v1.npy", "replays/wall_flip_reset_2v2.npy",
                              "replays/wall_flip_reset_3v3.npy"]
        ground_flip_reset_replays = ["replays/ground_flip_reset_1v1.npy", "replays/ground_flip_reset_2v2.npy",
                                   "replays/ground_flip_reset_3v3.npy"]
        kickoff_replays = ["replays/kickoff_1v1.npy", "replays/kickoff_2v2.npy", "replays/kickoff_3v3.npy"]
        ceiling_replays = ["replays/ceiling_1v1.npy", "replays/ceiling_2v2.npy", "replays/ceiling_3v3.npy"]
        air_dribble_replays = ["replays/air_dribble_1v1.npy", "replays/air_dribble_2v2.npy",
                               "replays/air_dribble_3v3.npy"]
        team_pinch_replays = ["replays/pinch_1v1.npy", "replays/team_pinch_2v2.npy", "replays/team_pinch_3v3.npy"]
        pinch_replays = ["replays/pinch_1v1.npy", "replays/pinch_2v2.npy", "replays/pinch_3v3.npy"]
        double_tap_replays = ["replays/double_tap_1v1.npy", "replays/double_tap_2v2.npy", "replays/double_tap_3v3.npy"]
        ground_dribble_replays = ["replays/ground_dribble_1v1.npy", "replays/ground_dribble_2v2.npy",
                               "replays/ground_dribble_3v3.npy"]
        demo_replays = ["replays/demos_1v1.npy", "replays/demos_2v2.npy", "replays/demos_3v3.npy"]

        if mode is None or mode == "normal":
            for i in range(3):
                self.setters.append(
                    WeightedSampleSetter(
                        (
                            DefaultState(),
                            AugmentSetter(ReplaySetter(replays[i], random_boost=True)),
                            AugmentSetter(ReplaySetter(aerial_replays[i], random_boost=True)),
                            AugmentSetter(GroundAirDribble(), True, False, False),
                            AugmentSetter(WallDribble(), True, False, False),
                            AugmentSetter(RandomState(cars_on_ground=True)),
                            AugmentSetter(RandomState(cars_on_ground=False)),
                        ),
                        # (0.05, 0.50, 0.20, 0.20, 0.025, 0.025)
                        (0.1, 0.7, 0.10, 0.025, 0.025, 0.025, 0.025)
                    )
                )
        elif mode == "selector":
            for i in range(3):
                self.setters.append(
                    WeightedSampleSetter(
                        (
                            DefaultState(),
                            AugmentSetter(ReplaySetter(kickoff_replays[i])),
                            AugmentSetter(ReplaySetter(replays[i], random_boost=True)),
                            AugmentSetter(ReplaySetter(aerial_replays[i], random_boost=True)),
                            AugmentSetter(GroundAirDribble(), True, False, False),
                            AugmentSetter(WallDribble(), True, False, False),
                            AugmentSetter(RandomState(cars_on_ground=True)),
                            AugmentSetter(RandomState(cars_on_ground=False)),
                            AugmentSetter(
                                ReplaySetter(flip_reset_replays[i], random_boost=True, remove_defender_weight=0.1)),
                            AugmentSetter(ReplaySetter(ceiling_replays[i], random_boost=True)),
                            AugmentSetter(ReplaySetter(air_dribble_replays[i], random_boost=True)),
                            AugmentSetter(
                                ReplaySetter(double_tap_replays[i], random_boost=True, remove_defender_weight=0.1)),
                            AugmentSetter(ReplaySetter(pinch_replays[i], random_boost=True)),
                            AugmentSetter(ReplaySetter(team_pinch_replays[i], random_boost=True)),
                        ),
                        # (0.05, 0.50, 0.20, 0.20, 0.025, 0.025)
                        (0.1, 0.05, 0.22, 0.05, 0.01, 0.02, 0.01, 0.01, 0.1, 0.1, 0.1, 0.09, 0.09, 0.05)
                    )
                )
        elif mode == "kickoff":
            for i in range(3):
                self.setters.append(
                    WeightedSampleSetter(
                        (
                            AugmentSetter(ReplaySetter(kickoff_replays[i])),
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
                            AugmentSetter(ReplaySetter(aerial_replays[i], random_boost=True)),
                            AugmentSetter(ReplaySetter(flip_reset_replays[i], random_boost=True, remove_defender_weight=0.1)),
                            AugmentSetter(ReplaySetter(ceiling_replays[i], random_boost=True)),
                            AugmentSetter(ReplaySetter(air_dribble_replays[i], random_boost=True)),
                            AugmentSetter(WallDribble(), True, False, False),
                            AugmentSetter(RandomState(cars_on_ground=False)),
                            AugmentSetter(ReplaySetter(double_tap_replays[i], random_boost=True, remove_defender_weight=0.1)),
                        ),
                        (0.05, 0.30, 0.20, 0.20, 0.10, 0.01, 0.14)
                    )
                )
        elif mode == "pinch":
            for i in range(3):
                self.setters.append(
                    WeightedSampleSetter(
                        (
                            AugmentSetter(ReplaySetter(pinch_replays[i], random_boost=True)),
                            AugmentSetter(ReplaySetter(team_pinch_replays[i], random_boost=True)),
                            AugmentSetter(WallDribble(), True, False, False),
                        ),
                        (0.85, 0.15, 0)
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
                                                       )
                                          )
                )
        elif mode == "flip_reset":
            for i in range(3):
                self.setters.append(
                    WeightedSampleSetter(
                        (
                            AugmentSetter(ReplaySetter(flip_reset_replays[i])),
                            AugmentSetter(WallDribble(), True, False, False),
                            AugmentSetter(WallDribble(speed_min=1450, speed_max=1700, max_rand_z=300), True, False, False),
                            AugmentSetter(ReplaySetter(wall_flip_reset_replays[i])),
                            AugmentSetter(WallDribble(speed_min=1700, speed_max=1900, max_rand_z=300), True, False,
                                          False),
                            AugmentSetter(ReplaySetter(ground_flip_reset_replays[i])),
                        ),
                        (0.05, 0.1, 0.2, 0.4, 0.05, 0.2)
                    )
                )

        elif mode == "demo":
            for i in range(3):
                self.setters.append(
                    AugmentSetter(ReplaySetter(demo_replays[i]))
                )

    def reset(self, state_wrapper: StateWrapper):
        self.setters[(len(state_wrapper.cars) // 2) - 1].reset(state_wrapper)
