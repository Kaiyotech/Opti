from rocket_learn.utils.dynamic_gamemode_setter import DynamicGMSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.state_setters import DefaultState
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter
from rlgym.utils.state_setters.random_state import RandomState
from mybots_statesets import GroundAirDribble, WallDribble


class CoyoteSetter(DynamicGMSetter):
    def __init__(self, mode):
        self.setters = []  # [1v1, 2v2, 3v3]
        replays = ["replays/ssl_1v1.npy", "replays/ssl_2v2.npy", "replays/ssl_3v3.npy"]
        aerial_replays = ["replays/aerial_1v1.npy", "replays/aerial_2v2.npy", "replays/aerial_3v3.npy"]
        flip_reset_replays = ["replays/flip_reset_1v1.npy", "replays/flip_reset_2v2.npy",
                              "replays/flip_reset_3v3.npy"]
        kickoff_replays = ["replays/kickoff_1v1.npy", "replays/kickoff_2v2.npy", "replays/kickoff_3v3.npy"]
        ceiling_replays = ["replays/ceiling_1v1.npy", "replays/ceiling_2v2.npy", "replays/ceiling_3v3.npy"]
        air_dribble_replays = ["replays/air_dribble_1v1.npy", "replays/air_dribble_2v2.npy",
                               "replays/air_dribble_3v3.npy"]
        team_pinch_replays = ["replays/pinch_1v1.npy", "replays/team_pinch_2v2.npy", "replays/team_pinch_3v3.npy"]
        pinch_replays = ["replays/pinch_1v1.npy", "replays/pinch_2v2.npy", "replays/pinch_3v3.npy"]
        double_tap_replays = ["replays/double_tap_1v1.npy", "replays/double_tap_2v2.npy", "replays/double_tap_3v3.npy"]

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
                        (0.3, 0.45, 0.1, 0.025, 0.025, 0.05, 0.05)
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
                        (0.65, 0.10, 0.25)
                    )
                )

    def reset(self, state_wrapper: StateWrapper):
        self.setters[(len(state_wrapper.cars) // 2) - 1].reset(state_wrapper)
