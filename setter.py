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
        replays = ["ssl_1v1.npy", "ssl_2v2.npy", "ssl_3v3.npy"]
        aerial_replays = ["aerial_ssl_1v1.npy", "aerial_ssl_2v2.npy", "aerial_ssl_3v3.npy"]
        flip_reset_replays = ["flip_resets_ssl_1v1.npy", "flip_resets_ssl_2v2.npy", "flip_resets_ssl_3v3.npy"]
        kickoff_replays = ["kickoff_ssl_1v1.npy", "kickoff_ssl_2v2.npy", "kickoff_ssl_3v3.npy"]
        ceiling_replays = ["ceiling_ssl_1v1.npy", "ceiling_ssl_2v2.npy", "ceiling_ssl_3v3.npy"]
        if mode is None or mode is "normal":
            for i in range(3):
                self.setters.append(
                    WeightedSampleSetter(
                        (
                            DefaultState(),
                            AugmentSetter(ReplaySetter(replays[i])),
                            AugmentSetter(ReplaySetter(aerial_replays[i])),
                            AugmentSetter(GroundAirDribble(), True, False, False),
                            AugmentSetter(WallDribble(), True, False, False),
                            AugmentSetter(RandomState(cars_on_ground=True)),
                            AugmentSetter(RandomState(cars_on_ground=False)),
                        ),
                        # (0.05, 0.50, 0.20, 0.20, 0.025, 0.025)
                        (0.5, 0.2, 0.1, 0, 0, 0.1, 0.1)
                    )
                )
        elif mode is "kickoff":
            for i in range(3):
                self.setters.append(
                    AugmentSetter(ReplaySetter(kickoff_replays[i]))
                )
        elif mode is "aerial":
            for i in range(3):
                self.setters.append(
                    WeightedSampleSetter(
                        (
                            AugmentSetter(ReplaySetter(aerial_replays[i])),
                            AugmentSetter(ReplaySetter(flip_reset_replays[i])),
                            AugmentSetter(ReplaySetter(ceiling_replays[i])),
                            AugmentSetter(WallDribble(), True, False, False),
                            AugmentSetter(RandomState(cars_on_ground=False)),
                        ),
                        (0.55, 0.15, 0.15, 0.1, 0.05)
                    )
                )

    def reset(self, state_wrapper: StateWrapper):
        self.setters[(len(state_wrapper.cars) // 2) - 1].reset(state_wrapper)
