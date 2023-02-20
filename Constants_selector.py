FRAME_SKIP = 4
TIME_HORIZON = 7  # horizon in seconds
T_STEP = FRAME_SKIP / 120   # real time per rollout step
ZERO_SUM = False
STEP_SIZE = 400_000
DB_NUM = 7
STACK_SIZE = 5
SELECTION_CHANNEL = "on_model_selection"

SUB_MODEL_NAMES = [
    "kickoff_1",
    "kickoff_2",
    "GP",
    "aerial",
    "flick_bump",
    "flick",
    "flip_reset_1",
    "flip_reset_2",
    "flip_reset_3",
    "pinch",
    "recover_0",
    "recover_-45",
    "recover_-90",
    "recover_-135",
    "recover_180",
    "recover_135",
    "recover_90",
    "recover_45",
    "recover_back_left",
    "recover_back_right",
    "recover_opponent",
    "recover_back_post",
    "recover_ball",
    "halfflip_back",
    "speedflip",
    "halfflip_ball",
    "walldash_straight",
    "walldash_up",
    "walldash_down",
    "walldash_boost",
    "walldash_ball",
]
