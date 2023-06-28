from pretrained_agents.nexto.nexto_v2 import NextoV2
from pretrained_agents.necto.necto_v1 import NectoV1
from pretrained_agents.KBB.kbb import KBB

FRAME_SKIP = 4
TIME_HORIZON = 7  # horizon in seconds
T_STEP = FRAME_SKIP / 120   # real time per rollout step
ZERO_SUM = True
STEP_SIZE = 400_000
DB_NUM = 8
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
    "recover_↑",
    "recover_↗",
    "recover_→",
    "recover_↘",
    "recover_↓",
    "recover_↙",
    "recover_←",
    "recover_↖",
    "recover_b_left",
    "recover_b_right",
    "recover_b_post",
    "recover_ball",
    "halfflip_back",
    "speedflip",
    "halfflip_ball",
    "walldash_str",
    # "walldash_up",
    # "walldash_down",
    "walldash_boost",
    "demo",
    "doubletap",
    "wall_play",
    # "left_turn",
    # "straight",
    # "right_turn"

]

model_name = "nexto-model.pt"
nexto = NextoV2(model_string=model_name, n_players=6)
model_name = "kbb.pt"
kbb = KBB(model_string=model_name)
model_name = "necto-model-30Y.pt"
necto = NectoV1(model_string=model_name, n_players=6)

pretrained_agents = {
    nexto: {'prob': 0, 'eval': True, 'p_deterministic_training': 1., 'key': "Nexto"},
    kbb: {'prob': 0, 'eval': True, 'p_deterministic_training': 1., 'key': "KBB"},
    necto: {'prob': 1, 'eval': True, 'p_deterministic_training': 1., 'key': "Necto"},
    }

# "recover_0",
# "recover_-45",
# "recover_-90",
# "recover_-135",
# "recover_180",
# "recover_135",
# "recover_90",
# "recover_45",

# "recover_12oclock",
# "recover_1030oclock",
# "recover_9oclock",
# "recover_730oclock",
# "recover_6oclock",
# "recover_430oclock",
# "recover_3oclock",
# "recover_130oclock",
