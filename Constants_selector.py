from pretrained_agents.nexto.nexto_v2 import NextoV2
from pretrained_agents.necto.necto_v1 import NectoV1
from pretrained_agents.KBB.kbb import KBB

FRAME_SKIP = 4
TIME_HORIZON = 7  # horizon in seconds
T_STEP = FRAME_SKIP / 120   # real time per rollout step
ZERO_SUM = True
STEP_SIZE = 1_000_000
DB_NUM = 7
STACK_SIZE = 5
SELECTION_CHANNEL = "on_model_selection"

SUB_MODEL_NAMES = [
    "kickoff",
    "GP",
    "aerial",
    "flick_bump",
    "flip_reset",
    "recover_b_post",
    "recover_ball",
    "walldash",
    "doubletap",
    "wall_play",

]

model_name = "nexto-model.pt"
nexto = NextoV2(model_string=model_name, n_players=6)
model_name = "kbb.pt"
kbb = KBB(model_string=model_name)
model_name = "necto-model-30Y.pt"
necto = NectoV1(model_string=model_name, n_players=6)

pretrained_agents = {
    nexto: {'prob': 0.33, 'eval': True, 'p_deterministic_training': 1., 'key': "Nexto"},
    kbb: {'prob': 0.33, 'eval': True, 'p_deterministic_training': 1., 'key': "KBB"},
    necto: {'prob': 0.33, 'eval': True, 'p_deterministic_training': 1., 'key': "Necto"},
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
