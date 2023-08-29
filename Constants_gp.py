from pretrained_agents.nexto.nexto_v2 import NextoV2
from pretrained_agents.KBB.kbb import KBB

FRAME_SKIP = 4
TIME_HORIZON = 16  # horizon in seconds
T_STEP = FRAME_SKIP / 120   # real time per rollout step
ZERO_SUM = True
STEP_SIZE = 2_000_000
DB_NUM = 4

model_name = "nexto-model.pt"
nexto = NextoV2(model_string=model_name, n_players=6)
model_name = "kbb.pt"
kbb = KBB(model_string=model_name)

pretrained_agents = {
    nexto: {'prob': 0.75, 'eval': True, 'p_deterministic_training': 1., 'key': "Nexto"},
    kbb: {'prob': 0.25, 'eval': True, 'p_deterministic_training': 1., 'key': "KBB"}
    }
