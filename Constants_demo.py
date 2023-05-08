from pretrained_agents.nexto.nexto_v2 import NextoV2
from pretrained_agents.KBB.kbb import KBB

FRAME_SKIP = 4
TIME_HORIZON = 2  # horizon in seconds
T_STEP = FRAME_SKIP / 120   # real time per rollout step
ZERO_SUM = False
STEP_SIZE = 500_000
DB_NUM = 14

model_name = "nexto-model.pt"
nexto = NextoV2(model_string=model_name, n_players=6)
model_name = "kbb.pt"
kbb = KBB(model_string=model_name)

pretrained_agents = {
    nexto: {'prob': 0.5, 'eval': True, 'p_deterministic_training': 1., 'key': "Nexto"},
    kbb: {'prob': 0.5, 'eval': True, 'p_deterministic_training': 1., 'key': "KBB"}
    }
