import torch
import os

from rocket_learn.agent.discrete_policy import DiscretePolicy
from torch.nn import Linear, Sequential, GELU
from rocket_learn.utils.util import SplitLayer


# TODO add your network here


actor = Sequential(Linear(237, 512), GELU(), Linear(512, 512), GELU(),
                       Linear(512, 512), GELU(), Linear(512, 512), GELU(), Linear(512, 90))

actor = DiscretePolicy(actor, (90,))

# PPO REQUIRES AN ACTOR/CRITIC AGENT

cur_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint = torch.load(os.path.join(cur_dir, "checkpoint.pt"))
actor.load_state_dict(checkpoint['actor_state_dict'])
actor.eval()
torch.jit.save(torch.jit.script(actor), 'jit.pt')

exit(0)