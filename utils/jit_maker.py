import torch
import os
import sys
import inspect

from rocket_learn.agent.discrete_policy import DiscretePolicy
from torch.nn import Linear, Sequential, GELU, LeakyReLU

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from agent import Opti

# TODO add your network here

filename = sys.argv[1]
splits = filename.split('_')
model_type = splits[0]
number = ""
if len(splits) > 2:
    number = splits[1]

print(f"model type is \"{model_type}\"")

# actor for pinch
if model_type == "pinch":
    actor = Sequential(Linear(222, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(),
                       Linear(512, 373))

    actor = DiscretePolicy(actor, (373,))

# actor for kickoff
if model_type == "kickoff":
    actor = Sequential(Linear(426, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(),
                       Linear(512, 373))

    actor = DiscretePolicy(actor, (373,))

# actor for gp
if model_type == "gp":
    actor = Sequential(Linear(426, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(), Linear(512, 512),
                       LeakyReLU(),
                       Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(),
                       Linear(512, 373))
    actor = Opti(embedder=Sequential(Linear(35, 128), LeakyReLU(), Linear(128, 35 * 5)), net=actor)

    actor = DiscretePolicy(actor, shape=(373,))

# actor for selector
if model_type == "selector":
    input_size = 426 + 20
    action_size = 23

    actor = Sequential(Linear(input_size, 256), LeakyReLU(), Linear(256, 256), LeakyReLU(), Linear(256, 128),
                       LeakyReLU(),
                       Linear(128, action_size))

    actor = Opti(embedder=Sequential(Linear(35, 128), LeakyReLU(), Linear(128, 35 * 5)), net=actor)

    actor = DiscretePolicy(actor, shape=(action_size,))

# actor for flip reset
if model_type == "flipreset":
    # actor = Sequential(Linear(222, 256), LeakyReLU(), Linear(256, 256), LeakyReLU(),
    #                    Linear(256, 256), LeakyReLU(),
    #                    Linear(256, 373))
    #
    # actor = DiscretePolicy(actor, (373,))
    actor = Sequential(Linear(222, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(),
                       Linear(512, 373))

    actor = DiscretePolicy(actor, (373,))

# actor for flick
if model_type == "flick":
    actor = Sequential(Linear(426, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(), Linear(512, 512),
                       LeakyReLU(),
                       Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(),
                       Linear(512, 373))
    actor = Opti(embedder=Sequential(Linear(35, 128), LeakyReLU(), Linear(128, 35 * 5)), net=actor)

    actor = DiscretePolicy(actor, shape=(373,))

# actor for ceiling pinch
if model_type == "ceilingpinch":
    actor = Sequential(Linear(222, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(),
                       Linear(512, 373))

    actor = DiscretePolicy(actor, (373,))

# actor for aerial
if model_type == "aerial":
    actor = Sequential(Linear(222, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(),
                       Linear(512, 373))

    actor = DiscretePolicy(actor, (373,))

# actor for recovery
if model_type == "recovery":
    actor = Sequential(Linear(222, 128), LeakyReLU(), Linear(128, 128), LeakyReLU(),
                       Linear(128, 128), LeakyReLU(),
                       Linear(128, 373))

    actor = DiscretePolicy(actor, (373,))

# PPO REQUIRES AN ACTOR/CRITIC AGENT
cur_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint = torch.load(os.path.join(cur_dir, filename))
actor.load_state_dict(checkpoint['actor_state_dict'])
actor.eval()
new_name = filename.split("_checkpoint")[0] + "_jit.pt"
if model_type == "gp" or model_type == "flick":
    test_input_embed = (torch.Tensor(1, 251), torch.Tensor(1, 5, 35))
    torch.jit.save(torch.jit.trace(actor, example_inputs=(test_input_embed,)), new_name)
elif model_type == "selector":
    test_input_embed = (torch.Tensor(1, 251 + 20), torch.Tensor(1, 5, 35))
    torch.jit.save(torch.jit.trace(actor, example_inputs=(test_input_embed,)), new_name)
else:
    test_input_norm = torch.Tensor(actor.net._modules['0'].in_features)
    torch.jit.save(torch.jit.trace(actor, example_inputs=(test_input_norm,)), new_name)

exit(0)
