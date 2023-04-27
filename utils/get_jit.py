from rocket_learn.rollout_generator.redis.utils import _unserialize_model, OPPONENT_MODELS  # noqa
import sys
from redis import Redis
import os
import torch
import pickle
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import agent  # for pickle

filename = sys.argv[1]
db_num = sys.argv[2]
version = sys.argv[3]  # like Opti_GP-v348, use hkeys opponent-models in redis to check

r = Redis(host="127.0.0.1",
          username="user1",
          password=os.environ["redis_user1_key"],
          db=int(db_num),
          )
model_ser = r.hget(OPPONENT_MODELS, version)
model = pickle.loads(model_ser)
model.eval()

test_input_embed = (torch.Tensor(1, 251), torch.Tensor(1, 5, 35))
torch.jit.save(torch.jit.trace(model, example_inputs=(test_input_embed,)), filename)


