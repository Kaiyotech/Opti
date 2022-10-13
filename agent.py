from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from torch import Tensor
from torch.nn import Linear


class ActorCriticEmbedderAgent(ActorCriticAgent):
    def __init__(self, actor, critic, embedder: Linear, optimizer):

        super().__init__(actor=actor, critic=critic, optimizer=optimizer)
        self.embedder = embedder

    def forward(self, *args, **kwargs):
        embedded = self.embedder((args[-1]))
        obs = args[:-1]
        obs.extend(embedded)
        return self.actor(obs, **kwargs), self.critic(obs, **kwargs)
