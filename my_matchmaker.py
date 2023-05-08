from rocket_learn.matchmaker.base_matchmaker import BaseMatchmaker
from trueskill import Rating
from rocket_learn.agent.types import PretrainedAgents
import numpy as np

class MatchmakerWith1v0(BaseMatchmaker):

    def __init__(self):
        self.rating = Rating(0, 1)

    def generate_matchup(self, redis, n_agents, evaluate):

        # Doing this instead of int(redis.get(VERSION_LATEST)) because the latest model is
        # whatever is currently in rollout worker
        latest_version = -1

        return [latest_version] * n_agents, [self.rating] * n_agents, False, 1, n_agents // 2


class MatchmakerSimple(BaseMatchmaker):

    def __init__(self):
        self.rating = Rating(0, 1)

    def generate_matchup(self, redis, n_agents, evaluate):

        # Doing this instead of int(redis.get(VERSION_LATEST)) because the latest model is
        # whatever is currently in rollout worker
        latest_version = -1

        return [latest_version] * n_agents, [self.rating] * n_agents, False, n_agents // 2, n_agents // 2


class MatchmakerFullVPretrained(BaseMatchmaker):

    def __init__(self, pretrained_agents: PretrainedAgents = None):
        if pretrained_agents is not None:
            self.consider_pretrained = True
            pretrained_agents_keys, pretrained_agents_values = zip(
                *pretrained_agents.items())
            self.pretrained_agents = pretrained_agents_keys
            pretrained_probs = [p["prob"] for p in pretrained_agents_values]
            self.pretrained_probs = np.array(
                pretrained_probs) / sum(pretrained_probs)
            self.pretrained_evals = [p["eval"]
                                     for p in pretrained_agents_values]
            self.pretrained_p_deterministic_training = [
                p["p_deterministic_training"] if p["p_deterministic_training"] is not None else 1 for p in pretrained_agents_values]
            self.pretrained_keys = [p["key"] for p in pretrained_agents_values]
            self.pretrained_eval_keys = [k for i, k in enumerate(
                self.pretrained_keys) if self.pretrained_evals[i]]
        self.rating = Rating(0, 1)

    def generate_matchup(self, redis, n_agents, evaluate):

        # Doing this instead of int(redis.get(VERSION_LATEST)) because the latest model is
        # whatever is currently in rollout worker
        latest_version = -1
        versions = [latest_version] * (n_agents // 2)

        n_each_pretrained = np.random.multinomial(
            1, self.pretrained_probs)
        pretrained_idx = list(n_each_pretrained).index(1)
        pretrained_key = self.pretrained_keys[pretrained_idx]
        versions += [pretrained_key] * (n_agents // 2)

        return versions, [self.rating] * n_agents, False, n_agents // 2, n_agents // 2
