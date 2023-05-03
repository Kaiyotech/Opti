from rocket_learn.matchmaker.base_matchmaker import BaseMatchmaker
from trueskill import Rating


class MatchmakerWith1v0(BaseMatchmaker):

    def __init__(self):
        self.rating = Rating(0, 1)

    def generate_matchup(self, redis, n_agents, evaluate):

        # Doing this instead of int(redis.get(VERSION_LATEST)) because the latest model is
        # whatever is currently in rollout worker
        latest_version = -1

        return [latest_version] * n_agents, [self.rating] * n_agents, False, 1, n_agents // 2
