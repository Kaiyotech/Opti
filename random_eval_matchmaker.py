import numpy as np
import itertools

from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.matchmaker.base_matchmaker import BaseMatchmaker
from rocket_learn.rollout_generator.redis.utils import get_rating, get_ratings, get_pretrained_ratings, LATEST_RATING_ID
from rocket_learn.utils.util import probability_NvsM
from rocket_learn.agent.types import PretrainedAgents


class RandomEvalMatchmaker(BaseMatchmaker):
# from JPK314 just didn't feel like updating rocket learn right now to use it
    def __init__(self, sigma_target=1, pretrained_agents: PretrainedAgents = None, non_latest_version_prob=[1, 0, 0, 0],
                 past_version_prob=1, full_team_trainings=0, full_team_evaluations=0, force_non_latest_orange=False,
                 showmatch=False,
                 orange_agent_text_file=None, min_to_test=0):
        """
        :param sigma_target: The sigma value at which agents stop playing in eval matches frequently
        :param pretrained_agents: a configuration dict for how and how often to use pretrained agents in matchups.
        :param non_latest_version_prob: An array such that the ith index is the probability that i agents in a training matchup are not the latest version of the model.
        :param past_version_prob: The probability that a non-latest agent in the matchup is a past version of the model.
        :param full_team_trainings: The probability that a match uses all agents of the same type on a given team in training.
        :param full_team_evaluations: The probability that a match uses all agents of the same type on a given team in evals.
        :param force_non_latest_orange: A boolean that, if true, ensures the first player in the list is latest agent (if one exists in the match).
        """
        self.min_to_test = min_to_test
        self.orange_agents_text_file = orange_agent_text_file
        self.showmatch = showmatch
        self.sigma_target = sigma_target
        self.non_latest_version_prob = np.array(
            non_latest_version_prob) / sum(non_latest_version_prob)
        self.past_version_prob = 1 if pretrained_agents is None else past_version_prob
        self.full_team_trainings = full_team_trainings or showmatch
        self.full_team_evaluations = full_team_evaluations or showmatch
        self.force_non_latest_orange = force_non_latest_orange or showmatch
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
        else:
            self.consider_pretrained = False

    def generate_matchup(self, redis, n_agents, evaluate):
        full_team_match = np.random.random() < (
            self.full_team_evaluations if evaluate else self.full_team_trainings)
        if evaluate:
            n_agents = np.random.choice([2, 4, 6])
        if not evaluate:
            n_non_latest = np.random.choice(
                len(self.non_latest_version_prob), p=self.non_latest_version_prob)
            if full_team_match:
                # We make either one or zero non-latest pick
                n_non_latest = int(n_non_latest > 0)
            else:
                # We only want a maximum of half the agents to be non latest in training matchups
                n_non_latest = min(n_agents // 2, n_non_latest)
            n_past_version = np.random.binomial(
                n_non_latest, self.past_version_prob)
            if self.consider_pretrained:
                n_each_pretrained = np.random.multinomial(
                    n_non_latest-n_past_version, self.pretrained_probs)

        per_team = n_agents // 2
        gamemode = f"{per_team}v{per_team}"
        gamemode = '1v0' if gamemode == '0v0' else gamemode
        latest_id = redis.get(LATEST_RATING_ID).decode("utf-8")
        latest_key = f"{latest_id}-stochastic"

        # This is the version of the most recent model (NOT just eval models)
        # Doing this instead of int(redis.get(VERSION_LATEST)) because the latest model is whatever is currently in rollout worker
        latest_version = -1

        # This is a training match with all latest agents, no further logic necessary
        if not evaluate and n_non_latest == 0:
            rating = get_rating(gamemode, latest_key, redis)
            return [latest_version] * n_agents, [rating] * n_agents, False, n_agents // 2, n_agents // 2

        past_version_ratings = get_ratings(gamemode, redis)
        past_version_ratings = {k: v for k, v in past_version_ratings.items()
                                if int(k.split('-')[1].split('v')[1]) >= self.min_to_test}

        # This is the rating of the most recent eval model
        latest_rating = past_version_ratings[latest_key]
        past_version_ratings_keys, past_version_ratings_values = zip(
            *past_version_ratings.items())

        # We also have the pretrained agents' ratings (the ones that have eval set to true, that is)
        if self.consider_pretrained:
            pretrained_ratings = get_pretrained_ratings(gamemode, redis)
            # Actually, we need all the ratings, but we'll split out the ones just to be used in the eval pool
            pretrained_ratings_items = pretrained_ratings.items()
            if pretrained_ratings.items():
                pretrained_ratings_keys, pretrained_ratings_values = zip(
                    *pretrained_ratings_items)
            else:
                pretrained_ratings_keys = ()
                pretrained_ratings_values = ()
            pretrained_ratings_items_eval = [p for p in pretrained_ratings_items if "-".join(
                p[0].split("-")[:-1]) in self.pretrained_eval_keys or p[0] in self.pretrained_eval_keys]
            if pretrained_ratings_items_eval:
                pretrained_ratings_keys_eval, pretrained_ratings_values_eval = zip(
                    *pretrained_ratings_items_eval)
            else:
                pretrained_ratings_keys_eval = ()
                pretrained_ratings_values_eval = ()
        else:
            pretrained_ratings = []
            pretrained_ratings_keys_eval = ()
            pretrained_ratings_values_eval = ()

        all_ratings_keys = past_version_ratings_keys + pretrained_ratings_keys_eval
        all_ratings_values = past_version_ratings_values + pretrained_ratings_values_eval

        if evaluate and len(all_ratings_keys) < 2:
            # Can't run evaluation game, not enough agents
            return [latest_version] * n_agents, [latest_rating] * n_agents, False, n_agents // 2, n_agents // 2

        if evaluate:  # Evaluation game, try to find agents with high sigma
            sigmas = np.array(
                [r.sigma for r in all_ratings_values])
            probs = np.clip(sigmas - self.sigma_target, a_min=0, a_max=None)
            s = probs.sum()
            if s == 0:  # No versions with high sigma available
                if np.random.normal(0, self.sigma_target) > 1:
                    # Some chance of doing a match with random versions, so they might correct themselves
                    probs = np.ones_like(probs) / len(probs)
                else:
                    return [latest_version] * n_agents, [latest_rating] * n_agents, False, n_agents // 2, n_agents // 2
            else:
                probs /= s
            chosen_first_idx = np.random.choice(len(probs), p=probs)
            chosen_first_key = all_ratings_keys[chosen_first_idx]
            chosen_first_rating = all_ratings_values[chosen_first_idx]
        else:
            chosen_first_key = latest_version
            chosen_first_rating = latest_rating

        if full_team_match:
            versions = [chosen_first_key] * per_team
            ratings = [chosen_first_rating] * per_team
        else:
            versions = [chosen_first_key]
            ratings = [chosen_first_rating]

        if not evaluate:
            # In a training match, we also need to add in the pretrained agents we have already decided upon
            has_pretrained = n_non_latest - n_past_version > 0
            if full_team_match and has_pretrained:
                pretrained_idx = list(n_each_pretrained).index(1)
                use_deterministic = np.random.random(
                ) < self.pretrained_p_deterministic_training[pretrained_idx]
                pretrained_agent = self.pretrained_agents[pretrained_idx]
                if isinstance(pretrained_agent, DiscretePolicy):
                    pretrained_key = self.pretrained_keys[pretrained_idx] + (
                        "-deterministic" if use_deterministic else "-stochastic")
                else:
                    pretrained_key = self.pretrained_keys[pretrained_idx]
                versions += [pretrained_key] * per_team
                ratings += [pretrained_ratings_values[pretrained_ratings_keys.index(
                    pretrained_key)]] * per_team
                if self.force_non_latest_orange or np.random.random() < 0.5:
                    # replace latest with peak
                    if self.showmatch:
                        assert full_team_match
                        sorted_ratings = {k: v for k, v in
                                          sorted(past_version_ratings.items(), key=lambda item: item[1].mu)}
                        peak_version = list(sorted_ratings)[-1]
                        for i in range(per_team):
                            versions[i] = peak_version
                    if self.orange_agents_text_file is not None:
                        with open(self.orange_agents_text_file, 'w') as file:
                            file.write(versions[per_team])
                    return versions, ratings, False, n_agents // 2, n_agents // 2
                else:
                    mid = len(versions) // 2
                    return (versions[mid:] + versions[:mid]), (ratings[mid:] + ratings[:mid]), False, n_agents // 2, n_agents // 2
            elif has_pretrained:
                for i, n_agent in enumerate(n_each_pretrained):
                    pretrained_agent = self.pretrained_agents[i]
                    if isinstance(pretrained_agent, DiscretePolicy):
                        n_deterministic = np.random.binomial(
                            n_agent, self.pretrained_p_deterministic_training[i])
                        versions += [self.pretrained_keys[i] +
                                     "-deterministic"] * n_deterministic
                        versions += [self.pretrained_keys[i] +
                                     "-stochastic"] * (n_agent - n_deterministic)
                        ratings += [pretrained_ratings_values[pretrained_ratings_keys.index(
                            self.pretrained_keys[i] + "-deterministic")]] * n_deterministic
                        ratings += [pretrained_ratings_values[pretrained_ratings_keys.index(
                            self.pretrained_keys[i] + "-stochastic")]] * (n_agent - n_deterministic)
                    else:
                        versions += [self.pretrained_keys[i]] * n_agent
                        ratings += [pretrained_ratings_values[pretrained_ratings_keys.index(
                            self.pretrained_keys[i])]] * n_agent

            # and the number of agents that are latest. If full_team_match is true then at this point in execution we must have n_past_version == 1 so there is nothing to add
            if not full_team_match:
                versions += [latest_version] * (n_agents - n_non_latest - 1)
                ratings += [latest_rating] * (n_agents - n_non_latest - 1)

        # At this point in execution, ratings contains all the information necessary to randomly fill the remaining spots with players via matchmaking pool.
        # If evaluate, this pool includes pretrained. If not, this pool is only past versions.

        ratings_values_pool = all_ratings_values if evaluate else past_version_ratings_values
        ratings_keys_pool = all_ratings_keys if evaluate else past_version_ratings_keys
        probs = np.zeros(len(ratings_values_pool))

        # The point of this matchmaker is to use fully random evals - no concept of fairness. Therefore we will weight all agents with equal probability of being chosen.
        # In evaluation matches where full_team_match is true, we don't want to have our remaining pick be the same agent as chosen_first.
        # Outside of evaluation matches, nothing changes.
        for i, rating in enumerate(ratings_values_pool):
            if evaluate and full_team_match and ratings_keys_pool[i] == versions[0]:
                p = 0
            elif evaluate:
                p = 0.5
            else:
                p = probability_NvsM(
                    [rating] * per_team, [ratings[0]] * per_team)
            probs[i] = p * (1 - p)

        probs /= probs.sum()

        if full_team_match:
            chosen_idx = np.random.choice(len(probs), p=probs)
            versions += [ratings_keys_pool[chosen_idx]] * per_team
            ratings += [ratings_values_pool[chosen_idx]] * per_team
            if self.force_non_latest_orange or np.random.random() < 0.5:
                return versions, ratings, evaluate, n_agents // 2, n_agents // 2
            else:
                mid = len(versions) // 2
                return (versions[mid:] + versions[:mid]), (ratings[mid:] + ratings[:mid]), evaluate, n_agents // 2, n_agents // 2
        else:
            chosen_idxs = np.random.choice(
                len(probs), size=n_agents - 1 if evaluate else n_past_version, p=probs)
            for chosen_idx in chosen_idxs:
                versions += [ratings_keys_pool[chosen_idx]]
                ratings += [ratings_values_pool[chosen_idx]]

            # We now have all the versions and ratings that will be used in parallel lists. Same as above - if evaluate, no fairness weighting.
            # Otherwise, weight all permutations of matches by fairness and pick one
            if len(versions) != n_agents:
                print("what??")
            matchups = []
            qualities = []
            for team1 in itertools.combinations(range(n_agents), per_team):
                team2 = [idx for idx in range(n_agents) if idx not in team1]
                team1_ratings_keys = [versions[v] for v in team1]
                team1_ratings_values = [ratings[v] for v in team1]
                team2_ratings_keys = [versions[v] for v in team2]
                team2_ratings_values = [ratings[v] for v in team2]
                # Don't want team against team in evals
                if sorted(str(k) for k in team1_ratings_keys) == sorted(str(k) for k in team2_ratings_keys):
                    p = 0
                elif evaluate:
                    p = 0.5
                else:
                    p = probability_NvsM(
                        team1_ratings_values, team2_ratings_values)
                matchups.append(list(team1))
                qualities.append(p * (1 - p))
            qualities = np.array(qualities)
            s = qualities.sum()
            if s == 0:
                # Bad luck, just do regular match
                return [latest_version] * n_agents, [latest_rating] * n_agents, False, n_agents // 2, n_agents // 2

            # Pick a matchup, randomize ordering within team, and reorder versions and ratings accordingly
            # Due to the way combinations works, randomizing which team team1 is on (orange or blue) doesn't change anything
            matchup_idx = np.random.choice(len(matchups), p=qualities / s)
            team1 = matchups[matchup_idx]
            team2 = [idx for idx in range(n_agents) if idx not in team1]
            np.random.shuffle(team1)
            np.random.shuffle(team2)
            if self.force_non_latest_orange and latest_version in versions:
                latest_version_idx = versions.index(latest_version)
                if latest_version_idx in team1:
                    blue_team = team1
                    orange_team = team2
                else:
                    blue_team = team2
                    orange_team = team1
                # Swap the latest version car to be in position 0
                temp_idx = blue_team.index(latest_version_idx)
                temp = blue_team[temp_idx]
                blue_team[temp_idx] = blue_team[0]
                blue_team[0] = temp
                team1 = blue_team
                team2 = orange_team

            ordering = team1 + team2
            versions = [versions[idx] for idx in ordering]
            ratings = [ratings[idx] for idx in ordering]

            return versions, ratings, evaluate, n_agents // 2, n_agents // 2
