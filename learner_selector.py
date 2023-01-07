import wandb
import torch.jit

from torch.nn import Linear, Sequential, LeakyReLU

from redis import Redis
from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from agent import Opti
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator
from CoyoteObs import CoyoteObsBuilder

from CoyoteParser import SelectorParser
import numpy as np
from rewards import ZeroSumReward
import Constants_selector

from utils.misc import count_parameters

import os
from torch import set_num_threads
from rocket_learn.utils.stat_trackers.common_trackers import Speed, Demos, TimeoutRate, Touch, EpisodeLength, Boost, \
    BehindBall, TouchHeight, DistToBall, AirTouch, AirTouchHeight, BallHeight, BallSpeed, CarOnGround, GoalSpeed, \
    MaxGoalSpeed
from my_stattrackers import GoalSpeedTop5perc
from rlgym.utils.reward_functions.common_rewards import VelocityReward, EventReward
from rlgym.utils.reward_functions.combined_reward import CombinedReward

# ideas for models:
# get to ball as fast as possible, sometimes with no boost, rewards exist
# pinches (ceiling and kuxir and team?), score in as few touches as possible with high velocity
# half flip, wavedash, wall dash, how to do this one?
# lix reset?
# normal play as well as possible, rewards exist
# aerial play without pinch, rewards exist
# kickoff, 5 second terminal, reward ball distance into opp half
set_num_threads(1)

if __name__ == "__main__":
    frame_skip = Constants_selector.FRAME_SKIP
    half_life_seconds = Constants_selector.TIME_HORIZON
    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    config = dict(
        actor_lr=1e-4,
        critic_lr=1e-4,
        n_steps=Constants_selector.STEP_SIZE,
        batch_size=100_000,
        minibatch_size=None,
        epochs=30,
        gamma=gamma,
        save_every=5,
        model_every=125,
        ent_coef=0.01,
    )

    run_id = "selector_run5.00"
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(dir="./wandb_store",
                        name="Selector_Run5.00",
                        project="Opti",
                        entity="kaiyotech",
                        id=run_id,
                        config=config,
                        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
                        )
    redis = Redis(username="user1", password=os.environ["redis_user1_key"],
                  db=Constants_selector.DB_NUM)  # host="192.168.0.201",
    redis.delete("worker-ids")

    stat_trackers = [
        Speed(normalize=True), Demos(), TimeoutRate(), Touch(), EpisodeLength(), Boost(), BehindBall(), TouchHeight(),
        DistToBall(), AirTouch(), AirTouchHeight(), BallHeight(), BallSpeed(normalize=True), CarOnGround(),
        GoalSpeed(), MaxGoalSpeed(), GoalSpeedTop5perc(),
    ]
    parser = SelectorParser()

    rollout_gen = RedisRolloutGenerator("Opti_Selector",
                                        redis,
                                        lambda: CoyoteObsBuilder(expanding=True,
                                                                 tick_skip=Constants_selector.FRAME_SKIP,
                                                                 team_size=3, extra_boost_info=True,
                                                                 embed_players=True,
                                                                 stack_size=Constants_selector.STACK_SIZE,
                                                                 action_parser=parser,
                                                                 selector=True,
                                                                 ),

                                        lambda: ZeroSumReward(zero_sum=Constants_selector.ZERO_SUM,
                                                              goal_w=5,
                                                              concede_w=-5,
                                                              team_spirit=1,
                                                              punish_action_change_w=0,
                                                              decay_punish_action_change_w=-0.001,
                                                              flip_reset_w=2,
                                                              flip_reset_goal_w=5,
                                                              aerial_goal_w=2,
                                                              double_tap_w=4,
                                                              punish_directional_changes=True,
                                                              cons_air_touches_w=0.4,
                                                              jump_touch_w=0.5,
                                                              wall_touch_w=1,
                                                              exit_velocity_w=1,
                                                              velocity_pb_w=0.005,
                                                              kickoff_w=0.005,
                                                              ),
                                        lambda: parser,
                                        save_every=logger.config.save_every * 3,
                                        model_every=logger.config.model_every,
                                        logger=logger,
                                        clear=False,
                                        stat_trackers=stat_trackers,
                                        # gamemodes=("1v1", "2v2", "3v3"),
                                        max_age=1,
                                        )
    input_size = 426 + Constants_selector.STACK_SIZE
    action_size = 23
    critic = Sequential(Linear(input_size, 256), LeakyReLU(), Linear(256, 256), LeakyReLU(),
                        Linear(256, 256), LeakyReLU(),
                        Linear(256, 1))

    actor = Sequential(Linear(input_size, 256), LeakyReLU(), Linear(256, 256), LeakyReLU(), Linear(256, 128),
                       LeakyReLU(),
                       Linear(128, action_size))

    critic = Opti(embedder=Sequential(Linear(35, 128), LeakyReLU(), Linear(128, 35 * 5)), net=critic)

    actor = Opti(embedder=Sequential(Linear(35, 128), LeakyReLU(), Linear(128, 35 * 5)), net=actor)

    actor = DiscretePolicy(actor, shape=(action_size,))

    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": logger.config.actor_lr},
        {"params": critic.parameters(), "lr": logger.config.critic_lr},
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)
    print(f"Gamma is: {gamma}")
    count_parameters(agent)

    action_dict = {0: "kickoff_1",
                   1: "kickoff_2",
                   2: "GP",
                   3: "aerial",
                   4: "flick_bump",
                   5: "flick",
                   6: "flip_reset_1",
                   7: "flip_reset_2",
                   8: "flip_reset_3",
                   9: "pinch",
                   10: "recover_0",
                   11: "recover_-45",
                   12: "recover_-90",
                   13: "recover_-135",
                   14: "recover_180",
                   15: "recover_135",
                   16: "recover_90",
                   17: "recover_45",
                   18: "recover_back_left",
                   19: "recover_back_right",
                   20: "recover_opponent",
                   21: "recover_back_post",
                   22: "recover_ball",
                   }

    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=logger.config.ent_coef,
        n_steps=logger.config.n_steps,
        batch_size=logger.config.batch_size,
        minibatch_size=logger.config.minibatch_size,
        epochs=logger.config.epochs,
        gamma=logger.config.gamma,
        logger=logger,
        zero_grads_with_none=True,
        disable_gradient_logging=True,
        action_selection_dict=action_dict,
        num_actions=action_size,
    )

    # alg.load("Selector_saves/Opti_1672978698.978992/Opti_1780/checkpoint.pt")
    alg.agent.optimizer.param_groups[0]["lr"] = logger.config.actor_lr
    alg.agent.optimizer.param_groups[1]["lr"] = logger.config.critic_lr

    alg.run(iterations_per_save=logger.config.save_every, save_dir="Selector_saves")
