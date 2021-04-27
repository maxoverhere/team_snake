from env.gameplay.environment import Environment
from DeepQLearning import DQNWrapper
from torch.utils.tensorboard import SummaryWriter
import gym

import time
import random
import torch
import json
from datetime import timedelta
import numpy as np

def process(state):
    p_state = np.array([np.zeros_like(state)]*4)
    p_state[0][np.where(state == 4)] = 100
    p_state[1][np.where(state == 1)] = 100
    p_state[2][np.where(state == 2)] = 100
    p_state[3][np.where(state == 3)] = 100
    return p_state

def train(config):
    # CREATE ENVIRONMENT
    with open(config['env']) as env_cfg:
        snake_cfg = json.load(env_cfg)
        env = Environment(config=snake_cfg, verbose=1)
    p = DQNWrapper(config['model_HP'], config['training_HP']) # CREATE TRAINING WRAPPED MODEL
    name = config["model_HP"]["model_name"]
    config = config['others']
    # Some trackers
    rewards = []
    num_episodes = config['epochs']
    for episode in range(num_episodes):
        timestep = env.new_episode()
        game_over = False
        loss = 0.0
        exploration_rate = 0.1
        while not game_over:
            state = torch.from_numpy(process(timestep.observation)).float().unsqueeze(dim=0)
            action = p.get_action(state)
            print(timestep.observation)
            print("Agent chosen", {0:"UP",1:"RIGHT",2:"DOWN",3:"LEFT"}[action])
            env.choose_action(action)
            timestep = env.timestep()
            # extract stuff
            reward = timestep.reward
            rewards.append(reward)
            state_next = torch.from_numpy(process(timestep.observation)).float().unsqueeze(dim=0)
            game_over = timestep.is_episode_end
            loss = p.train_model(state, action, state_next, reward, game_over)
        if episode % config['console_update_interval'] == 0:
            summary = 'Episode {0}/{1} | Loss {2} | Exploration {3} | ' + \
                      'Fruits {4} | Timesteps {5} | Total Reward {6}'
            print(summary.format(
                episode + 1, num_episodes, loss, exploration_rate,
                env.stats.fruits_eaten, env.stats.timesteps_survived, env.stats.sum_episode_rewards,
            ))
            p.save_weights()
        t_board = SummaryWriter("runs/{0}".format(name))
        t_board.add_scalar('Test/FruitsEaten', env.stats.fruits_eaten, global_step = episode)
        t_board.add_scalar('Test/TimestepsSurvived', env.stats.timesteps_survived, global_step = episode)
        t_board.add_scalar('Test/SumReward', env.stats.sum_episode_rewards, global_step = episode)
        if loss != -1:
            t_board.add_scalar('Test/Loss', loss, global_step = episode)


config = {
    "env": "levels/6x6-blank.json",
    "model_HP": {
        "model_name": "ConvNetworkv21",
        "save_name": "ConvNN1v21",
        "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
        "is_dueling": False,
        "is_doubleL": False
    },
    "training_HP": {
        "gamma": 0.9,
        "epsilon": 0.1,
        "lr": 1e-4,
        "replayMemorySize": 1000,
        "targetNetUpdate": 10,
        "trainBatchSize": 512
    },
    "others": {
        "epochs": 60000,
        "console_update_interval": 1
    }
}

# train(config)
