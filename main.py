from gameplay.environment import Environment
from DeepQLearning import DQNWrapper

import time
import torch
import json
from datetime import timedelta

def train(config):
    # CREATE ENVIRONMENT
    with open(config['env']) as env_cfg:
        snake_cfg = json.load(env_cfg)
        env = Environment(config=snake_cfg, verbose=1)
    p = DQNWrapper(config['model_HP'], config['training_HP']) # CREATE TRAINING WRAPPED MODEL
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
            state = torch.from_numpy(timestep.observation).float().unsqueeze(dim=0).unsqueeze(dim=0)
            action = p.get_action(state)
            env.choose_action(action)
            timestep = env.timestep()
            # extract stuff
            reward = timestep.reward
            rewards.append(reward)
            state_next = torch.from_numpy(timestep.observation).float().unsqueeze(dim=0).unsqueeze(dim=0)
            game_over = timestep.is_episode_end
            loss = p.train_model(state, action, state_next, reward, game_over)
        if episode % config['console_update_interval']:
            summary = 'Episode {0}/{1} | Loss {2} | Exploration {3} | ' + \
                      'Fruits {4} | Timesteps {5} | Total Reward {6}'
            print(summary.format(
                episode + 1, num_episodes, loss, exploration_rate,
                env.stats.fruits_eaten, env.stats.timesteps_survived, env.stats.sum_episode_rewards,
            ))
            p.save_weights()

config = {
    "env": "levels/10x10-blank.json",
    "model_HP": {
        "model_name": "ConvNetwork",
        "save_name": "ConvNN1",
        "device": 'cuda' if torch.cuda.is_available() else 'cpu'
    },
    "training_HP": {
        "gamma": 0.9,
        "epsilon": 0.1,
        "lr": 1e-4,
        "replayMemorySize": 10000,
        "targetNetUpdate": 10,
        "trainBatchSize": 512
    },
    "others": {
        "epochs": 60000,
        "console_update_interval": 10
    }
}

train(config)
