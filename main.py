# from DeepQLearning import DQNWrapper

import time
import torch
import json
import gym
import random
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
        if episode % config['console_update_interval'] == 0:
            summary = 'Episode {0}/{1} | Loss {2} | Exploration {3} | ' + \
                      'Fruits {4} | Timesteps {5} | Total Reward {6}'
            print(summary.format(
                episode + 1, num_episodes, loss, exploration_rate,
                env.stats.fruits_eaten, env.stats.timesteps_survived, env.stats.sum_episode_rewards,
            ))
            p.save_weights()

env = gym.make('SpaceInvaders-v0')
height, width, channels = env.observation_space.shape
print(height, width, channels)
actions = env.action_space.n
print(actions)

# episodes = 5
# for episode in range(episodes):
#     state = env.reset()
#     done = False
#     score = 0
#     while not done:
#         env.render()
#         action = random.choice(range(actions))
#         n_state, reward, done, info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam

def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3, height, width, channels)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model = build_model(height, width, channels, actions)

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000
                  )
    return dqn

tensorflow.reset_default_graph()
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-4))
dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)
# config = {
#     "env": "levels/5x5-blank.json",
#     "model_HP": {
#         "model_name": "ConvNetwork",
#         "save_name": "ConvNN1",
#         "device": 'cuda:0' if torch.cuda.is_available() else 'cpu'
#     },
#     "training_HP": {
#         "gamma": 0.9,
#         "epsilon": 0.1,
#         "lr": 1e-4,
#         "replayMemorySize": 10000,
#         "targetNetUpdate": 10,
#         "trainBatchSize": 512
#     },
#     "others": {
#         "epochs": 60000,
#         "console_update_interval": 10
#     }
# }
#
# train(config)
