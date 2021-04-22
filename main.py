from DeepQLearning import DQN
from LinearNetwork import LinearModel

from snake_game import Snake_Game

import time
import torch
from datetime import timedelta

def train(epochs=5000, update_interval=500):
    BOARD_SIZE = 40
    g = Snake_Game(width=BOARD_SIZE, height=BOARD_SIZE)
    p = DQN(model_name="default2")
    stime = time.time()
    steps, n_apple_count = 0, 0
    for epoch in range(1, epochs+1):
        state = g.reset()
        board, state = p.process(state)
        state = state.unsqueeze(0)
        game_end = False
        while not (game_end):
            action = p.get_action(state)
            next_state, reward, game_end = g.step(action.item())
            apple_count = len(next_state[3])
            board, next_state = p.process(next_state)
            next_state = next_state.unsqueeze(0)
            # def train_model(self, n_batch, state, action, next_state, reward, end_game):
            p.train_model(100, state, action, next_state, reward, game_end)
            state = next_state
            steps += 1
        n_apple_count += apple_count
        if epoch % update_interval == 0:
            print("Elapsed time: {0} Epoch: {1}/{2} Average game steps: {3}, Average apples: {4}"
                .format(str(timedelta(seconds=(time.time() - stime) )), str(epoch), str(epochs), steps/update_interval, n_apple_count/update_interval))
            p.save_weights()
            stime = time.time()
            steps, n_apple_count = 0, 0

train(update_interval=1000)
