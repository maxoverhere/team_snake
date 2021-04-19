from dqn import DQN
from neural_link import NeuralLink
from snake_game import Snake_Game

import time
import torch
from datetime import timedelta

device = 'cuda' if torch.cuda.is_available() else 'cpu'

g = Snake_Game()
p = DQN()

def process(raw_state):
    board, snake_head, apple, snake_body = raw_state
    state = torch.zeros(board.shape, device=device, dtype=torch.float).unsqueeze(0).repeat(3, 1, 1)
    state[0][tuple(apple)] = 1
    state[1][tuple(snake_head)] = 1
    state[2][[tuple(x) for x in snake_body]] = 1
    return board, state

def train(epochs=5000, update_interval=500):
    stime = time.time()
    steps, n_apple_count = 0, 0
    for epoch in range(1, epochs+1):
        state = g.reset()
        board, state = process(state)
        state = state.unsqueeze(0)
        game_end = False
        while not (game_end):
            action = p.get_action(state)
            next_state, reward, game_end = g.step(action.item())
            apple_count = len(next_state[3])
            board, next_state = process(next_state)
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

train(update_interval=10)
