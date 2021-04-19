import numpy as np
import copy

SNAKE_PART = 1
APPLE_PART = -1

class Snake_Game():
    def __init__(self, width=10, height=10):
        self.width, self.height = width, height
        self.direction = {0 : np.array([-1, 0]),   ## up
                          2 : np.array([ 1, 0]),   ## down
                          3 : np.array([ 0,-1]),   ## left
                          1 : np.array([ 0, 1])}  ## right

    def get_apple(self):
        posx = np.random.randint(0, self.width)
        posy = np.random.randint(0, self.height)

        while self.board[posx][posy] != 0:
            return self.get_apple()
        return np.array([posx, posy])

    def reset(self):
        self.snake_head = np.array([ self.width//2, self.height//2 ])
        self.snake_body = []
        self.snake_direction = 0

        self.board = np.zeros((self.width, self.height))
        self.board[tuple(self.snake_head)] = SNAKE_PART
        self.apple = self.get_apple()
        self.board[tuple(self.apple)] = APPLE_PART
        return self.board, self.snake_head, self.apple, self.snake_body

    def validate_move(self, action):
        temp_pos = self.snake_head + self.direction[action]
        print('HEAD', self.snake_head, 'TEMP', temp_pos)
        
        ## if touch it self
        if self.board[tuple(temp_pos)] == SNAKE_PART:
            return False

        if temp_pos[0] < 0 or temp_pos[0] >= self.width:
            return False
        
        if temp_pos[1] < 0 or temp_pos[1] >= self.height:
            return False

        return True

    def step(self, action):
        reward = 0
        game_end = False

        self.snake_direction = action
        self.snake_body.append(self.snake_head.copy())

        ## Check move
        if not self.validate_move(action): 
            reward = -1
            game_end = True
        else:
            self.snake_head += self.direction[action]
            self.board[tuple(self.snake_head)] = SNAKE_PART

        ## Remove tail - if no apples
        if (self.snake_head[0] == self.apple[0] and self.snake_head[1] == self.apple[1]):
            self.apple = self.get_apple()
            self.board[tuple(self.apple)] = APPLE_PART
            reward = 1
        else:
            self.board[tuple(self.snake_body[0])] = 0
            self.snake_body.remove(self.snake_body[0])
          
        package = (self.board, self.snake_head, self.apple, self.snake_body)
        print(self.board)
        return package, reward, game_end

    def get_action(self):
        pass


game = Snake_Game()
print(game.reset())

while(True):
    action = int(input())
    package, reward, end = game.step(action)
    print('Reward', reward)

