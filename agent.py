import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_partidas = 0
        self.epsilon = 0 # Aleatoriedad
        self.gamma = 0.9 # Importancia de recompensas immediatas
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 128, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    # Definir el estado del juego
    def get_state(self, game):
        
        head = game.snake[0]
        point_l = Point(head.x - 25, head.y)
        point_r = Point(head.x + 25, head.y)
        point_u = Point(head.x, head.y - 25)
        point_d = Point(head.x, head.y + 25)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Peligro recto
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Peligro derecha
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Peligro izquierda
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Dirección de movimiento
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Posición de la comida 
            game.food.x < game.head.x,  # Comida izquierda
            game.food.x > game.head.x,  # Comida derecha
            game.food.y < game.head.y,  # Comida arriba
            game.food.y > game.head.y   # Comida abajo
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Borra memoria más antigua si se sobrepasa MAX_MEMORY
        self.memory.append((state, action, reward, next_state, done)) 

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):

        # Aplicación del algoritmo epsilon-greedy 
        
        self.epsilon = max(0, 80 - self.n_partidas)
        moves = [[1,0,0], [0,1,0], [0,0,1]]
        final_move = [0,0,0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move = moves[move]
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move = moves[move]

        return final_move

# Función de entrenamiento
def train():
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True:

        # Determinar estado
        state = agent.get_state(game)

        # Determinar movimiento
        final_move = agent.get_action(state)
        
        # Hacer movimiento y conseguir nuevo estado
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Entrenar memoria a corto plazo
        agent.train_short_memory(state, final_move, reward, state_new, done)

        # Recordar movimiento con resultados
        agent.remember(state, final_move, reward, state_new, done)

        # Fin del episodio
        if done:
            # Entrenar memoria a largo plazo
            game.reset()
            agent.n_partidas += 1
            agent.train_long_memory()

            if score > record:
                record = score

            print('Partida:', agent.n_partidas, 'Puntuación:', score, 'Record:', record)

if __name__ == '__main__':
    train()