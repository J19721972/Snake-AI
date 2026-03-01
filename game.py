import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np


# Iniciar pygame
pygame.init()

font = pygame.font.SysFont('arial', 20)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Constantes colores RGB
BLACK = (39, 40, 56)
WHITE = (246, 247, 235)
RED   = (232, 67, 71)
GREEN = (106, 141, 115)

BLOCK_SIZE = 500//20
SPEED = 10000000000000

# Clase del juego
class SnakeGameAI:

    def __init__(self, w=1000, h=1000):
        self.w = w
        self.h = h

        # Iniciar la pantalla
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('SnakeAI')
        self.clock = pygame.time.Clock()
        self.reset()

    # Reiniciar estado
    def reset(self):
        
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)

        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    # Colocar manzana aleatoriamente
    def _place_food(self):
        x = random.randint(0, int((self.w - BLOCK_SIZE) // BLOCK_SIZE)) * int(BLOCK_SIZE)
        y = random.randint(0, int((self.h - BLOCK_SIZE) // BLOCK_SIZE)) * int(BLOCK_SIZE)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    # Ejecutar iteracion de juego
    def play_step(self, action):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.frame_iteration += 1
        
        self._move(action)
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        
        # En caso de especificación de punto
        if pt is None:
            pt = self.head

        # En caso de choque con pared o fin de tiempo
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        
        # En caso de choque contra serpiente
        if pt in self.snake[1:]:
            return True

        return False

    # Actualizar interfaz
    def _update_ui(self):
        self.display.fill(BLACK)

        for point in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(point.x+0.5, point.y+0.5, BLOCK_SIZE-1, BLOCK_SIZE-1))
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x+0.5, self.food.y+0.5, BLOCK_SIZE-1, BLOCK_SIZE-1))
        text = font.render("PUNTUACIÓN: " + str(self.score), True, WHITE)
        self.display.blit(text, [5,5])

        pygame.display.flip()


    def _move(self, action):

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            # Seguir recto
            new_dir = clock_wise[index]
        elif np.array_equal(action, [0, 1, 0]):
            # Giro derecha
            next_index = (index + 1) % 4
            new_dir = clock_wise[next_index]
        else: # [0, 0, 1]
            # Giro izquierda
            next_index = (index - 1) % 4
            new_dir = clock_wise[next_index]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        # Actualizar coordenadas
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


        
