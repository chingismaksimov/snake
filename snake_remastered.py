import numpy as np
import pygame
from pygame.locals import *
import sys

pygame.init()

display_width = 400
display_height = 400
block_size = 20

fps = 100
clock = pygame.time.Clock()
display = pygame.display.set_mode((display_width, display_height))

white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)

x_initial = np.random.randint(5, 15)
y_initial = np.random.randint(5, 15)


def game():

    snake = [(x_initial, y_initial)]
    fruit = (np.random.randint(0, int(display_width / block_size)), np.random.randint(0, int(display_height / block_size)))

    while True:

        display.fill(white)

        for body_part in snake:
            draw_rect(body_part, red)
        draw_rect(fruit, green)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    snake = move_snake(snake, 'right')
                elif event.key == pygame.K_LEFT:
                    snake = move_snake(snake, 'left')
                elif event.key == pygame.K_DOWN:
                    snake = move_snake(snake, 'down')
                elif event.key == pygame.K_UP:
                    snake = move_snake(snake, 'up')

        pygame.display.flip()
        clock.tick(fps)


def draw_rect(pos, color):
    pygame.draw.rect(display, color, (pos[0] * block_size, pos[1] * block_size, block_size, block_size))


def move_snake(snake, direction):
    if direction == 'right' and snake[0][0] < 19:
        snake = [(snake[0][0] + 1, snake[0][1])] + snake
        snake.pop(-1)
    elif direction == 'left' and snake[0][0] > 0:
        snake = [(snake[0][0] - 1, snake[0][1])] + snake
        snake.pop(-1)
    elif direction == 'up' and snake[0][1] > 0:
        snake = [(snake[0][0], snake[0][1] - 1)] + snake
        snake.pop(-1)
    elif direction == 'down' and snake[0][1] < 19:
        snake = [(snake[0][0], snake[0][1] + 1)] + snake
        snake.pop(-1)
    return snake


if __name__ == '__main__':
    game()
