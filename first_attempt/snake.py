import pygame
from pygame.locals import *
import sys
import numpy as np
import random
import math
import copy
import dill

pygame.init()

fps = 1000

display_width = 300
display_height = 300

box_size = 15
boxes_per_row = int(display_width / box_size)
boxes_per_column = int(display_height / box_size)

starting_x = boxes_per_row / 2
starting_y = boxes_per_column / 2

black = (0, 0, 0)
white = (255, 255, 255)
green = (0, 255, 0)
red = (255, 0, 0)

fps_clock = pygame.time.Clock()
display_surface = pygame.display.set_mode((display_width, display_height))

'''
We provide the initial direction of the snake's movement, the angle of rotation and the chance the snake makes a random move
'''
direction = [0, 1]
angle = math.pi / 2
exploration = 0.00001


def main():

    global display_surface, display_height, display_width, direction, angle, exploration

    fruit = Fruit(x=np.random.randint(0, boxes_per_row), y=np.random.randint(0, boxes_per_column))

    # snake = Brain((15, 30, 4), activation_function='sigmoid', cost_function='quadratic', x=starting_x, y=starting_y)

    with open('trained_snake.pkl', 'rb') as f:
        snake = dill.load(f)

    counter = 1

    max_length = snake.size

    while True:

        display_surface.fill((white))

        '''
        The reward is set to 0 at the beginning of each move
        '''
        reward = 0

        '''
        The 4 directions that the snake can take
        '''
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        snake.draw()
        fruit.draw()

        obstacle_down = 0
        obstacle_up = 0
        obstacle_right = 0
        obstacle_left = 0

        edge_right = 0
        edge_left = 0
        edge_up = 0
        edge_down = 0

        food_right = 0
        food_left = 0
        food_up = 0
        food_down = 0

        '''
        We use pixel information to check for the presence of food
        '''
        try:
            if display_surface.get_at_mapped((int((snake.x + 1) * box_size), int((snake.y) * box_size))) == 16711680:
                food_right = 1
        except:
            pass

        try:
            if display_surface.get_at_mapped((int((snake.x - 1) * box_size), int((snake.y) * box_size))) == 16711680:
                food_left = 1
        except:
            pass

        try:
            if display_surface.get_at_mapped((int((snake.x) * box_size), int((snake.y + 1) * box_size))) == 16711680:
                food_down = 1
        except:
            pass

        try:
            if display_surface.get_at_mapped((int((snake.x) * box_size), int((snake.y - 1) * box_size))) == 16711680:
                food_up = 1
        except:
            pass

        if snake.x == boxes_per_row - 1:
            edge_right = 1
        if snake.x == 0:
            edge_left = 1
        if snake.y == 0:
            edge_up = 1
        if snake.y == boxes_per_column - 1:
            edge_down = 1

        '''
        We use pixel information to check for different kinds of obstacles and the snake's own tail
        '''
        try:
            if display_surface.get_at_mapped((int((snake.x + 1) * box_size), int((snake.y) * box_size))) != 16711680 and display_surface.get_at_mapped((int((snake.x + 1) * box_size), int((snake.y) * box_size))) != -256:
                obstacle_right = 1
        except:
            pass

        try:
            if display_surface.get_at_mapped((int((snake.x - 1) * box_size), int((snake.y) * box_size))) != 16711680 and display_surface.get_at_mapped((int((snake.x - 1) * box_size), int((snake.y) * box_size))) != -256:
                obstacle_left = 1
        except:
            pass

        try:
            if display_surface.get_at_mapped((int((snake.x) * box_size), int((snake.y + 1) * box_size))) != 16711680 and display_surface.get_at_mapped((int((snake.x) * box_size), int((snake.y + 1) * box_size))) != -256:
                obstacle_down = 1
        except:
            pass

        try:
            if display_surface.get_at_mapped((int(snake.x * box_size), int((snake.y - 1) * box_size + 1))) != 16711680 and display_surface.get_at_mapped((int((snake.x) * box_size), int((snake.y - 1) * box_size))) != -256:
                obstacle_up = 1
        except:
            pass

        relative_x = snake.x / boxes_per_row
        relative_y = snake.y / boxes_per_column

        '''
        The snake takes the following state of the world into account: its relative x position, its relative y position, the angle between the snake and the food; if there is an obstacle right below it, if there is an obstacle above, obstacle to the right, obstacle to the left; if there is an edge below, up, to the right or to the left; if there is food to the right, to the left, up or down. It cheks only 1 step in each direction => limitations
        '''
        current_state = [relative_x, relative_y, math.atan2(snake.x - fruit.x, snake.y - fruit.y), obstacle_down, obstacle_up, obstacle_right, obstacle_left, edge_down, edge_up, edge_right, edge_left, food_right, food_left, food_up, food_down]

        '''
        We propagate the information about the current state of the world through the net and get the Q-values associated with a move in each direction
        '''
        q_values = snake.propagate_forward(current_state)

        if np.random.uniform(0, 1) < exploration:
            direction = random.choice(directions)
        else:
            if np.argmax(q_values) == 0:
                direction = directions[0]
            elif np.argmax(q_values) == 1:
                direction = directions[1]
            elif np.argmax(q_values) == 2:
                direction = directions[2]
            else:
                direction = directions[3]

        '''
        We incorporate information about the distance to food before and after making a step to encourage the snake's behaviour
        '''
        initial_distance = math.sqrt((fruit.x - snake.x) ** 2 + (fruit.y - snake.y) ** 2)

        snake.move()

        distance_after = math.sqrt((fruit.x - snake.x) ** 2 + (fruit.y - snake.y) ** 2)

        '''
        Below we do reward assignments for each step a snake makes depending on where it ends up. We penalise the snake heavily for dying to encourage it to survive for longer. However, due to the limited info about the state, it still does not manage well.
        '''
        if distance_after > initial_distance:
            reward = -1
        else:
            reward = 0.5

        if snake.x == fruit.x and snake.y == fruit.y:
            snake.size += 1
            lala = (snake.x, snake.y)
            snake.tail.append(lala)
            fruit.x = np.random.randint(0, boxes_per_row)
            fruit.y = np.random.randint(0, boxes_per_column)
            '''
            Below piece of code is required to make sure that the fruit does not appear somewhere on the snake
            '''
            while fruit.check_position(snake):
                fruit.x = np.random.randint(0, boxes_per_row)
                fruit.y = np.random.randint(0, boxes_per_column)
            reward = 7

        if snake.x < 0 or snake.x > boxes_per_row - 1 or snake.y < 0 or snake.y > boxes_per_column - 1:
            snake.x = starting_x
            snake.y = starting_y
            snake.tail = []
            snake.size = 0
            reward = -10

        if direction == [1, 0] and obstacle_right == 1:
            snake.x = starting_x
            snake.y = starting_y
            reward = -15
            snake.tail = []
            snake.size = 0
        elif direction == [-1, 0] and obstacle_left == 1:
            snake.x = starting_x
            snake.y = starting_y
            reward = -15
            snake.tail = []
            snake.size = 0
        elif direction == [0, 1] and obstacle_down == 1:
            snake.x = starting_x
            snake.y = starting_y
            reward = -15
            snake.tail = []
            snake.size = 0
        elif direction == [0, -1] and obstacle_up == 1:
            snake.x = starting_x
            snake.y = starting_y
            reward = -15
            snake.tail = []
            snake.size = 0

        display_surface.fill((white))
        snake.draw()
        fruit.draw()

        '''
        Once we have made a move and drawn the resulting state of the world, we again gather information about the world
        '''

        obstacle_down_next = 0
        obstacle_up_next = 0
        obstacle_right_next = 0
        obstacle_next_left = 0

        edge_right_next = 0
        edge_left_next = 0
        edge_up_next = 0
        edge_down_next = 0

        food_right_next = 0
        food_left_next = 0
        food_up_next = 0
        food_down_next = 0

        if snake.x == boxes_per_row - 1:
            edge_right_next = 1
        if snake.x == 0:
            edge_left_next = 1
        if snake.y == 0:
            edge_up_next = 1
        if snake.y == boxes_per_column - 1:
            edge_down_next = 1

        try:
            if display_surface.get_at_mapped((int((snake.x + 1) * box_size), int((snake.y) * box_size))) != 16711680 and display_surface.get_at_mapped((int((snake.x + 1) * box_size), int((snake.y) * box_size))) != -256:
                obstacle_right_next = 1
        except:
            pass

        try:
            if display_surface.get_at_mapped((int((snake.x - 1) * box_size), int((snake.y) * box_size))) != 16711680 and display_surface.get_at_mapped((int((snake.x - 1) * box_size), int((snake.y) * box_size))) != -256:
                obstacle_left_next = 1
        except:
            pass

        try:
            if display_surface.get_at_mapped((int((snake.x) * box_size), int((snake.y + 1) * box_size))) != 16711680 and display_surface.get_at_mapped((int((snake.x) * box_size), int((snake.y + 1) * box_size))) != -256:
                obstacle_down_next = 1
        except:
            pass

        try:
            if display_surface.get_at_mapped((int((snake.x) * box_size), int((snake.y - 1) * box_size))) != 16711680 and display_surface.get_at_mapped((int((snake.x) * box_size), int((snake.y - 1) * box_size))) != -256:
                obstacle_up_next = 1
        except:
            pass

        try:
            if display_surface.get_at_mapped((int((snake.x + 1) * box_size), int((snake.y) * box_size))) == 16711680:
                food_right_next = 1
        except:
            pass

        try:
            if display_surface.get_at_mapped((int((snake.x - 1) * box_size), int((snake.y) * box_size))) == 16711680:
                food_left_next = 1
        except:
            pass

        try:
            if display_surface.get_at_mapped((int((snake.x) * box_size), int((snake.y + 1) * box_size))) == 16711680:
                food_down_next = 1
        except:
            pass

        try:
            if display_surface.get_at_mapped((int((snake.x) * box_size), int((snake.y - 1) * box_size))) == 16711680:
                food_up_next = 1
        except:
            pass

        relative_x_next = snake.x / boxes_per_row
        relative_y_next = snake.y / boxes_per_column

        next_state = [relative_x_next, relative_y_next, math.atan2(snake.x - fruit.x, snake.y - fruit.y), obstacle_down_next, obstacle_up_next, obstacle_right_next, obstacle_next_left, edge_down_next, edge_up_next, edge_right_next, edge_left_next, food_right_next, food_left_next, food_up_next, food_down_next]

        '''
        Below is the essence of the snake's learning.
        '''
        next_q_value = max(snake.propagate_forward(next_state)) + reward
        target = copy.deepcopy(q_values)
        target[np.argmax(q_values)] = next_q_value

        snake.propagate_forward(current_state)

        snake.propagate_backward(target)

        counter += 1

        if max_length < snake.size:
            max_length = snake.size

        '''
        Every 10,000 pixels we save the progress
        '''
        # if counter % 10000 == 0:
        #     snake.save_as('trained_snake')

        '''
        Below are some checks
        '''
        print(max_length)
        print(q_values)
        print(target)
        # print(obstacle_right)
        # print(obstacle_left)
        # print(obstacle_down)
        # print(obstacle_up)
        # print(edge_right)
        # print(edge_left)
        # print(edge_up)
        # print(edge_down)
        # print(food_right)
        # print(food_left)
        # print(food_up)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    direction = directions[2]
                elif event.key == pygame.K_LEFT:
                    direction = directions[3]
                elif event.key == pygame.K_DOWN:
                    direction = directions[0]
                elif event.key == pygame.K_UP:
                    direction = directions[1]

        pygame.display.flip()
        fps_clock.tick(fps)


class Fruit:

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def draw(self):
        pygame.draw.rect(display_surface, green, (self.x * box_size, self.y * box_size, box_size, box_size))

    def check_position(self, obj):
        for i in range(len(obj.tail)):
            if (self.x, self.y) == obj.tail[i]:
                return True


# def get_pixels():
#     pixels = []
#     for i in range(display_width):
#         for j in range(display_height):
#             pixels.append(display_surface.get_at_mapped((i, j)))
#     return pixels


def get_pixels():
    pixels = []
    for i in range(boxes_per_row):
        for j in range(boxes_per_column):
            if display_surface.get_at_mapped((i * box_size + 2, j * box_size + 2)) == -256:
                pixels.append(0)
            elif display_surface.get_at_mapped((i * box_size + 2, j * box_size + 2)) == 0:
                pixels.append(-1)
            elif display_surface.get_at_mapped((i * box_size + 2, j * box_size + 2)) == 16711680:
                pixels.append(1)
            else:
                raise ValueError('Wrong pixel info')
    return pixels


class Brain(object):

    def __init__(self, structure, activation_function='relu', cost_function='quadratic', x=None, y=None, color=None):
        '''
        The structural (defining) characteristics of a NN
        '''
        self.structure = structure
        self.L = len(structure)
        self.T = len(structure) - 1
        self.K = structure[-1]
        self.n = structure[0]
        self.biases = [np.random.randn(x, 1) for x in self.structure[1:]]
        self.weights = None
        self.weights_initializer()
        self.activation_function = activation_function
        self.cost_function = cost_function

        '''
        Data for learning
        '''
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.m = None
        self.m_training = None
        self.m_validation = None
        self.m_test = None

        '''
        Variables for learning
        '''
        self.learning_rate = 0.01
        self.regularization_parameter = 0
        self.batch_size = None
        self.number_of_epochs = None

        '''
        Forward and backpropagation variables
        '''
        self.z = None
        self.a = None
        self.errors = None
        self.gradients = None

        '''
        On validation data
        '''
        self.cost = None
        self.accuracy = None

        '''
        On test data
        '''
        self.expected_cost = None
        self.expected_accuracy = None

        '''
        For reinforcement learning problems
        '''
        self.state = 'alive'
        self.fitness = 0
        self.x = x
        self.y = y
        self.color = color
        self.tail = []
        self.size = 0

    def draw(self):
        pygame.draw.rect(display_surface, red, (self.x * box_size, self.y * box_size, box_size, box_size))

        for i in range(len(self.tail)):
            pygame.draw.rect(display_surface, black, (self.tail[-(i + 1)][0] * box_size, self.tail[-(i + 1)][1] * box_size, box_size, box_size))

    def move(self):
        location = (self.x, self.y)
        self.tail.append(location)
        self.tail.pop(0)
        self.x = self.x + direction[0]
        self.y = self.y + direction[1]

    def propagate_forward(self, x):
        """
        If the input is a simple python list of observations, we transform it to numpy array type. If it is already a numpy array type, there will be no change.
        """
        length = len(x)

        assert length == self.structure[0], 'ERROR: the length of the input list is not equal to the number of input neurons'

        x = np.reshape(x, (length, 1)).astype(float)

        self.z = []
        self.a = [x]

        for j in range(self.T):
            self.z.append(np.matmul(self.weights[j], self.a[j]) + self.biases[j])
            self.a.append(self.activate(self.z[j]))

        return self.a[-1]

    def propagate_backward(self, y):

        length = len(y)

        y = np.reshape(y, (length, 1)).astype(float)

        self.errors = []
        self.gradients = []

        if self.cost_function == 'quadratic':
            activation_gradient = self.a[-1] - y
            self.errors.append(np.multiply(activation_gradient, self.estimate_derivative(self.z[-1])))
        elif self.cost_function == 'entropic':
            self.errors.append(self.a[-1] - y)

        for j in range(self.T - 1):
            self.errors.append(np.multiply(np.matmul(np.transpose(self.weights[-1 - j]), self.errors[j]), self.estimate_derivative(self.z[-2 - j])))

        for j in range(self.T):
            self.gradients.append(np.matmul(self.errors[j], np.transpose(self.a[-2 - j])))

        for j in range(self.T):
            self.biases[j] = self.biases[j] - self.learning_rate * self.errors[-1 - j]
            self.weights[j] = self.weights[j] - self.learning_rate * self.gradients[-1 - j]

    def estimate_accuracy(self, data):

        length = len(data)

        number_of_correct = 0

        for i in range(length):
            if np.argmax(self.propagate_forward(data[i][0])) == np.argmax(data[i][1]):
                number_of_correct += 1

        return number_of_correct * 100 / length

    def estimate_cost(self, data):

        length = len(data)

        cost = np.array(0)

        if self.cost_function == 'quadratic':
            for i in range(length):
                x = self.propagate_forward(data[i][0])
                y = data[i][-1]
                cost = cost + self.estimate_quadratic_cost(x, y)
                return cost
        elif self.cost_function == 'entropic':
            for i in range(length):
                x = self.propagate_forward(data[i][0])
                y = data[i][-1]
                cost = cost + self.estimate_entropic_cost(x, y)
                return cost

    def learn(self, training_data, validation_data, test_data, learning_rate=0.01, regularization_parameter=0.01, batch_size=10, number_of_epochs=10, save='off'):

        assert len(training_data) != 0, 'Please, provide training data'
        assert len(validation_data) != 0, 'Please, provide validation data'
        assert len(test_data) != 0, 'Please, provide test data'

        self.training_data = list(training_data)
        self.validation_data = list(validation_data)
        self.test_data = list(test_data)

        self.learning_rate = learning_rate
        self.regularization_parameter = regularization_parameter
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs

        self.m_training = len(self.training_data)
        self.m_validation = len(self.validation_data)
        self.m_test = len(self.test_data)
        self.m = self.m_training + self.m_validation + self.m_test

        print('Training data size is %s' % (self.m_training))
        print('Validation data size is %s' % (len(self.validation_data)))
        print('Test data size is %s' % (len(self.test_data)))

        highest_accuracy = 0

        for i in range(self.number_of_epochs):
            np.random.shuffle(self.training_data)
            batches = [self.training_data[k: k + self.batch_size] for k in range(0, self.m_training, self.batch_size)]
            for batch in batches:
                self.update_batch(batch)

            self.accuracy = self.estimate_accuracy(self.validation_data)
            self.cost = self.estimate_cost(self.validation_data)

            print('Epoch number %s is over. Accuracy on validation set is %s percent, cost on validation set is %s ' % (i + 1, self.accuracy, self.cost))

            if save == 'on':
                if self.accuracy > highest_accuracy:
                    self.expected_accuracy = self.estimate_accuracy(self.test_data)
                    self.expected_cost = self.estimate_cost(self.test_data)
                    self.save_as('trained_brain')
                    highest_accuracy = self.accuracy

        if save == 'on':
            print('The learning is over. The best-performing brain was saved. The expected accuracy on unseen data is %s. The expected cost on unseen data is %s' % (self.expected_accuracy, self.expected_cost))
        else:
            self.expected_accuracy = self.estimate_accuracy(self.test_data)
            print('The learning is over. The expected accuracy on unseen data is %s. The expected cost on unseen data is %s' % (self.expected_accuracy, self.expected_cost))

    def update_batch(self, data):

        batch_length = len(data)

        self.z = [[] for i in range(batch_length)]
        self.a = [[data[i][0]] for i in range(batch_length)]

        self.gradients = [[] for i in range(batch_length)]
        self.errors = [[] for i in range(batch_length)]

        for i in range(batch_length):
            for j in range(self.T):
                self.z[i].append(np.matmul(self.weights[j], self.a[i][j]) + self.biases[j])
                self.a[i].append(self.activate(self.z[i][j]))

        activation_gradient = []

        for i in range(batch_length):
            if self.cost_function == 'quadratic':
                activation_gradient.append(self.a[i][-1] - data[i][-1])
                self.errors[i].append(np.multiply(activation_gradient[i], self.estimate_derivative(self.z[i][-1])))
            elif self.cost_function == 'entropic':
                self.errors[i].append(self.a[i][-1] - data[i][-1])

        for i in range(batch_length):
            for j in range(self.T - 1):
                self.errors[i].append(np.multiply(np.matmul(np.transpose(self.weights[-1 - j]), self.errors[i][j]), self.estimate_derivative(self.z[i][-2 - j])))

        for i in range(batch_length):
            for j in range(self.T):
                self.gradients[i].append(np.matmul(self.errors[i][j], np.transpose(self.a[i][-2 - j])))

        for j in range(self.T):
            sum_of_biases = 0
            sum_of_weights = 0
            for i in range(batch_length):
                sum_of_biases = sum_of_biases + self.errors[i][-1 - j]
                sum_of_weights = sum_of_weights + self.gradients[i][-1 - j]
            self.biases[j] = self.biases[j] - self.learning_rate / batch_length * sum_of_biases
            self.weights[j] = self.weights[j] * (1 - self.learning_rate * self.regularization_parameter / batch_length) - self.learning_rate / batch_length * sum_of_weights

    def update_structure(self, structure):
        """
        This method allows us to update the structure of the brain within the brains itself. We re-initialize the weights and biases (and all the linked parameters of the brain)
        """
        assert type(structure) == tuple, 'ERROR: structure should be type tuple'
        self.structure = structure
        self.L = len(structure)
        self.T = len(structure) - 1
        self.K = structure[-1]
        self.n = structure[0]
        self.biases = [np.random.randn(x, 1) for x in self.structure[1:]]
        self.weights_initializer()

    def restart(self):
        """
        The below allows the network to 'forget' everything it learned previously
        """
        self.biases = [np.random.randn(x, 1) for x in self.structure[1:]]
        self.weights_initializer()

    def weights_initializer(self):
        """
        The below code initializes the weights based on the number of newurons in each respective layer
        """
        self.weights = [np.random.normal(0, 1 / np.sqrt(x), (x, y)) for x, y in list(zip(self.structure[1:], self.structure[:-1]))]

    def save_as(self, filename):
        """
        Allows us to save a trained Brain object to file for later use
        """
        assert type(filename) == str, 'ERROR: filename should be type str'
        if '.pkl' in filename:
            with open(filename, 'wb') as f:
                dill.dump(self, f)
        else:
            with open(filename + '.pkl', 'wb') as f:
                dill.dump(self, f)

    def estimate_quadratic_cost(self, x, y):
        return np.sum(np.power(y - x, 2) / 2)

    def estimate_entropic_cost(self, x, y):
        return np.sum(-y * np.log(x) - (1 - y) * np.log(1 - x))

    def activate(self, x):
        if self.activation_function == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_function == 'relu':
            return self.relu(x)
        elif self.activation_function == 'tanh':
            return self.tanh(x)

    def estimate_derivative(self, x):
        if self.activation_function == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation_function == 'relu':
            return self.relu_derivative(x)
        elif self.activation_function == 'tanh':
            return self.tanh_derivative(x)

    def sigmoid(self, x):
        return np.divide(1, (1 + np.exp(-x)))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


if __name__ == '__main__':
    main()
