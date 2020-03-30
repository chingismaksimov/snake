import sys
import pygame
import numpy as np
import random
from pygame.locals import *
import dill
import copy
import vectors_module
import math

pygame.init()

fps = 60


'''
Set dispaly dimensions
'''
displayWidth = 625
displayHeight = 625

'''
Set colors
'''
white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)

'''
Set dimensions for the boxes in which snakes are developing
'''
boxWidth = 125
boxHeight = 125

'''
Check that we can fit all the boxes within the dimensions of the display
'''
assert displayWidth % boxWidth == 0
assert displayHeight % boxHeight == 0

'''
Introduce game variables
'''
# boxesPerRow = int(displayWidth / boxWidth)
boxesPerRow = 80
boxesPerColumn = int(displayHeight / boxHeight)
objectSize = 5
angle = math.pi / 2

'''
Set the structure of the brains of our snakes
'''
brainStructure = (5, 3)

display = pygame.display.set_mode((displayWidth, displayHeight))
clock = pygame.time.Clock()
pygame.display.set_caption('Snake')

'''
We create a list of boxex and a list of corners, where corners are the top left corners of the boxes. Boxes are rectangualar objects. This will be needed later when we check if a given snake is still within its box (i.e. if a box contains a snake object)
'''
boxes = []
corners = []


for i in range(boxesPerColumn):
    for j in range(boxesPerRow):
        corners.append((boxWidth * j, boxHeight * i))
        boxes.append(Rect(boxWidth * j, boxHeight * i, boxWidth, boxHeight))
# print('The corners are %s' % (corners))
# print('The boxes are %s' % (boxes))


'''
Below we are creating a list with corners of the squares that are within the first box
'''
cornersInFirstBox = []
for i in range(len(corners)):
    cornersInFirstBox.append((corners[i][0] / objectSize, corners[i][1] / objectSize))
# print('The corners of the squares that are within the first box are %s' % (cornersInFirstBox))

counter = 0


def main():
    '''
    Below we are initializing the staring positions of the Fruit objects. We randomly select a corner of the squares in the first box and then transform them to pixel values
    '''

    global counter

    fruits = [Fruit(random.randint(cornersInFirstBox[i][0] + 1, cornersInFirstBox[i][0] + 23), random.randint(cornersInFirstBox[i][1] + 1, cornersInFirstBox[i][1] + 23)) for i in range(len(cornersInFirstBox))]

    population = Population(boxesPerRow * boxesPerColumn, brainStructure)

    print('Population size is %s' % (population.size))

    for i in range(population.size):
        population.members[i].x = random.randint(cornersInFirstBox[i][0] + 1, cornersInFirstBox[i][0] + 23)
        population.members[i].y = random.randint(cornersInFirstBox[i][1] + 1, cornersInFirstBox[i][1] + 23)

    while True:

        display.fill(white)

        for i in range(len(boxes)):
            pygame.draw.rect(display, black, boxes[i], 1)

        for i in range(len(fruits)):
            fruits[i].draw()
            if fruits[i].x == population.members[i].x and fruits[i].y == population.members[i].y:
                fruits[i].x = random.randint(cornersInFirstBox[i][0] + 1, cornersInFirstBox[i][0] + 23)
                fruits[i].y = random.randint(cornersInFirstBox[i][1] + 1, cornersInFirstBox[i][1] + 23)

        for i in range(population.size):
            population.members[i].move(fruits[i], corners[i])

        population.draw()

        for i in range(population.size):
            population.members[i].eat(fruits[i])

        population.check_state()

        if population.state == 'dead':
            counter = 0
            population.evolve(probability=0.15)
            for i in range(population.size):
                population.members[i].x = random.randint(cornersInFirstBox[i][0] + 1, cornersInFirstBox[i][0] + 23)
                population.members[i].y = random.randint(cornersInFirstBox[i][1] + 1, cornersInFirstBox[i][1] + 23)
                population.members[i].tail = []
                population.members[i].size = 1
                population.members[i].fitness = 1

        if counter > 600:
            counter = 0
            population.state = 'dead'
            population.evolve(probability=0.15)
            for i in range(population.size):
                population.members[i].x = random.randint(cornersInFirstBox[i][0] + 1, cornersInFirstBox[i][0] + 23)
                population.members[i].y = random.randint(cornersInFirstBox[i][1] + 1, cornersInFirstBox[i][1] + 23)
                population.members[i].tail = []
                population.members[i].size = 1
                population.members[i].fitness = 1

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == KEYDOWN:
                if event.key == pygame.K_SPACE:
                    counter = 0
                    population.state = 'dead'
                    population.evolve(probability=0.15)
                    for i in range(population.size):
                        population.members[i].x = random.randint(cornersInFirstBox[i][0] + 1, cornersInFirstBox[i][0] + 23)
                        population.members[i].y = random.randint(cornersInFirstBox[i][1] + 1, cornersInFirstBox[i][1] + 23)
                        population.members[i].tail = []
                        population.members[i].size = 1
                        population.members[i].fitness = 1

        pygame.display.flip()
        clock.tick(fps)

        counter += 1


def squareToXPix(x, squareSize):
    return (x * squareSize)


def squareToYPix(y, squareSize):
    return (y * squareSize)


def estimate_distance(obj1, obj2):
    dx = obj2.x - obj1.x
    dy = obj2.y - obj1.y
    return math.sqrt(dx ** 2 + dy ** 2)


def estimate_angle(obj1, obj2):
    dx = obj2.x - obj1.x
    dy = obj2.y - obj1.y
    return math.atan2(dy, dx)


def check_if_inside(x, y, some_list):
    check1 = squareToXPix(x, objectSize) > some_list[0]
    check2 = squareToXPix(x, objectSize) + objectSize < some_list[0] + boxWidth
    check3 = squareToYPix(y, objectSize) > some_list[1]
    check4 = squareToYPix(y, objectSize) + objectSize < some_list[1] + boxHeight

    if not (check1 and check2 and check3 and check4):
        return False
    else:
        return True


class Fruit:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        pygame.draw.rect(display, green, (squareToXPix(self.x, objectSize), squareToYPix(self.y, objectSize), objectSize, objectSize))


class Brain(object):

    def __init__(self, structure, activation_function='sigmoid', x=None, y=None, color=None):
        """
        Below are the values that we want our brain to keep. Also there are some additional characteristics that can be useful for gaming: x and y positions, color, state
        """
        self.structure = structure
        self.number_of_layers = len(self.structure)
        self.number_of_transitions = self.number_of_layers - 1
        self.biases = [np.random.randn(x, 1) for x in self.structure[1:]]
        self.weights = None
        self.weights_initializer()
        self.activation_function = activation_function
        self.fitness = 1
        self.state = 'alive'
        self.output = None
        self.x = x
        self.y = y
        self.direction_x = -1
        self.direction_y = 0
        self.color = color
        self.tail = []
        self.size = 1

    def predict(self, data):
        """
        If the input is a simple python list of observations, we transform it to numpy array type. If it is already a numpy array type, there will be no change.
        """
        length = len(data)

        """
        We check if the size of the data is equal to the number of input neurons
        """
        assert length == self.structure[0], 'ERROR: the length of the input list is not equal to the number of input neurons'

        data = np.reshape(data, (length, 1)).astype(float)

        # print(type(data))

        """
        We loop over all the transitions between the layers of our brain
        """
        for i in range(self.number_of_transitions):
            if self.activation_function == 'sigmoid':
                data = self.sigmoid(np.dot(self.weights[i], data) + self.biases[i])
            elif self.activation_function == 'ReLU':
                data = self.ReLU(np.dot(self.weights[i], data) + self.biases[i])
            elif self.activation_function == 'tanh':
                data = self.tanh(np.dot(self.weights[i], data) + self.biases[i])

        """
        We allow our brain to store the last prediction. This might be helpful for printing it out on the screen for the user to investigate
        """
        self.output = data

        return data

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

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    """
    Below we are introducing some functions related to neuro-evolution
    """

    def draw(self):
        """
        This function is handy in pygame when we are drawing our object on the screen. Should be modified depending on the situation
        """
        if self.state == 'alive':
            for i in range(len(self.tail)):
                pygame.draw.rect(display, black, (squareToXPix(self.tail[-(i + 1)][0], objectSize), squareToYPix(self.tail[-(i + 1)][1], objectSize), objectSize, objectSize))

            pygame.draw.rect(display, black, (squareToXPix(self.x, objectSize), squareToYPix(self.y, objectSize), objectSize, objectSize))

        else:
            for i in range(len(self.tail)):
                pygame.draw.rect(display, red, (squareToXPix(self.tail[-(i + 1)][0], objectSize), squareToYPix(self.tail[-(i + 1)][1], objectSize), objectSize, objectSize))

            pygame.draw.rect(display, red, (squareToXPix(self.x, objectSize), squareToYPix(self.y, objectSize), objectSize, objectSize))

    def move(self, fruit, corner):
        """
        This function is handy in pygame when we are drawing our object on the screen. Should be modified depending on the situation
        """

        if not check_if_inside(self.x, self.y, corner):
            self.state = 'dead'
            self.fitness = max(1, self.fitness - 5)

        if self.size > 4:
            for i in range(self.size - 1):
                if (self.x, self.y) == self.tail[-(i + 2)]:
                    self.state = 'dead'
                    self.fitness = max(1, self.fitness - 5)

        if self.state == 'alive':

            location = (self.x, self.y)
            self.tail.append(location)
            self.tail.pop(0)

            data = []

            distance = estimate_distance(self, fruit)
            angle = estimate_angle(self, fruit)

            x_direction_left = round(self.direction_x * math.cos(angle) - self.direction_y * math.sin(angle))
            y_direction_left = round(self.direction_x * math.sin(angle) + self.direction_y * math.cos(angle))

            x_direction_right = round(self.direction_x * math.cos(angle) + self.direction_y * math.sin(angle))
            y_direction_right = round(-self.direction_x * math.sin(angle) + self.direction_y * math.cos(angle))

            if not check_if_inside(self.x + x_direction_left, self.y + y_direction_left, corner):
                obstacle_to_left = 1
            else:
                obstacle_to_left = 0

            if not check_if_inside(self.x + x_direction_right, self.y + y_direction_right, corner):
                obstacle_to_right = 1
            else:
                obstacle_to_right = 0

            if not check_if_inside(self.x + self.direction_x, self.y + self.direction_y, corner):
                obstacle_ahead = 1
            else:
                obstacle_ahead = 0

            data.append(distance)
            data.append(angle)
            data.append(obstacle_ahead)
            data.append(obstacle_to_left)
            data.append(obstacle_to_right)

            self.output = self.predict(data)

            if np.argmax(self.output) == 0:
                self.direction_x = x_direction_left
                self.direction_y = y_direction_left
            elif np.argmax(self.output) == 1:
                self.direction_x = x_direction_right
                self.direction_y = y_direction_right

            self.x = self.x + self.direction_x
            self.y = self.y + self.direction_y

            distance_after = estimate_distance(self, fruit)

            # if distance_after < distance:
            #     self.fitness += 6
            # else:
            #     self.fitness = max(1, self.fitness - 7.5)

    def eat(self, obj):
        if self.x == obj.x and self.y == obj.y:
            self.size += 1
            location = (self.x, self.y)
            self.tail.append(location)
            self.fitness += 10

    def copy(self):
        """
        This function allows us to create a copy of the brain
        """
        brain = Brain((self.structure), activation_function=self.activation_function)
        brain.weights = copy.deepcopy(self.weights)
        brain.biases = copy.deepcopy(self.biases)

    def mutate(self, probability, rate):
        """
        This is very similar to the mutate function below but instead of giving a completely new weights or bias we are adding an increment which depends on the rate. The probability argument controls the chances that a given weight or bias mutates (as usual)
        """
        for i in range(self.number_of_transitions):
            shape = np.shape(self.weights[i])
            size = self.weights[i].size
            weights = self.weights[i].flatten()
            for j in range(len(weights)):
                if np.random.uniform(0, 1) < probability:
                    weights[j] = weights[j] + rate * np.random.normal(0, 1 / np.sqrt(shape[0]))
            self.weights[i] = weights.reshape(shape)
            for j in range(len(self.biases[i])):
                if np.random.uniform(0, 1) < probability:
                    self.biases[i][j] = self.biases[i][j] + rate * np.random.normal(0, 1)

    def mutate1(self, probability):
        """
        The below code mutates the weights and biases in the brain. We go over all the transition in the brain, we remember the shape of the weight matrix for each transition, then we reshape the weights into a 1 dimensional array. We go over this 1 dimensional array and if a uniformly distributed random variable takes a value less than some probability, we give this particular weight a random value. Then we transform the 1 dimensional array back to its original size. The procedure is less complex for the biases as they are already in 1 dimensional size
        """
        for i in range(self.number_of_transitions):
            shape = np.shape(self.weights[i])
            size = self.weights[i].size
            weights = self.weights[i].flatten()
            for j in range(len(weights)):
                if np.random.uniform(0, 1) < probability:
                    weights[j] = np.random.normal(0, 1 / np.sqrt(shape[0]))
            self.weights[i] = weights.reshape(shape)
            for j in range(len(self.biases[i])):
                if np.random.uniform(0, 1) < probability:
                    self.biases[i][j] = np.random.normal(0, 1)


def crossover(obj1, obj2):
    """
    This function takes two Brain objects as inputs and returns another Brain object with weights and biases taken from the parent objects randomly
    """

    assert obj1.structure == obj2.structure, 'The structures of the two brains are different'
    assert obj1.activation_function == obj2.activation_function, 'The activation functions of the two brains are different'

    new_brain = Brain((obj1.structure), activation_function=obj1.activation_function)

    for i in range(obj1.number_of_transitions):
        shape = obj1.weights[i].shape
        weights1 = obj1.weights[i].flatten()
        weights2 = obj2.weights[i].flatten()
        biases1 = obj1.biases[i]
        biases2 = obj2.biases[i]
        weights_combined = []
        biases_combined = []
        for j in range(len(weights1)):
            if np.random.uniform(0, 1) < 0.5:
                weights_combined.append(weights1[j])
            else:
                weights_combined.append(weights2[j])
        for j in range(len(biases1)):
            if np.random.uniform(0, 1) < 0.5:
                biases_combined.append(biases1[j])
            else:
                biases_combined.append(biases2[j])
        new_brain.weights[i] = np.asarray(weights_combined).reshape(shape)
        new_brain.biases[i] = np.asarray(biases_combined)

    return new_brain


class Population(object):

    def __init__(self, size, structure, activation_function='sigmoid', body=None):
        self.size = size
        self.members = [Brain(structure, activation_function=activation_function) for i in range(self.size)]
        self.state = 'alive'
        self.member_states = None
        self.number_alive = None
        self.total_population_fitness = None
        self.member_fitness = None
        self.mating_pool = None
        self.children = None
        self.generation = 1
        self.fittest_brain = None

    def draw(self):
        for i in range(self.size):
            self.members[i].draw()

    def move(self):
        for i in range(self.size):
            self.members[i].move()

    def check_state(self):
        self.member_states = [self.members[i].state for i in range(self.size)]
        self.number_alive = self.member_states.count('alive')
        if 'alive' not in self.member_states:
            self.state = 'dead'

    def evolve(self, elitism='on', save='off', probability=0.05, rate=0.05):
        """
        This code allows our population to evolve and get better by means of crossover and mutation. If parameter 'elitism' is 'on', we will be keeping the best overall performer (not necessarily from the current generation)
        """
        if self.state == 'dead':

            self.member_fitness = [self.members[i].fitness for i in range(self.size)]

            self.fittest_brain = self.members[self.member_fitness.index(max(self.member_fitness))]

            if save == 'on':
                self.fittest_brain.save_as('fittest_brain')

            self.total_population_fitness = sum(self.member_fitness)

            print('Total population fitness is %s' % (self.total_population_fitness))

            self.mating_pool = [[self.members[i]] * round(self.member_fitness[i] * 1000 / self.total_population_fitness) for i in range(self.size)]

            self.mating_pool = [brain for sublist in self.mating_pool for brain in sublist]

            self.children = []

            if elitism == 'on':

                self.children.append(self.fittest_brain)

                for i in range(self.size - 1):
                    parent1 = random.choice(self.mating_pool)
                    parent2 = random.choice(self.mating_pool)
                    child = crossover(parent1, parent2)
                    child.mutate(probability, rate)
                    self.children.append(child)
            else:
                for i in range(self.size):
                    parent1 = random.choice(self.mating_pool)
                    parent2 = random.choice(self.mating_pool)
                    child = crossover(parent1, parent2)
                    child.mutate(probability, rate)
                    self.children.append(child)

            self.members = self.children

            self.members[0].state = 'alive'

            self.state = 'alive'
            self.generation += 1

        else:
            print('Cannot evolve: some members are still alive')


if __name__ == '__main__':
    main()
