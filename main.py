
import pyglet
import random
import math
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from collections import OrderedDict
from sympy import geometry

pyglet.resource.path = [""]
pyglet.resource.reindex()

SCREEN_SIZE = (800, 600)

# "Brain" values
inputs = 3
first_layer = 8
outputs = 2

class Food(pyglet.sprite.Sprite):
    """
    Sprite class for the food the creatures eat
    """
    def __init__(self,x,y):
        """
        Init food - set img, pos and scale
        """
        food_image = pyglet.resource.image("food-boi.png")
        
        #food_image.anchor_x = food_image.width/2
        #food_image.anchor_y = food_image.height/2

        super(Food, self).__init__(food_image)

        self.x = x
        self.y = y
        self.scale = 0.05

    def checkLineCollision(self, P1, P2):
        """
        Determine whether the line from P1 to P2 intersects the food circle
        """
        Q = np.array((self.x, self.y), dtype=np.float64)  # Circle pos - float 64 to prevent compute errors
        r = 6                                             # Circle radius
        V = np.subtract(P2, P1)                           # Vector along line segment

        a = np.dot(V, V)
        b = 2 * np.dot(V, np.subtract(P1, Q))
        c = np.dot(P1, P1) + np.dot(Q,Q) - 2 * np.dot(P1, Q) - r**2

        disc = b**2 - 4 * a * c
        if disc < 0:
            return False
        else:
            sqrt_disc = math.sqrt(disc)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)

            if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
                return False
            else:
                return True

class Creature(pyglet.sprite.Sprite):
    """
    Holds logic for creature, including their brain
    """
    def __init__(self, x, y, genes=None):

        creature_image = pyglet.resource.image("creature-boi.png")
        creature_image.anchor_x = creature_image.width/2
        creature_image.anchor_y = creature_image.height/2

        super(Creature, self).__init__(creature_image)

        # initial values
        self.x = x
        self.y = y
        self.scale = 0.17
        self.rotation = random.randint(0,360)
        self.speed = 1
        self.score = 0

        # randomized fully-connected brain model
        self.model = Sequential([
            Dense(first_layer, input_shape=(inputs,)),
            Activation('tanh'),
            Dense(outputs),
            Activation('tanh'),
        ])

        print(self.model.get_weights())
        
        if not genes:
            self.w1 = [[(random.random() * 2) - 1.0 for _ in range(first_layer)] for _ in range(inputs)]
            self.b1 = [(random.random() * 2) - 1.0 for _ in range(first_layer)]
            self.w2 = [[(random.random() * 2) - 1.0 for _ in range(outputs)] for _ in range(first_layer)]
            self.b2 = [(random.random() * 2) - 1.0 for _ in range(outputs)]

            print(self.b2)

            self.genes = [self.w1, self.b1, self.w2, self.b2]
            self.model.set_weights(self.genes)
            #print(self.model.get_weights())
        else:
            self.genes = genes
    
    def update(self, food):
        """
        General method to update logic of creature
        """
        # Move the creature according to current rotation and speed
        self.x += math.sin(math.radians(self.rotation)) * self.speed
        self.y += math.cos(math.radians(self.rotation)) * self.speed

        self.move(food)

    def draw(self):
        super(Creature, self).draw()

        self.draw_eyes(self.rotation, 100)
        self.draw_eyes(self.rotation + 20, 100)
        self.draw_eyes(self.rotation - 20, 100)

    def move(self, food):
        """
        Process Neural Network inputs and move with the outputs
        """

        # Determine line ends for the 3 "eyes"
        left_l = self.calc_line_end((self.x, self.y), self.rotation - 20, 100)
        center_l = self.calc_line_end((self.x, self.y), self.rotation, 100)
        right_l = self.calc_line_end((self.x, self.y), self.rotation + 20, 100)
        
        # initialize the inputs
        in1 = 0
        in2 = 0
        in3 = 0
        
        # For every food, if collision exists, set corresponding
        # input to 1
        for f in food:

            if(f.checkLineCollision((self.x, self.y), center_l) == True):
                in1 = 1
            if(f.checkLineCollision((self.x, self.y), left_l) == True):
                in2 = 1
            if(f.checkLineCollision((self.x, self.y), right_l) == True):
                in3 = 1

        # NOT USED - Check collisions to edges of the window.
        #if (new_x < 0 or new_y < 0) or (new_x > SCREEN_SIZE[0] or new_y > SCREEN_SIZE[1]):
        #    in1 = -1
        #if (new_x_1 < 0 or new_y_1 < 0) or (new_x_1 > SCREEN_SIZE[0] or new_y_1 > SCREEN_SIZE[1]):
        #    in2 = -1
        #if (new_x_2 < 0 or new_y_2 < 0) or (new_x_2 > SCREEN_SIZE[0] or new_y_2 > SCREEN_SIZE[1]):
        #    in3 = -1

        # Use current neural net to predict outputs
        output = self.model.predict(np.array([[in1,in2,in3]]))[0]

        # Update values with NN output
        self.rotation += output[0] * 5
        self.speed = output[1] + .5
    
    def draw_eyes(self, rotation, length):
        """
        Function to draw the creatures eyes out in either direction
        """
        end_pos = self.calc_line_end((self.x, self.y), rotation, length)

        pyglet.graphics.draw(2, pyglet.gl.GL_LINE_STRIP,
        ('v2i', (int(self.x), int(self.y), end_pos[0], end_pos[1]))
        )

    def calc_line_end(self, P, rotation, length):
        """
        With a line starting at point P, with length and rotation, calculate and return the end point
        """
        rotation = math.radians(rotation)
        new_x = int(P[0] + (length * math.sin(rotation)))
        new_y = int(P[1] + (length * math.cos(rotation)))

        return (new_x, new_y)


class AppWindow(pyglet.window.Window):
    """
    Main class - handles the pyglet window
    """
    def __init__(self):
        super(AppWindow, self).__init__(width=SCREEN_SIZE[0],height=SCREEN_SIZE[1])
        
        # set size of window
        self.width = SCREEN_SIZE[0]
        self.height = SCREEN_SIZE[1]

        #Set key handler.
        self.keys = pyglet.window.key.KeyStateHandler()
        self.push_handlers(self.keys)

        self.set_caption("Groboids Evolution Sim")

        # initialize object lists
        self.creatures = []
        self.food = []

        # place x creatures in center of screen
        for _ in range(14):
            self.creatures.append(Creature(SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2))
        
        # place x food at random spots on screen
        for _ in range(40):
            self.food.append(Food(random.randint(0,SCREEN_SIZE[0]), random.randint(0,SCREEN_SIZE[1])))

        self.generation = 0
        self.ticks = 0
        pyglet.clock.schedule_interval(self.game_tick, 0.01)

    def game_tick(self, dt):
        """
        Game loop method that calls update and draw
        """
        self.update()
        self.draw()

    def update(self):
        """
        Handles the logic of the program
        """
        self.ticks += 1
        if self.ticks > 1000:
            ranked_creatures = []
            for creature in self.creatures:
                ranked_creatures.append((creature.score, creature.genes))
            ranked_creatures = sorted(ranked_creatures, key = lambda t: t[0])
            
            top_creatures = ranked_creatures[:5]

            for i in range(20):
                c1 = random.choice(top_creatures)
                c2 = random.choice(top_creatures)
                
                for i in np.nditer(c1[1]):
                    print(i,)
                #for i, gene in enumerate(c1[1][0]):
                #    for j, genome in enumerate(gene):
                #        print(genome)
                #        for t in numpy.nditer(genome):
                #            print(t)
                while True:
                    pass
                                #self.creatures.append(Creature(random.randint(0,SCREEN_SIZE[0]), random.randint(0,SCREEN_SIZE[1])),
                                # )

            # Generation is done, reset - mutate, and start again
            # empty list
            
            # run function
                # find the to
            self.ticks = 0

        print(self.ticks)
        # update creatures
        for creature in self.creatures:
            creature.update(self.food)
        
        # update foods
        for f in self.food:
            f.update()
            # check for collision between food and creatures
            for c in self.creatures:
                dist = math.hypot(f.x - c.x, f.y - c.y)
                
                # remove if the two circles are close enough
                if dist < 11:
                    self.food.remove(f)
                    self.food.append(Food(random.randint(0,SCREEN_SIZE[0]), random.randint(0,SCREEN_SIZE[1])))


    def draw(self):
        """
        Handles the drawing of everything
        """
        # Clear the screen at the start of every draw cycle
        self.clear()

        # render creatures
        for creature in self.creatures:
            creature.draw()

        # render foods
        for f in self.food:
            f.draw()

    def center_image(self, image):
        """
        Honestly not sure if this method is useful
        """
        image.anchor_x = image.width/2
        image.anchor_y = image.height/2

# Start ze program, yaaa
game_window = AppWindow()
pyglet.app.run()