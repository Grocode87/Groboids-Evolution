
import pyglet
import random
import math
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from collections import OrderedDict
from pyglet.gl import *

pyglet.resource.path = [""]
pyglet.resource.reindex()

SCREEN_SIZE = (800, 600)

# "Brain" values
inputs = 3
first_layer = 5
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
        
        food_image.anchor_x = food_image.width/2
        food_image.anchor_y = food_image.height/2

        super(Food, self).__init__(food_image)

        self.x = x
        self.y = y
        self.scale = 0.1
        

    def checkLineCollision(self, P1, P2):
        """
        Determine whether the line from P1 to P2 intersects the food circle

        Credit for this goes to the answer in this question - thank you 'Gareth Rees'
        https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
        """
        Q = np.array((self.x, self.y), dtype=np.float64)  # Circle pos - float 64 to prevent compute errors
        r = 6                                             # Circle radius
        V = np.subtract(P2, P1)                           # Vector along line segment

        # compute the coefficients
        a = np.dot(V, V)
        b = 2 * np.dot(V, np.subtract(P1, Q))
        c = np.dot(P1, P1) + np.dot(Q,Q) - 2 * np.dot(P1, Q) - r**2

        # check if discriminent is negative
        disc = b**2 - 4 * a * c
        if disc < 0:
            return False
        else:
            # check if line segment is long enough to collide with circle
            sqrt_disc = math.sqrt(disc)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)
            
            # return true or false according to above
            if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
                return False
            else:
                return True

class Creature(pyglet.sprite.Sprite):
    """
    Holds logic for creature, including their brain
    """
    def __init__(self, x, y, genes=None, new_color=None):

        creature_image = pyglet.resource.image("creature-boi.png")
        creature_image.anchor_x = creature_image.width/2
        creature_image.anchor_y = creature_image.height/2

        super(Creature, self).__init__(creature_image)

        # initial values
        self.x = x
        self.y = y
        self.scale = 0.20
        self.rotation = random.randint(0,360)
        self.speed = 1
        self.score = 0
        self.show_eyes = False

        # randomized fully-connected brain model
        self.model = Sequential([
            Dense(first_layer, input_shape=(inputs,)),
            Activation('tanh'),
            Dense(outputs),
            Activation('tanh'),
        ])
        
        # If genes are not passed in, create random dna
        if not genes:
            self.w1 = [[(random.random() * 2) - 1.0 for _ in range(first_layer)] for _ in range(inputs)]
            self.b1 = [(random.random() * 2) - 1.0 for _ in range(first_layer)]
            self.w2 = [[(random.random() * 2) - 1.0 for _ in range(outputs)] for _ in range(first_layer)]
            self.b2 = [(random.random() * 2) - 1.0 for _ in range(outputs)]

            self.genes = [self.w1, self.b1, self.w2, self.b2]
            self.model.set_weights(self.genes)
        # set genes to passed in genes
        else:
            if len(genes) == 2:
                self.genes = genes[1]
            else:
                self.genes = genes
            self.model.set_weights(self.genes)
        
        # if no color is passed in, create random color
        if not new_color:
            self.color = [random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255)]
        else:
            # set color to passed in + mutation
            self.color = self.mutate_colors(new_color)

    
    def update(self, food):
        """
        General method to update logic of creature
        """
        # Move the creature according to current rotation and speed
        self.x += math.sin(math.radians(self.rotation)) * self.speed
        self.y += math.cos(math.radians(self.rotation)) * self.speed

        self.move(food)
        # make sure rotation stays between 0 and 360
        if self.rotation > 360:
            self.rotation = self.rotation - 360
        elif self.rotation < 0:
            self.rotation = 360 - self.rotation

    def draw(self):
        #super(Creature, self).draw()
        self.circle(self.x, self.y, 10)
        # draw the 3 eyes of the creatures
        #self.draw_eyes(self.rotation, 100)
        if self.show_eyes:
            self.draw_eyes(self.rotation + 20, 100)
            self.draw_eyes(self.rotation - 20, 100)

    def move(self, food):
        """
        Process Neural Network inputs and move with the outputs
        """

        # Determine line ends for the 3 "eyes"
        left_l = self.calc_line_end((self.x, self.y), self.rotation - 20, 100)
        #center_l = self.calc_line_end((self.x, self.y), self.rotation, 100)
        right_l = self.calc_line_end((self.x, self.y), self.rotation + 20, 100)
        
        # initialize the inputs
        in1 = 0
        in2 = 0
        in3 = 0
        
        # For every food, if collision exists, set corresponding
        # input to 1
        for f in food:
            if(f.checkLineCollision((self.x, self.y), left_l) == True):
                in2 = 1
            if(f.checkLineCollision((self.x, self.y), right_l) == True):
                in3 = 1

        # Use current neural net to predict outputs
        output = self.model.predict(np.array([[in2,in3, (self.rotation / 180) - 1]]))[0]

        # Update values with NN output
        self.rotation += output[0] * 10
        self.speed = output[1] * 1.5
    
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

    def circle(self, x, y, radius):
        """
        Function I got online - used to print a circle in GL
        """
        iterations = int(1*radius*math.pi)
        s = math.sin(2*math.pi / iterations)
        c = math.cos(2*math.pi / iterations)

        dx, dy = radius, 0

        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)
        for i in range(iterations+1):
            glVertex2f(x+dx, y+dy)
            dx, dy = (dx*c - dy*s), (dy*c + dx*s)
        glEnd()

    def mutate_colors(self, colors):
        """
        Mutate the color by a little bit
        """
        new_color = []

        for color in colors:
            color_change = (random.randint(0,255) / 1) / 10

            if random.randint(0, 1) == 0:
                color_change = -color_change
            
            if color - color_change > 255 or color - color_change < 0:
                color_change = -color_change

            new_color.append(color - color_change)
        
        return new_color

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
            self.creatures.append(Creature(random.randint(0,SCREEN_SIZE[0]), random.randint(0,SCREEN_SIZE[1])))
        
        # place x food at random spots on screen
        for _ in range(40):
            self.food.append(Food(random.randint(0,SCREEN_SIZE[0]), random.randint(0,SCREEN_SIZE[1])))

        self.generation = 0
        self.ticks = 0
        self.score_record = 0

        # Set the label default values and their pos
        self.generation_label = pyglet.text.Label('Generation: 0',
                                                  font_name='Times New Roman',
                                                  font_size=16,
                                                  x=self.width//2, y=self.height - 20,
                                                  anchor_x='center', anchor_y='center')
        self.tick_label = pyglet.text.Label('Tick: 0',
                                                  font_name='Times New Roman',
                                                  font_size=13,
                                                  x=10, y=self.height - 10,
                                            anchor_x='left', anchor_y='top')
        self.highscore_label = pyglet.text.Label('Best Score: 0',
                                            font_name='Times New Roman',
                                            font_size=13,
                                            x=self.width - 10, y=30,
                                            anchor_x='right', anchor_y='bottom')
        self.record_label = pyglet.text.Label('Record: 0',
                                            font_name='Times New Roman',
                                            font_size=13,
                                            x=self.width - 10, y=10,
                                            anchor_x='right', anchor_y='bottom')

        self.paused = False
        pyglet.clock.schedule_interval(self.game_tick, -1)

    def game_tick(self, dt):
        """
        Game loop method that calls update and draw
        """
        # update everything if the game is not paused
        if not self.paused:
            self.update()
        self.draw()

    def update(self):
        """
        Handles the logic of the program
        """
        self.ticks += 1
        self.tick_label.text = "Tick: " + str(self.ticks)

        # 2000 ticks reached = perform evolution
        if self.ticks > 2000:
            ranked_creatures = []
            for creature in self.creatures:
                ranked_creatures.append((creature.score, creature.genes, creature.color))
            ranked_creatures = sorted(ranked_creatures, key = lambda t: t[0], reverse=True)
            
            top_creatures = ranked_creatures[:4]
            new_creatures = []

            # determine if best creature beat score record
            if ranked_creatures[0][0] > self.score_record:
                self.score_record = ranked_creatures[0][0]
                self.record_label.text = "Record:" + str(self.score_record)

            for i in range(14):
                # Pick two random creatures from the top list
                c1 = random.choice(top_creatures)
                c2 = random.choice(top_creatures)
                
                new_genes = []
                new_color = []
                for chromo_index, chromosone in enumerate(c1[1]):
                    ls_dimensions = len(np.array(chromosone).shape)
                    updated_chromosone = []

                    for x_index, x in enumerate(chromosone):
                        if ls_dimensions == 1:
                            compare_gene = c2[1][chromo_index][x_index]
                            selected_gene = self.mutate_gene(random.choice([x, compare_gene]))
                            
                            updated_chromosone.append(selected_gene)
                        else:
                            updated_chromosone.append([])
                            for y_index, y in enumerate(x):
                                compare_gene = c2[1][chromo_index][x_index][y_index]
                                selected_gene = self.mutate_gene(random.choice([y, compare_gene]))

                                updated_chromosone[x_index].append(selected_gene)

                    new_genes.append(updated_chromosone)
                print(new_genes)

                for index_c, color in enumerate(c1[2]):
                    new_color.append(random.choice([c1[2][index_c], c2[2][index_c]]))
                new_creatures.append(Creature(random.randint(0,SCREEN_SIZE[0]), random.randint(0,SCREEN_SIZE[1]),
                                      new_genes, new_color))
                
                #for i, gene in enumerate(c1[1][0]):
                #    for j, genome in enumerate(gene):
                #        print(genome)
                #        for t in numpy.nditer(genome):
                #            print(t)
                                #self.creatures.append(Creature(random.randint(0,SCREEN_SIZE[0]), random.randint(0,SCREEN_SIZE[1])),
                                # )
            self.creatures = []
            self.creatures = new_creatures
            
            self.food = []
            for _ in range(40):
                self.food.append(Food(random.randint(0,SCREEN_SIZE[0]), random.randint(0,SCREEN_SIZE[1])))

            # Generation is done, reset - mutate, and start again
            # empty list
            
            # run function
                # find the to
            self.ticks = 0
            self.generation += 1;
            self.generation_label.text = "Generation: " + str(self.generation)

        #print(self.ticks)
        # update creatures
        top_score=0
        for creature in self.creatures:
            creature.update(self.food)
            if creature.score > top_score:
                top_score = creature.score
        self.highscore_label.text = "Best Score: " + str(top_score)
        
        # update foods
        food_to_remove = []
        for f in self.food:
            f.update()
            # check for collision between food and creatures
            for c in self.creatures:
                dist = math.hypot(f.x - c.x, f.y - c.y)


                # remove if the two circles are close enough
                if dist < 14:
                    if f in self.food:
                        self.food.remove(f)
                        c.score += 1
                    self.food.append(Food(random.randint(0,SCREEN_SIZE[0]), random.randint(0,SCREEN_SIZE[1])))

    def draw(self):
        """
        Handles the drawing of everything
        """
        # Clear the screen at the start of every draw cycle
        self.clear()

        # render creatures
        for creature in self.creatures:
            glColor3f(creature.color[0] / 255, creature.color[1] / 255, creature.color[2] / 255)
            creature.draw()

        # render foods
        for f in self.food:
            f.draw()

        self.generation_label.draw()
        self.tick_label.draw()
        self.highscore_label.draw()
        self.record_label.draw()

    def center_image(self, image):
        """
        Honestly not sure if this method is useful
        """
        image.anchor_x = image.width/2
        image.anchor_y = image.height/2

    def on_key_release(self, symbol, modifiers):
        if symbol == pyglet.window.key.E:
            for c in self.creatures:
                c.show_eyes = not c.show_eyes
        elif symbol == pyglet.window.key.SPACE:
            self.paused = not self.paused

    def mutate_gene(self, gene, mutation_chance=0.18, mutation_amount=0.16):
        """
        Perform random mutations on the passed in genes
        """
        total_genes = ((inputs + 1) * first_layer) + ((first_layer + 1) * outputs)
        rand_num = random.randint(0,total_genes)

        if rand_num < total_genes * mutation_chance:
            # do mutate
            mutation = (random.random() * 2 - 1.0) * mutation_amount
            gene = gene + mutation
            # make sure gene values saty within limits
            if gene > 1.0:
                gene = 1.0
            elif gene < -1.0:
                gene = -1.0

            return gene
        return gene
            
# Start ze program, yaaa
game_window = AppWindow()
pyglet.app.run()
