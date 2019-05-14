import numpy as np
import math
import pyglet


rotation = 0


window = pyglet.window.Window()


def collision(P1, P2):
    Q = np.array((362, 254), dtype=np.float64)             # Centre of circle
    r = 10                  # Radius of circle
    V = np.subtract(P2, P1)  # Vector along line segment

    a = np.dot(V, V)
    b = 2 * np.dot(V, np.subtract(P1, Q))
    c = np.dot(P1, P1) + np.dot(Q,Q) - 2 * np.dot(P1, Q) - r**2

    disc = b**2 - 4 * a * c
    if disc < 0:
        return False, None
    else:
        sqrt_disc = math.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)

        if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
            return False, None
        else:
            return True

print(collision((330, 309), (387, 227)))

P1 = (330, 309)
rotation = 0
P2 = (387, 227)



image = pyglet.resource.image('food-boi.png')
image.width = 12
image.height = 12
@window.event
def on_draw():
    window.clear()
    rotation = 1
    new_x = P1[0] + (100 * math.sin(rotation))
    new_y = P1[1] + (100 * math.cos(rotation))

    print(collision((330, 309), (387, 227)))

    pyglet.graphics.draw(2, pyglet.gl.GL_LINE_STRIP,
    ('v2i', (int(P1[0]), int(P1[1]), int(new_x), int(new_y)))
    )
    image.blit(362, 254)
pyglet.app.run()
