import numpy as np
from imagesAndFiles import showAndSaveRGBgrid, saveRGBgrid, pointsToBWgrid

# Options
dt = 200
drag = dt / 100000
G = 1
w, h = (1920, 1080)
attrD = 200
attractors = [[w / 2 - attrD, h / 2 - attrD], [w / 2 - attrD, h / 2 + attrD], [w / 2 + attrD, h / 2 - attrD], [w / 2 + attrD, h / 2 + attrD]]
attractorColors = [(12, 10, 62), (123, 30, 122), (179, 63, 98), (249, 86, 79)]



# Code
attractors = [np.array(attractor) for attractor in attractors]
attractorColors = np.array(attractorColors)
assert len(attractors) == len(attractorColors)


def generateGridPoints(width, height):
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    return np.column_stack([x.ravel(), y.ravel()]).astype(np.float64)

def findClosestAttractors(arr):
    dists2 = np.linalg.norm(arr[:, np.newaxis] - np.array(attractors), axis=2)
    return np.argmin(dists2, axis=1)

def colorBasedOnClosestAttractor(arr):
    grid = findClosestAttractors(arr).reshape((h, w))
    gridRGB = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            gridRGB[i, j] = attractorColors[grid[i, j]]
    return gridRGB



p = generateGridPoints(w, h)
v = np.zeros(p.shape, dtype=np.float64)

try:
    steps = 0
    while True:
        steps += 1
        saveRGBgrid(pointsToBWgrid(p, w, h), animationNumber='1a')
        saveRGBgrid(colorBasedOnClosestAttractor(p), animationNumber='1b')

        f = np.zeros(p.shape, dtype=np.float64)
        for attractor in attractors:
            f += G * (attractor - p) / (np.linalg.norm(p - attractor, axis=1)[:, np.newaxis] ** 3)
        v += dt * f
        v += -v * drag
        p += dt * v
        print(steps, '|', p[1])

        if steps % 100 == 0:
            dists = np.zeros(p.shape[0], dtype=np.float64)
            for attractor in attractors:
                dists += np.maximum(np.linalg.norm(p - attractor, axis=1), 1e-10)
            dists = np.nan_to_num(dists)
            print(dists)
            print(np.max(dists))
            if np.max(dists) <= 200:
                showAndSaveRGBgrid(colorBasedOnClosestAttractor(p))



except Exception as e:
    showAndSaveRGBgrid(pointsToBWgrid(p, w, h))
    showAndSaveRGBgrid(colorBasedOnClosestAttractor(p))
    raise e
except:
    showAndSaveRGBgrid(pointsToBWgrid(p, w, h))
    showAndSaveRGBgrid(colorBasedOnClosestAttractor(p))







