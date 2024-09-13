import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def nextIndex(root, filename='index.txt'):
    filename = root + filename
    try:
        with open(filename, 'r') as file:
            index = int(file.read().strip())
    except FileNotFoundError:
        index = 1

    with open(filename, 'w') as file:
        file.write(str(index + 1))
    return index

def saveRGBgrid(arr, animationNumber=1, overridePath=None):
    img = Image.fromarray(arr, mode='RGB')
    img.save('Images/Animations/Animation' + str(animationNumber) + '/' + str(nextIndex('Images/Animations/Animation' + str(animationNumber) + '/')) + '.png' if overridePath is None else overridePath)

def showAndSaveRGBgrid(arr):
    saveRGBgrid(arr, overridePath='Images/' + str(nextIndex('Images/')) + '.png')

    plt.imshow(arr)
    plt.axis('off')
    plt.show()


def pointsToBWgrid(p, w, h):
    grid = np.zeros((h, w, 3), dtype=np.uint16)

    pRound = np.round(p).astype(np.uint16)
    pRound[:, 0] = np.clip(pRound[:, 0], 0, w - 1)  # Constrain x-values to [0, w-1]
    pRound[:, 1] = np.clip(pRound[:, 1], 0, h - 1)  # Constrain y-values to [0, h-1]

    grid[pRound[:, 1], pRound[:, 0]] = (255, 255, 255)

    return grid.astype(np.uint8)

if __name__ == '__main__':
    showAndSaveRGBgrid(np.full((1080, 1920, 3), 255, dtype=np.uint8))
    showAndSaveRGBgrid(np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8))
