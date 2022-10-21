import numpy as np

def from28to40(images):
    return np.pad(images, ((0, 0),(6, 6),(6, 6)) )