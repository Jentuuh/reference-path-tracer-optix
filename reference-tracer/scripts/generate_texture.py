import cv2 
import numpy as np

texture = np.zeros((1024, 1024, 4))

for x in range(1024):
    for y in range(1024):
        if y > 246 and y < 293 and x > 266 and x < 308:
            texture[x,y] = [255,255,255,255]

cv2.imwrite("../textures/black.png", texture)