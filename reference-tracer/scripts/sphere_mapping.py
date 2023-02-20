import cv2
import numpy as np
import math

def xyz_to_cube_uv(x, y, z):
    absX = abs(x)
    absY = abs(y)
    absZ = abs(z)

    x_positive = x > 0 
    y_positive = y > 0 
    z_positive = z > 0 

    # Positive X
    if(x_positive and max(absX, absY, absZ) == absX):
        # u from +z to -z
        # v from -y to +y
        maxAxis = absX
        uc = -z
        vc = y
        index = 0
    # Negative X
    elif(not x_positive and max(absX, absY, absZ) == absX):
        maxAxis = absX
        uc = z
        vc = y
        index = 1
    # Positive Y
    elif(y_positive and max(absX, absY, absZ) == absY):
        maxAxis = absY
        uc = x
        vc = -z
        index = 2
    # Negative Y
    elif(not y_positive and max(absX, absY, absZ) == absY):
        maxAxis = absY
        uc = x
        vc = z
        index = 3
    # Positive Z
    elif(z_positive and max(absX, absY, absZ) == absZ):
        maxAxis = absZ
        uc = x
        vc = y
        index = 4
    elif(not z_positive and max(absX, absY, absZ) == absZ):
        maxAxis = absZ
        uc = -x
        vc = y
        index = 5

    # Shift from [-1; 1] to [0; 1]
    u = 0.5 * (uc / maxAxis + 1.0)
    v = 0.5 * (vc / maxAxis + 1.0)
    return u, v, index

def convert_cube_uv_to_xyz(index, u, v):
    # Shift [0; 1] to [-1; 1]
    uc = 2.0 * u - 1.0
    vc = 2.0 * v - 1.0

    if index == 0:
        x = 1.0 
        y= vc
        z= -uc
    elif index == 1:
        x = -1.0 
        y= vc
        z= uc
    elif index == 2:
        x = uc 
        y= 1.0
        z= -vc
    elif index == 3:
        x = uc
        y = -1.0
        z = vc
    elif index == 4:
        x = uc
        y = vc
        z = 1.0
    elif index == 5:
        x = -uc
        y = vc
        z = -1.0
    return x, y, z


def sample_environment():
    res = 128
    cube_map = [np.zeros((res, res, 3)), np.zeros((res, res, 3)), np.zeros((res, res, 3)), np.zeros((res, res, 3)), np.zeros((res, res, 3)), np.zeros((res, res, 3))]

    for i in range(6):
        for x in range(res):
            for y in range(res):
                u = float(x) / float(res)
                v = float(y) / float(res)
                dir_x, dir_y, dir_z = convert_cube_uv_to_xyz(i, u, v)
                radius = math.sqrt(dir_x ** 2 + dir_y ** 2 + dir_z ** 2)

                theta = np.arccos(dir_z / radius)
                phi = np.arctan2(dir_y, dir_x)

                if theta > (np.pi / 2.0):
                    color = np.array([0, 0, 255])
                else:
                    color = np.array([255, 255, 255])

                # if i == 0:
                #     color = np.array([0, 0, 255])
                # elif i == 1:
                #     color = np.array([255, 0, 0])
                # elif i == 2:
                #     color = np.array([0, 255, 0])
                # elif i == 3:
                #     color = np.array([255, 255, 0])
                # elif i == 4:
                #     color = np.array([255, 0, 255])
                # elif i == 5:
                #     color = np.array([255, 255, 255])

                cube_map[i][x, y] = color

    for i in range(6):
        cv2.imshow(f"Cube face {i}", cube_map[i])
        cv2.waitKey(0)



def make_sphere_map():
    sphere_map = np.zeros(512, 512, 3)
    counter = np.zeros((1024, 1024))

    thetas = np.linspace(0, np.pi, 180)
    phis = np.linspace(0, 2 * np.pi, 360)


    for theta in thetas:
        for phi in phis:
            x = (np.sin(theta) * np.cos(phi)) + 1  # x,y normally in [-1;1] --> shift to [0;2]
            y = (np.sin(theta) * np.sin(phi)) + 1 

            if theta > (np.pi / 2):
                color = np.array([255, 0, 0])
            else:
                color = np.array([0, 0, 255])
            
                        
            sphere_map[min(math.floor(x * 512), 1023), min(math.floor(y * 512), 1023)] += color
            counter[min(math.floor(x * 512), 1023),  min(math.floor(y * 512), 1023)] += 1
    
    cv2.imshow("Sphere map", sphere_map)
    cv2.waitKey(0)


def main():
    sample_environment()

main()