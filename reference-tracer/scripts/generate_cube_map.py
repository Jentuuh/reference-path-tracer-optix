import cv2
import numpy as np
import glob

def generate_cube_map(res:int, faces:list):
    width = 4 * res
    height = 3 * res
    output_cube_map = np.zeros((width, height, 3))

    # Negative X face 
    output_cube_map[0 : res, res : 2 * res] = faces[1]
    # Positive Z face
    output_cube_map[res : 2* res, res : 2 * res] = faces[4]
    # Positive X face 
    output_cube_map[2 * res : 3 * res, res : 2 * res] = faces[0]
    # Negative Z face 
    output_cube_map[3 * res : 4 * res, res : 2 * res] = faces[5]
    # Positive Y face 
    output_cube_map[res : 2 * res, 0 : res] = faces[2]
    # Negative Y face 
    output_cube_map[res : 2 * res, 2 * res : 3 * res] = faces[3]

    #cv2.imshow("Cubemap", output_cube_map)
    cv2.imwrite("output/cubemap.png", output_cube_map * 255)
    cv2.waitKey(0)

def main():
    # right, left, up, down, back, front
    faces = []
    for img in glob.glob("data/cubemap_face_*.png"):
        f = cv2.imread(img)/255
        faces.append(f)

    generate_cube_map(128, faces)

main()