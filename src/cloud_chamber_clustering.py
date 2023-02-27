import os
import cv2
import random
import numpy as np
import math

# Obtenir le nombre d'image:
path = 'IMAGE\PATH\TO\GET\THE\ACQUISITION'
num_img = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

# Constante:
CONTINUER = True
WHITE = 254.0
SENSIBILITY_B_W = 50
DELTA = 1 # correspond aux nombres d'images que l'on souhaite utiliser pour réaliser le calque
MIN_VAL = 70

def norme(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def cluster_detection(mask_img):
    data_coordinate = []
    for i in range(mask_img.shape[0]):
        for j in range(mask_img.shape[1]):
            color = mask_img[i][j]
            color.sort()
            if color[-1] == WHITE and (i, j) not in data_coordinate:
                data_coordinate.append((i, j))
    return data_coordinate

K = 35 # random.randint(0, num_img -DELTA) # correspond à la séquence que l'on étudie (vaire entre 0 et num_img)
print("IMAGE N°", K)
# Lecture des images::
img_1 = cv2.imread(path + "\\img_{}.jpeg".format(K,))
img_i = cv2.imread(path + "\\img_{}.jpeg".format(K + DELTA,))
# Paramétrages:
diff = cv2.absdiff(img_1, img_i)
ret, thresh = cv2.threshold(diff, SENSIBILITY_B_W, 255, cv2.THRESH_BINARY)
# inverted = cv2.bitwise_not(thresh)

# Define the structuring element
kernel_1 = np.ones((5, 5), np.uint8)

# Perform morphological closing
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_1)
closing = cv2.fastNlMeansDenoisingColored(closing, None, 10, 10, 7, 21)
coord = cluster_detection(closing)
# Coordonnée des clusters:
cluster_container = []
cluster_lst = []
while len(coord) > 0:
    if len(cluster_lst) < 1:
        cluster_lst.append(coord[0])
        coord.remove(coord[0])
    for point in cluster_lst:
        for points in coord:
            if norme(point, points) > 0 and norme(point, points) <= 2:
                print("NORME", point, "ET", points, "=", norme(point, points))
                cluster_lst.append(points)
                coord.remove(points)
    cluster_container.append(cluster_lst)
    cluster_lst = []

index = []
for i in range(0, len(cluster_container) - 1):
    if len(cluster_container[i]) < MIN_VAL:
        index.append(i)
index.sort(reverse=True)
for i in range(0, len(index) - 1):
    cluster_container.pop(index[i])
for clusters in cluster_container:
    for coord in clusters:
        closing[coord[0], coord[1]] = (0, 0, 255)
# Affichage:
cv2.imshow("img_1", img_1)
cv2.imshow("test", thresh)
cv2.imshow("closing", closing)
# sauvegarde:
# cv2.imwrite('image_compte rendu/nettoyage.jpg', thresh)
cv2.imwrite('WHERE/YOU/WANT/TO/SAVE/cluster.jpg', closing)
# cv2.imshow("inverted", inverted)
cv2.waitKey(0)
cv2.destroyAllWindows()
