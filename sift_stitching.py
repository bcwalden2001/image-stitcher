import numpy as np
import cv2
from matplotlib import pyplot as plt
import random

np.set_printoptions(suppress=True)

img1 = cv2.imread('queenstownLeft.jpg',0) # queryImage
img2 = cv2.imread('queenstownRight.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)        # keypoints are stored in a tuple
kp2, des2 = sift.detectAndCompute(img2,None)        # descriptors are stored in a numpy n-dimensional array

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2) #NOTE: 'None' parameter has to be added (not in documentation)

# plt.imshow(img3),plt.show()

pts1 = np.zeros((len(good),2), np.float32)
pts2 = np.zeros((len(good),2), np.float32)
for m in range(len(good)):
	pts1[m] = kp1[good[m][0].queryIdx].pt   
	pts2[m] = kp2[good[m][0].trainIdx].pt

def find_best_homography_matrix(RANSAC_iterations):

    smallest_dist = 1
    best_H = None
    dist = 0   

    for _ in range(RANSAC_iterations):

        # 1) Selecting 4 random matches/indices from the good matches

        i = random.randint(0, len(good) - 1)

        # Match 1
        qIdx = good[i][0].queryIdx
        tIdx = good[i][0].trainIdx
        x1 = kp1[qIdx].pt[0] 
        y1 = kp1[qIdx].pt[1] 
        x2 = kp2[tIdx].pt[0]
        y2 = kp2[tIdx].pt[1]

        i = random.randint(0, len(good) - 1)

        # Match 2
        qIdx_2 = good[i][0].queryIdx
        tIdx_2 = good[i][0].trainIdx
        x1_2 = kp1[qIdx_2].pt[0] 
        y1_2 = kp1[qIdx_2].pt[1] 
        x2_2 = kp2[tIdx_2].pt[0]
        y2_2 = kp2[tIdx_2].pt[1]

        i = random.randint(0, len(good) - 1)

        # Match 3
        qIdx_3 = good[i][0].queryIdx
        tIdx_3 = good[i][0].trainIdx
        x1_3 = kp1[qIdx_3].pt[0] 
        y1_3 = kp1[qIdx_3].pt[1] 
        x2_3 = kp2[tIdx_3].pt[0]
        y2_3 = kp2[tIdx_3].pt[1]

        i = random.randint(0, len(good) - 1)

        # Match 4
        qIdx_4 = good[i][0].queryIdx
        tIdx_4 = good[i][0].trainIdx
        x1_4 = kp1[qIdx_4].pt[0] 
        y1_4 = kp1[qIdx_4].pt[1] 
        x2_4 = kp2[tIdx_4].pt[0]
        y2_4 = kp2[tIdx_4].pt[1]

        # 2) Contructing 8x9 A matrix
        A = np.array([[0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2],                       # Match 1
                      [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2],

                      [0, 0, 0, -x1_2, -y1_2, -1, y2_2*x1_2, y2_2*y1_2, y2_2],         # Match 2
                      [x1_2, y1_2, 1, 0, 0, 0, -x2_2*x1_2, -x2_2*y1_2, -x2_2],

                      [0, 0, 0, -x1_3, -y1_3, -1, y2_3*x1_3, y2_3*y1_3, y2_3],         # Match 3
                      [x1_3, y1_3, 1, 0, 0, 0, -x2_3*x1_3, -x2_3*y1_3, -x2_3],
                
                      [0, 0, 0, -x1_4, -y1_4, -1, y2_4*x1_4, y2_4*y1_4, y2_4],         # Match 4
                      [x1_4, y1_4, 1, 0, 0, 0, -x2_4*x1_4, -x2_4*y1_4, -x2_4]], dtype=np.float32)

        # 3) Running SVD on A to get a homography matrix
        U, s, V = np.linalg.svd(A, full_matrices=True)
        hCol = np.zeros((9,1), np.float64)
        hCol = V[8,:]
        H = hCol.reshape(3, 3).astype(np.float32)   # Re-arranging last column in V (9x1) into a 3x3

        # 4) Evaluating homography matrix
        for m in range(len(good)):    # Loop through all good matches
            pt1 = np.array([pts1[m][0], pts1[m][1], 1.0], dtype=np.float64)     # Initializing homogenous coordinates (x and x') 
            pt2 = np.array([pts2[m][0], pts2[m][1], 1.0], dtype=np.float64)     # 64 bit floats prevent rounding to 0.0

            distance_vector = abs(np.dot(H, pt1) - pt2)   
            distance_vector_scaled = distance_vector
            distance_vector_scaled /= distance_vector[2]    # Scaling all values by homogenous coordinate so w = 1

            dist = np.linalg.norm(distance_vector_scaled[:2] - pts2[m])     # Normalizing points for scalar comparison
            
            # Finding the smallest distance (closest to 0) -> closest match of pt1 to pt2
            if dist < smallest_dist:
                #   print("Distance: ", dist)
                #   print("Smallest distance initial", smallest_dist)
                  smallest_dist = dist  
                #   print("New smallest distance: ", smallest_dist )
                  best_H = H    

    best_H_scaled = best_H
    best_H_scaled /= best_H[2,2]    # Scaling all values by homogenous coordinate so h33 = 1
    return best_H_scaled

print(f"Best homography matrix:\n {find_best_homography_matrix(500)}")

opencvH, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
print("H matrix estimated by OpenCV (for comparison):\n", opencvH)


###### Invert matrix
##    Hinv = np.linalg.inv(H)


