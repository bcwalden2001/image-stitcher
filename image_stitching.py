import numpy as np
import cv2

np.set_printoptions(suppress=True)

# Loading grayscale images for keypoint detection
img1 = cv2.imread('churchLeft.png', 0)
img2 = cv2.imread('churchRight.png', 0)

# Loading images in color for linear blending
img1_color = cv2.imread('churchLeft.png')
img2_color = cv2.imread('churchRight.png')

# SIFT keypoints and descriptors
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Match descriptors
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Ratio test for filtering "good" matches
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])

# Extracting matching points from the keypoints
pts1 = np.zeros((len(good), 2), np.float32)
pts2 = np.zeros((len(good), 2), np.float32)
for i in range(len(good)):
    pts1[i] = kp1[good[i][0].queryIdx].pt
    pts2[i] = kp2[good[i][0].trainIdx].pt

# OpenCV Homography Matrix
H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

# Getting dimensions of the color images
h1, w1 = img1_color.shape[:2]
h2, w2 = img2_color.shape[:2]

# Defining the four corners of left image as homogenous coordinates
corners_img1 = np.array([
    [0, 0, 1], 
    [w1, 0, 1], 
    [0, h1, 1], 
    [w1, h1, 1]
], dtype=np.float32)

# Projecting the corners of left image using homography matrix
projected_corners = []
for pt in corners_img1:
    proj = H @ pt   # Applying the homography to each corner
    proj /= proj[2] # Normalizing to 2D coordinates
    projected_corners.append(proj[:2])

projected_corners = np.array(projected_corners)

# Defining the four corners of right image
corners_img2 = np.array([
    [0, 0], 
    [w2, 0], 
    [0, h2], 
    [w2, h2]
], dtype=np.float32)

# Calcuating the bounds for the panorama for the final image
all_x = np.concatenate((projected_corners[:, 0], corners_img2[:, 0]))
all_y = np.concatenate((projected_corners[:, 1], corners_img2[:, 1]))

# Finding the smallest and largest x and y values for determining the panorama's size
min_x = int(np.floor(np.min(all_x)))
max_x = int(np.ceil(np.max(all_x)))
min_y = int(np.floor(np.min(all_y)))
max_y = int(np.ceil(np.max(all_y)))

# Getting the width and height of the panorama
pan_width = max_x - min_x
pan_height = max_y - min_y
offset_x = -min_x   # Offsets for x and y coordinates
offset_y = -min_y

# Creating an empty image for the panorama and a weight mask for linear blending
panorama = np.zeros((pan_height, pan_width, 3), dtype=np.float32)
weight = np.zeros((pan_height, pan_width), dtype=np.float32)

# Pasting right image into the panorama
for y in range(h2):
    for x in range(w2):
        px = x + offset_x   # x-coordinates in panorama
        py = y + offset_y   # y-coordinates in panorama
        if 0 <= px < pan_width and 0 <= py < pan_height:
            panorama[py, px] = img2_color[y, x]     # Copying pixels to panorama
            weight[py, px] = 1.0

# Computing the overlap between both images based on the projected corners
right_start = offset_x
left_proj = (H @ np.array([w1, 0, 1])).flatten()    # Projecting the rightmost corner of the left image
left_proj /= left_proj[2]   # Normalizing to 2D coordinates
left_end = int(left_proj[0]) + offset_x     # Finding the x-coordinates of the left most corner in the panorama

# Defining overlapping region (where images overlap each other)
overlap_start = max(right_start, 0)
overlap_end = min(left_end, pan_width - 1)
blend_width = overlap_end - overlap_start if overlap_end > overlap_start else 1

# Warp img1 using the inverse homography matrix
H_inv = np.linalg.inv(H)
for y in range(pan_height):
    for x in range(pan_width):
        pt_pan = np.array([x - offset_x, y - offset_y, 1.0]) # Panorama coordinates
        pt_img1 = H_inv @ pt_pan    # Mapping back to left image space  
        pt_img1 /= pt_img1[2]       # Normalizing to 2D coordinates

        x1, y1 = pt_img1[0], pt_img1[1] # Getting x and y coordinates from left image
        x1_int = int(np.floor(x1))      # Converting to pixel coordinates to integers
        y1_int = int(np.floor(y1))

        # Checking if point is within the bounds of left image
        if 0 <= x1_int < w1 and 0 <= y1_int < h1:
            pixel = img1_color[y1_int, x1_int]
            if weight[y, x] > 0:
                # If pixel in panorama is already covered by right image, blend them together
                if overlap_start <= x <= overlap_end:
                    alpha = float(x - overlap_start) / blend_width  # Linear blending coefficient
                    panorama[y, x] = alpha * panorama[y, x] + (1 - alpha) * pixel   # Linear blending
                else:
                    panorama[y, x] = pixel  # Otherwise, copy the pixel from left image
            else:
                panorama[y, x] = pixel  # If right image is NOT covered, copy the pixel from the left image
            weight[y, x] += 1.0     # Setting a weight to mark the pixel as "covered"

# Convert to displayable format and save the final panorama
panorama = np.clip(panorama, 0, 255).astype(np.uint8)
#cv2.imwrite('image.png', panorama_uint8)
cv2.imshow('Panorama', panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()