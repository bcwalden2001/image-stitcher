## Summary

This program stitches a pair of images together where the 
images can have partial or complete overlap. 

The task involved detecting keypoints using the SIFT 
algorithm, estimating a homography matrix with Random Sample Consensus, warping one 
image into the other, and blending the overlapping regions using linear blending. The final 
result is a representation of both images combined together as a panorama.

## Methodology 

The process begins by loading the same pairs of images in both greyscale (for keypoint 
extraction) and color (for the final blending). The SIFT algorithm detects and 
computes keypoints and descriptors from the greyscale images and a brute-force matcher 
is used along with a ratio test to find reliable correspondences between the two images. 

Using the filtered matches, corresponding points were extracted and passed to OpenCV’s 
findHomography() function with the RANSAC algorithm to compute a robust homography 
matrix that best related the points of both images. This matrix represents the geometric 
transformation from one image to the other.  

The dimensions of the resulting panorama were calculated by transforming the corners of the 
first image using the homography and combining them with the bounds of the second image. 
An offset was introduced to ensure all projected points fit within the panorama dimensions. 

The right image was then directly copied into the panorama, and a weight matrix was used to 
track coverage. The left image was then warped into the panorama using the inverse 
homography matrix. Each pixel from the left image was mapped to its new position in the 
panorama using homogenous coordinates. Where both images overlapped, linear blending 
was then applied to smooth out seams that appeared after stitching. 

Finally, the panorama image was normalized to a range that could best be displayed in a 
window for visualization. 

## Results

<img width="534" height="507" alt="image" src="https://github.com/user-attachments/assets/bbd32aa3-5dfd-4ce6-a1db-24512343f1db" />

<img width="516" height="552" alt="image" src="https://github.com/user-attachments/assets/d2009053-5d62-4a17-8d40-c24266bf9220" />

<img width="528" height="527" alt="image" src="https://github.com/user-attachments/assets/f1147653-ac29-4a79-a0a3-aacdcc3ecd4e" />

<img width="598" height="577" alt="image" src="https://github.com/user-attachments/assets/c022fffe-7e90-46b6-99ee-743699b5711c" />

<img width="595" height="525" alt="image" src="https://github.com/user-attachments/assets/e2247fad-f501-4ccf-9fb4-58bb4cb30744" />

## Conclusion 

The image stitching process successfully demonstrates the fundamental concepts of feature 
matching, homography estimation, image stitching + warping, and linear blending. This program 
is adaptable to images with overlapping regions and can successfully produce a 
panorama with smooth edges as well as adjust to extreme brightness intensity differences between two images in a pair.
