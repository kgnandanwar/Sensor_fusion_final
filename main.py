import cv2
import numpy as np
import scipy.io as sio
from numpy import dot as dot
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

RANSAC_N = 1000
SIFT_SIZE = 7

from findmatch import *
from computeF import *
from tri import *
from other import *
from pose import *
from camerapose import *
from densematch import *
from epipolar import *


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('l.jpg', 1)
    img_right = cv2.imread('r.jpg', 1)
   # visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    visualize_find_match(img_left, img_right, pts1, pts2,2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    print(F)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    # K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    K = np.array([[3351.6, 0, 2016], [0, 3351.6, 1512], [0, 0, 1]])
    
    Rs, Cs = compute_camera_pose(F, K)
    print(Rs, Cs)
    visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    pts3Ds = []
    P1 = np.dot(K , np.hstack((np.eye(3), np.zeros((3, 1)))))
    for i in range(len(Rs)):
        P2 = np.dot(K , np.hstack((Rs[i], np.dot(-Rs[i],Cs[i]))))
        pts3D= triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    cv2.waitKey(0)
    cv2.destroyAllWindows()