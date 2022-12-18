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

def triangulation(P1, P2, pts1, pts2):
    pts3D = []
    for pt1, pt2 in zip(pts1, pts2):
        pt1_3d = list(pt1) + [1]
        pt2_3d = list(pt2) + [1]
        pt1_skew_symmteric =np.asarray([
            [0, -pt1_3d[2], pt1_3d[1]],
            [pt1_3d[2], 0, -pt1_3d[0]],
            [-pt1_3d[1], pt1_3d[0], 0], ])

        pt2_skew_symmteric =np.asarray([
            [0, -pt2_3d[2], pt2_3d[1]],
            [pt2_3d[2], 0, -pt2_3d[0]],
            [-pt2_3d[1], pt2_3d[0], 0], ])

        pt1_cross_P1 =np.dot(pt1_skew_symmteric, P1)
        pt2_cross_P2 =np.dot(pt2_skew_symmteric, P2)
        A = np.vstack((pt1_cross_P1[:2], pt2_cross_P2[:2]))
        X = null_space(A, rcond=1e-1)
        # Take the first null space entry
        X = X[:, 0]
        # Divide by w
        X = X / X[3]
        pts3D.append(X[:3])
    pts3D = np.asarray(pts3D)
    return pts3D