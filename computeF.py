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

def compute_F(pts1, pts2):
    assert pts1.shape == pts2.shape
    n, nn = pts1.shape
    indices = np.arange(n)
    min_loss = None
    best_F = None
    for nn in range(1500):
        np.random.shuffle(indices)
        first_eight_indices = indices[:8]
        # Compute tentative F using null space of 8 points matrix
        ps1,ps2= pts1[first_eight_indices], pts2[first_eight_indices]
        assert ps1.shape == ps2.shape == (8, 2)
        A = np.zeros((8, 9))
        for i, (u, v) in enumerate(zip(ps1, ps2)):
            # print(i, u, v)
            A[i, 0], A[i, 1], A[i, 2], A[i, 3],= u[0] * v[0],u[1] * v[0],v[0], u[0] * v[1]
            A[i, 4] , A[i, 5], A[i, 6], A[i, 7], A[i, 8]= u[1] * v[1], v[1],u[0],u[1],1
        F = null_space(A)
        # Take only the first solution to null space
        F = F[:, 0]
        F_tentative = F.reshape(3, 3)
        #F_tentative = compute_F_by_8_point_algo(pts1[first_eight_indices], pts2[first_eight_indices])
        # Do SVD cleanup

       # F_cleaned = do_svd_cleapup(F_tentative)
        u, d, vt = np.linalg.svd(F_tentative)
        d[2] = 0
        F_cleaned = np.dot(u * d, vt)

        loss = []
        for pt1, pt2 in zip(pts1, pts2):
            u1 = np.asarray([pt1[0], pt1[1], 1])
            v1 = np.asarray([pt2[0], pt2[1], 1])
            per_point_loss = np.dot(np.matmul(v1, F_cleaned), u1)
            # print(v, F_cleaned, u, per_point_loss)
            loss.append(per_point_loss)
        loss = np.asarray(loss)
        loss = np.sum(loss ** 2)

        # Compute loss
        #loss = compute_RANSAC_loss(pts1, pts2, F_cleaned)
        # print(loss)
        if min_loss is None or loss < min_loss:
            min_loss = loss
            best_F = F_cleaned
    print('Min loss = {} for RANSAC iterations = {}'.format(min_loss, RANSAC_N))
    #print('Best fundamental matrix = {}'.format(best_F))
    return best_F