import cv2
import numpy as np

def compute_camera_pose(F, K):
    #E = K.T @ F @ K
    E = np.dot(np.dot(K.T , F) , K)
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        #Cs.append(-Rs[i].T @ ts[i])
        Cs.append(np.dot(-Rs[i].T , ts[i]))

    return Rs, Cs