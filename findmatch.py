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


def visualize_find_match(img1, img2, pts1, pts2, c):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if c==0:
        for i in range(pts1.shape[0]):
            plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'g.-', linewidth=0.5, markersize=5)
        plt.axis('off')
        plt.show()
    if c==1:
        for i in range(pts1.shape[0]):
            plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'r.-', linewidth=0.5, markersize=5)
        plt.axis('off')
        plt.show()    
    if c==2:
        for i in range(pts1.shape[0]):
            plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
        plt.axis('off')
        plt.show()

def find_match(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    im,x = [], [] 
    im.append(img1)
    im.append(img2)
    ##### do the matching first from 1st img to 2nd one, then the other side
    for j in range(2):  
        tem_kp, tem_des = sift.detectAndCompute(im[j], None)
        tar_kp, tar_des = sift.detectAndCompute(im[1-j], None)
       
        model = NearestNeighbors(n_neighbors=2).fit(tar_des)
        dist, indices = model.kneighbors(tem_des)
        u,v   =[], []
        uu,vv=[],[]
        for i in range(len(tem_kp)):
            point1 = tem_kp[i].pt
            point2 = tar_kp[indices[i][0]].pt
            d1, d2 = dist[i]
            if (d1 / d2) <= 0.5 :
                u.append(point1)
                v.append(point2)
            uu.append(point1)
            vv.append(point2)
        uu=uu[::8]
        vv=vv[::8]    
        u,v  = np.asarray(u), np.asarray(v)
        uu,vv  = np.asarray(uu), np.asarray(vv)
        x.append(u)
        x.append(v)
       
        visualize_find_match(im[j], im[1-j], uu, vv, j)
        visualize_find_match(im[j], im[1-j], u, v, j)
        print('the total features are :{},  and {} after  filtering with a Ratio {} '.format(len(tem_kp),len(u), 0.7))
       
    x1_fo, x2_fo, x1_ba, x2_ba =x[0],x[1],x[2],x[3]

    f_dict = {}
    for x1, x2 in zip(x1_fo,x2_fo):
        f_dict[tuple(x1)] = tuple(x2)

    b_dict = {}
    for x1, x2 in zip( x2_ba , x1_ba):
        b_dict[tuple(x2)] = tuple(x1)

    x1_f, x2_f = [], [] 
    for x1, x2 in zip( x1_fo , x2_fo ):
        try:
            if b_dict[f_dict[tuple(x1)]] == tuple(x1):
                x1_f.append(x1)
                x2_f.append(x2)
        except KeyError:
            pass
    x1_f, x2_f = np.asarray(x1_f), np.asarray(x2_f)       
    x1 , x2 =x1_f , x2_f
    print('{} SIFT feature matches with bi-directional check'.format(len(x1)))

    return x1, x2