import cv2
import numpy as np

def dense_match(img1, img2):
    assert img1.shape == img2.shape
    sift = cv2.xfeatures2d.SIFT_create() 
    im=[]
    im.append(img1)
    im.append(img2)
    desc=[]
    for b in range(2):
        h, w = im[b].shape
        kp = []
        for i in range(h):
            for j in range(w):
                kp.append(cv2.KeyPoint(x=j, y=i, _size=7))  ##make pixels as key-points
        kps, des = sift.compute(im[b], kp)   ## compute descreptors
        des = np.asarray(des).reshape((h, w, 128))
        desc.append(des)

    dense1 = desc[0]
    dense2 = desc[1]
    disparity = np.ones(img1.shape) ##initializing disparity
    h, w = img1.shape

    for i in range(h):
        for j in range( w):
            if img1[i, j] == 0:
                continue    ##ignoring background
            d1_d2_dists = []
            d1 = dense1[i, j]  ## 1st point's descriptor
            for k in range(0, j + 1): ##slipping on a row
                d2 = dense2[i, k]  
                d1_d2_dists.append(np.linalg.norm(d1 - d2))
            disparity[i, j] = np.abs(np.argmin(d1_d2_dists) - j)
    return disparity