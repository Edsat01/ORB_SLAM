import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

# Extraction of the camera pose from the fundamental matrix
def extractPose(F):
    W  = np.mat([[0,-1,0],[1,0,0],[0,0,1]])
    U, d, Vt = np.linalg.svd(F)
    assert np.linalg.det(U) > 0

    if np.linalg.det(Vt) < 0:
        Vt *= -1
    
    R = np.dot(np.dot(U, W), Vt) # according to Hartley & Zisserman

    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    print(d)

    return ret