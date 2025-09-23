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

# ORB (Oriented FAST Rotated BRIEF) feature extraction for real-time purpose
def extract(img):

    orb = cv2.ORB_create()

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detection of features

    points = cv2.goodFeaturesToTrack(gray_img, 8000, qualityLevel=0.01, minDistance=10)
    if points is None:
        return np.array([]), None
    # Features description

    kpts = [cv2.KeyPoint(f[0][0], f[0][1], 20) for f in points]
    kpts, des = orb.compute(gray_img, kpts)
    
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kpts]), des

def add_ones(x):
    # Convert Nx2 array to Nx3 array where each point is represented in
    # Homogeneous coordinates as [x, y, 1]
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis =1)

def normalize(Kinv, pts):
    # Convert pixel coordinates to normalized image coordinates 
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalize(K, pt):
    ret = np.dot(K, [pt[0], pt[1], 1.0])
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))
