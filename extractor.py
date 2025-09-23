from socket import IPV6_RTHDR
import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

# Extraction of the camera pose from the fundamental matrix
def extractPose(F):
    W  = np.array([[0,-1,0],[1,0,0],[0,0,1]])
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

class Matcher(object):
    def __init__(self):
        self.last = None

# define a function responsible of matching frames features
def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(f1.des, f2.des, k=2)

    # Lowe's ratio test
    ret =[]
    idx1, idx2 = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]

            # Make distance test to ensure that the euclidian distance
            # between p1 and p2 is less than 0.1

            if np.linalg.norm(p1, p2) < 0.1:
                # keep idxs
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                ret.append((p1, p2))

                pass
    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # Fit matrix
    model, inliers = ransac((ret[:, 0],
                             ret[:, 1]), FundamentalMatrixTransform,
                             min_samples=8, residual_threshold=0.005,
                             max_trials=2000)
    # Ignore outliers

    ret = ret[inliers]
    Rt =  extractPose(model.params)

    return idx1[inliers], idx2[inliers], Rt

class Frame(object):
    def __init__(self, mapp, img, K):
        self.K = K
        self.kinv = np.linalg.inv(self.K)
        self.pose = np.eye(4)

        self.id = len(mapp.frames)
        mapp.frames.append(self)

        pts, self.des = extract(img)

        if self.des is not None and len(self.des) > 0:
            self.pts = normalize(self.kinv, pts)
        else:
            self.pts = np.array([])


