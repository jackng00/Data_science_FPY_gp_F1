import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import csv
import logging


def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


IRt = np.eye(4)

def extractPose(F):
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]])
    U,d,Vt = np.linalg.svd(F)
    assert np.linalg.det(U) > 0
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    print(d)
    return ret

def extract(img):
    orb = cv2.ORB_create()
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detection
    pts = cv2.goodFeaturesToTrack(gray_img, 8000, qualityLevel=0.001, minDistance=10)

    if pts is None:
        return np.array([]), None

    # Extraction
    kps = [cv2.KeyPoint(f[0][0], f[0][1], 20) for f in pts]
    kps, des = orb.compute(gray_img, kps)

    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def normalize(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]
    # `[:, 0:2]` selects the first two columns of the resulting array, which are the normalized x and y coordinates.
    # `.T` transposes the result back to N x 3.


def denormalize(K, pt):
    ret = np.dot(K, [pt[0], pt[1], 1.0])
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))


class Matcher(object):
    def __init__(self):
        self.last = None


def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    # Lowe's ratio test
    ret = []
    idx1, idx2 = [], []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]
            
            # Distance test
            # dditional distance test, ensuring that the 
            # Euclidean distance between p1 and p2 is less than 0.1
            if np.linalg.norm((p1-p2)) < 0.1:
                # Keep idxs
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                ret.append((p1, p2))
                pass


    assert len(ret) >= 2
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # Fit matrix
    model, inliers = ransac((ret[:, 0], 
                            ret[:, 1]), FundamentalMatrixTransform, 
                            min_samples=8, residual_threshold=0.005, 
                            max_trials=200)
    
    # Ignore outliers
    ret = ret[inliers]
    Rt = extractPose(model.params)

    return idx1[inliers], idx2[inliers], Rt


class Frame(object):
    def __init__(self, mapp, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.pose = IRt

        self.id = len(mapp.frames)
        mapp.frames.append(self)

        pts, self.des = extract(img)
        
        if self.des.any()!=None:
            self.pts = normalize(self.Kinv, pts)


class Map(object):
    def __init__(self):
        self.frames = [] # camera frames [means camera pose]
        self.points = [] # 3D points of map
        self.state = None # variable to hold current state of the map and cam pose
        self.q = None # A queue for inter-process communication. | q for visualization process
        self.q_image = None
    
    
    def export_to_csv(self, filename="pointdata.csv"):
        """
        Exports all 3D points to a CSV file without a header row.
        
        Args:
            filename (str): The name/path of the CSV file to save.
        """
        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                #writer.writerow(['x', 'y', 'z'])  # Header
                # Removed the header row
                for point in self.points:
                    x, y, z = point.pt[:3]  # Extract x, y, z
                    writer.writerow([x, y, z])
            logging.info(f"Point data successfully exported to {filename} without headers.")
        except Exception as e:
            logging.error(f"Failed to export point data to CSV: {e}")


class Point(object):
    # A Point is a 3-D point in the world
    # Each point is observed in multiple frames

    def __init__(self, mapp, loc):
        self.frames = []
        self.pt = loc
        self.idxs = []

        # assigns a unique ID to the point based on the current number of points in the map.
        self.id = len(mapp.points)
        # adds the point instance to the map’s list of points.
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        # Frame is the frame class
        self.frames.append(frame)
        self.idxs.append(idx)