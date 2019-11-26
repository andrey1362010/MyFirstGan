import cv2
import numpy as np
from skimage import transform as trans

src = np.array([
    [48.0252, 71.7366], #nose
    [33.5493, 92.3655], #left m
    [62.7299, 92.2041], #right m
    [30.2946, 51.6963], #left eye
    [65.5318, 51.5014], #right eye
    ], dtype=np.float32)
src[:, 0] += 8.0

def preprocess_face(img, landmark, image_size ):
    dst = landmark.astype(np.float32)
    #-----optimizing-----
    #todo надо ли
    #-------------------
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue = 0.0)
    return warped