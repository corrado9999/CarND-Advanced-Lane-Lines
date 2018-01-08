import os
import pickle
import hashlib
import numpy as np
import cv2

def get_camera_calibration_params(filenames, chessboard_shape=(9,6)):
    """Given a list of calibration images, extract calibration parameters.
    Inspired to https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((np.product(chessboard_shape), 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for fname in filenames:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_shape, None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return {'dist': dist, 'mtx': mtx}

def undistort(image, params):
    return cv2.undistort(image, params['mtx'], params['dist'], None, params['mtx'])

class Calibration(object):
    """Class for computing, caching and applying calibration parameters."""
    def __init__(self, filenames, chessboard_shape=(9,6), cache_dir=os.path.curdir):
        hash = hashlib.md5(str(sorted(filenames)).encode()).hexdigest()
        self.cache = os.path.join(cache_dir, "calibration-%s.P" % hash)
        try:
            with open(self.cache, 'rb') as f:
                self.params = pickle.load(f)
        except:
            self.params = None

        if not self.params:
            self.params = get_camera_calibration_params(filenames, chessboard_shape)
            with open(self.cache, 'wb') as f:
                pickle.dump(self.params, f)

    def __call__(self, image):
        return undistort(image, self.params)

class PerspectiveTransformation(object):
    def __init__(self, corners, dest_corners=None, input_shape=None, output_shape=None):
        src = np.array(corners, dtype='float32')
        if dest_corners is not None:
            dst = np.array(dest_corners, dtype='float32')
        else:
            # define destination points as the bounding box
            dst = np.array([(src[...,0].min(), src[...,1].max()),
                            (src[...,0].min(), 0),
                            (src[...,0].max(), 0),
                            (src[...,0].max(), src[...,1].max())],
                           dtype='float32')
        # get the transform matrix
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.source_corners = src
        self.dest_corners = dst
        self._inverse = None
        self.input_shape = input_shape
        self.output_shape = output_shape

    @property
    def inverse(self):
        if self._inverse is None:
            self._inverse = PerspectiveTransformation(self.dest_corners, self.source_corners,
                                                      self.output_shape, self.input_shape)
            self._inverse._inverse = self
        return self._inverse

    def __call__(self, img, y=None):
        """Apply perspective transformation to an image or to points.
        If y is None, warp the image img. Otherwise, apply transformation to
        points with coordinates img,y.
        """
        if y is None:
            return cv2.warpPerspective(img, self.M, (self.output_shape or img.shape)[1::-1])
        else:
            x, y, w = np.broadcast_arrays(img, y, 1)
            x, y, w = np.dot(self.M, np.stack((x, y, w)))
            return x/w, y/w
