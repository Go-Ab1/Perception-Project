
import cv2 as cv
import numpy as np
class ShiTomasiCornerDetector:
  # Class to detect corners in an image using the Shi-Tomasi corner detection algorithm.
    def __init__(self, filename, max_corners =20, min_distance =15):
        self.filename = filename
        self.max_corners = max_corners
        self.corners = []
        self.min_distance = min_distance

    def detect_corners(self):
        img = cv.imread(self.filename)
        img = cv.resize(img, (420, 320))  
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(gray, self.max_corners, 0.05, self.min_distance)
        corners = np.int0(corners)

        for i in corners:
            x, y = i.ravel()
            self.corners.append((x, y))
            cv.circle(img, (x, y), 1, (0,255,255), -1)

        return img, self.corners