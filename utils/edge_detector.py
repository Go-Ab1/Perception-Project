
import cv2 as cv
import numpy as np
class CannyEdgeDetector:
   # Class to detect edges in an image using the Canny edge detection algorithm.
    def __init__(self, filename):
        self.filename = filename

    def detect_edges(self):
        img = cv.imread(self.filename, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (420, 320))    
        edges = cv.Canny(img, 30, 150)
        edges_pos = np.argwhere(edges > 0)
        return edges, edges_pos
 