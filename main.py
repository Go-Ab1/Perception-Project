import cv2 as cv
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.corner_detector import ShiTomasiCornerDetector
from utils.edge_detector import CannyEdgeDetector   
from utils.config import Config 
from utils.icp_reg import ICP
from typing import List, Tuple, Optional
import open3d as o3d

# Configuration class to store the configuration data
# ModelImage class to store and process the model image data
class ModelImage:
    def __init__(self, config):
        self.directory = config.directory
        self.image_path = config.image_path
        self.text_path = config.text_path
        self.timestamp = config.init_timestamp
        self.model_path = config.init_model_path
    
    # Method to read the model image
    def read_image(self):
        self.model_img = cv.imread(self.model_path, cv.IMREAD_GRAYSCALE)
        self.model_img = cv.resize(self.model_img, (420, 320)) 
        self.model_img = cv.cvtColor(self.model_img, cv.COLOR_GRAY2BGR)
        return self.model_img
    
    # Method to search for the intinsity image by timestamp
    def search_by_timestamp(self, timestamp):
        with open(self.text_path, 'r') as file:
            for line in file:
                line_timestamp, line_image_path = line.strip().split(' ', 1)
                if abs(float(line_timestamp) - float(timestamp)) < 1e-2:  # compare floats with precision
                    self.timestamp = line_timestamp
                    self.model_path = self.directory + line_image_path
                    break
                elif abs(float(line_timestamp) - float(timestamp)) < 1e-1:  # compare floats with higher tolerance
                    self.timestamp = line_timestamp
                    self.model_path = self.directory + line_image_path 
                    break
    
    # Method to get intinsity image by timestamp
    def get_by_timestamp(self, timestamp):
        self.timestamp = float(timestamp)
        self.search_by_timestamp(self.timestamp)
        return self

# Event class to store the event data
class Event:
    def __init__(self, timestamp, events_path):
        self.events_path = events_path
        self.timestamp = timestamp

    # Method to get the event image
    def get_image(self):
        self.events_img = cv.imread(self.events_path, cv.IMREAD_GRAYSCALE)
        self.events_img = cv.resize(self.events_img, (420, 320)) 
        self.rgb_event = cv.cvtColor(self.events_img, cv.COLOR_GRAY2BGR)
        return self.events_img, self.rgb_event

# Patch class to store the batch of events
class Patch:
    def __init__(self, center: List[Tuple[float, float]], size: Tuple[int, int],
                 events_points: List[Tuple[int, int]], model_points: List[Tuple[int, int]],
                 start_time: float, end_time: float):
        self.center = center
        self.size = size
        self.model_points = model_points
        self.events_points = events_points
        self.is_valid = True
        self.s_threshold = 10
        self.model_points_threshold = 10
        self.T = np.eye(4)
        self.start_time = start_time
        self.end_time = end_time
        self.patch_status()

    def __str__(self):
        return f"Center: {self.center}, Size: {self.size}, Events Points: {self.events_points}, Model Points: {self.model_points}"

    # Set the transformation matrix for the patch
    def set_transformation(self, T):
        self.T = T

    # Determine the status of the patch based on point thresholds
    def patch_status(self):
        if len(self.model_points) < self.model_points_threshold or len(self.events_points) < self.s_threshold:
            self.is_valid = False

    # Update event points and re-evaluate patch status
    def update_events_points(self, events_points):
        self.events_points = events_points
        self.patch_status()

    # Apply transformation to patch
    def transform_patch(self, event):
        points_to_be_aligned_3d = np.hstack((np.array(self.model_points), np.zeros((len(self.model_points), 1))))
        aligned_points = np.asarray(self.T @ np.concatenate((points_to_be_aligned_3d, np.ones((len(points_to_be_aligned_3d), 1))), axis=1).T).T[:, :2]
        self.model_points = aligned_points[:, :2]

        # Transform the center point
        center_3d = np.array([self.center[-1][0], self.center[-1][1], 0, 1])
        transformed_center = np.dot(self.T, center_3d)
        self.center.append(transformed_center[:2])
        self.end_time = event.timestamp

# EventBasedTracker class to handle tracking of features
class EventBasedTracker:
    def __init__(self, model_img, events):
        self.model_img = model_img
        self.events = events
        self.batch_size = 10
        self.edge_threshold = 10
        self.patch_threshold = 10
        self.initialize = True
        self.patches: List[Optional[Patch]] = []
        self.num_corners = 0

    # Main tracking function
    def track(self):
        self.img_with_corners, self.corners, self.edges, self.edge_pos = self.initialization(self.model_img)
        self.num_corners = len(self.corners)

        width, height = 420, 320
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter('event_tracking.avi', fourcc, 20.0, (width, height))
        time_list = []

        for event in self.events:
            event_img, rgb_event = event.get_image()

            if self.initialize:
                for index, corner in enumerate(self.corners):
                    cx, cy = corner        
                    x, y = int(cx), int(cy)
                    events_points = self.extract_s(event_img, x, y)
                    model_points = self.extract_model_points(self.edges, x, y)
                    self.patches.append(Patch(center=[(cx, cy)], size=(self.batch_size, self.batch_size),
                                              events_points=events_points, model_points=model_points,
                                              start_time=event.timestamp, end_time=event.timestamp))
                self.initialize = False
            else:
                for index, corner in enumerate(self.corners):
                    cx, cy = corner        
                    x, y = int(cx), int(cy)
                    events_points = self.extract_s(event_img, x, y)
                    self.patches[index].update_events_points(events_points)

            count = 0  # Counter for invalid patches
            time = event.timestamp
            model_image = self.model_img.get_by_timestamp(float(time) + (2 * 0.044065001))
            filename = model_image.model_path
            img = cv.imread(filename)
            img = cv.resize(img, (420, 320))
            timestamp_str = f"t = {float(event.timestamp):.2f}"
            cv.putText(img, timestamp_str, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
            for event_point in np.argwhere(rgb_event > 0):
                cv.circle(img, (event_point[1], event_point[0]), radius=1, color=(255, 255, 255), thickness=1)

            for index in range(len(self.patches)):
                if not self.patches[index].is_valid:
                    count += 1
                else:
                    patch = self.patches[index]
                    c = patch.center[-1]
                    points = [(int(c[0]), int(c[1])) for c in patch.center[-40:]]
                    cv.polylines(img, [np.array(points)], isClosed=False, color=(255, 0, 0), thickness=1)
                    cv.circle(img, (int(c[0]), int(c[1])), 1, (0, 0, 255), -1)
                    cv.rectangle(img, (int(c[0]) - self.batch_size, int(c[1]) - self.batch_size),
                                 (int(c[0]) + self.batch_size, int(c[1]) + self.batch_size), (0, 255, 0), 1)
            out.write(img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            average_time = 0
            if len(self.patches) - count < int(self.num_corners / 2): # Dynamic threshold for reinitialization
                
                print("Reinitialize Image ....")
                for patch in self.patches:
                    average_time += (float(patch.end_time) - float(patch.start_time))
                if average_time > 0:
                    print("Average Time", average_time / len(self.patches))
                    time_list.append(average_time / len(self.patches))
                    print("Time list", time_list)
                self.initialize = True
                self.patches = []
                self.img_with_corners, self.corners, self.edges, self.edge_pos = self.initialization(self.model_img, event.timestamp)

            self.apply_tracking(event)
        out.release()

    # Apply tracking to each patch
    def apply_tracking(self, event):
        for index, patch in enumerate(self.patches):
            if patch.is_valid:
                T = self.apply_icp(patch, event)
                self.corners[index] = patch.center[-1]

    # Apply ICP to align patches
    def apply_icp(self, patch, event):
        icp = ICP(patch.events_points, patch.model_points, patch.T)
        T = icp.transform_patch()
        patch.set_transformation(T)
        patch.transform_patch(event)
        return T

    # Extract event points in the patch area
    def extract_s(self, event_img, x, y):
        event_positions = np.argwhere(event_img[y - self.batch_size:y + self.batch_size, x - self.batch_size:x + self.batch_size] > 0)
        event_positions = [(pos[1] + x - self.batch_size, pos[0] + y - self.batch_size) for pos in event_positions]
        return list(set(event_positions))

    # Extract model points in the patch area
    def extract_model_points(self, edges, x, y):
        edge_positions = np.argwhere(edges[y - self.batch_size:y + self.batch_size, x - self.batch_size:x + self.batch_size] > 0)
        edge_positions = [(pos[1] + x - self.batch_size, pos[0] + y - self.batch_size) for pos in edge_positions]
        return edge_positions

    # Detect corners using Shi-Tomasi corner detector
    def detect_corners(self, model_image, max_corners=35, min_distance=15):
        filename = model_image.model_path
        #print("Detecting corners in", filename)
        corner_detector = ShiTomasiCornerDetector(filename, max_corners, min_distance)
        img_with_corners, corners = corner_detector.detect_corners()
        return img_with_corners, corners

    # Detect edges using Canny edge detector
    def detect_edges(self, model_image):
        filename = model_image.model_path
        #print("Detecting edges in", filename)
        edge_detector = CannyEdgeDetector(filename)
        edge_img, edge_pos = edge_detector.detect_edges()
        return edge_img, edge_pos

    # Initialize tracking process
    def initialization(self, model_img, timestamp=0.107328000):
        model_img = model_img.get_by_timestamp(float(timestamp) + (2 * 0.044065001))
        img_with_corners, corners = self.detect_corners(model_img)
        edge_img, edge_pos = self.detect_edges(model_img)
        return img_with_corners, corners, edge_img, edge_pos

# Function to create events from configuration
def create_events(config: Config):
    events: List[Optional[Event]] = []
    with open(config.events_image_txt, 'r') as file:
        for line in file:
            line_timestamp, line_image_path = line.strip().split(', ', 1)
            events.append(Event(line_timestamp, config.events_group_path + line_image_path))
    return events

if __name__ == "__main__":
    # Model data initialization
    config = Config()
    model_img = ModelImage(config=config)
    
    # Events data initialization
    events = create_events(config)  # Create events objects from grouped event images 
    tracker = EventBasedTracker(model_img, events)
    tracker.track()

# Wait for user input to close windows
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
