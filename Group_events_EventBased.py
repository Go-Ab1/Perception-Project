#!/usr/bin/env python3

import os
import cv2
import numpy as np
import concurrent.futures

class EventsGroup:
    def __init__(self, data_file="./Dataset/events.txt", events_per_group=100):
        
        self.data_file = data_file
        self.events_per_group = events_per_group
        self.grouped_events = []
        self.max_x = 0
        self.max_y = 0
        self.num_groups = 0
        self.output_directory = "events_grouped_txt"
    
    def read_data(self):
    
        data = []
        with open(self.data_file, "r") as file:
            for line in file:
                timestamp, x, y, polarity = map(float, line.strip().split())
                data.append((timestamp, x, y, polarity))
        return data


    def calculate_groups(self, data):
        """
        Calculates the number of groups based on the total number of events and the events per group.
        """
        num_events = len(data)
        self.num_groups = (num_events + self.events_per_group - 1) // self.events_per_group

    def find_max_coordinates(self, data):
        """
        Finds the maximum x and y coordinates from the event data.
        """
        max_x = max(data, key=lambda x: x[1])[1]
        max_y = max(data, key=lambda x: x[2])[2]
        return max_x, max_y

    def group_events(self, data):
        """
        Groups the events based on the events per group and returns a list of grouped events.
        """
        grouped_events = [[] for _ in range(self.num_groups)]
        data.sort(key=lambda x: x[0])  # Sort based on timestamp
        for event_index, event in enumerate(data):
            group_index = event_index // self.events_per_group
            grouped_events[group_index].append(event)
        return grouped_events

    def create_output_directory(self):
        """
        Creates the output directory if it doesn't exist.
        """
        os.makedirs(self.output_directory, exist_ok=True)
  
        
    def save_image(self, group_index, group_events, first_event_timestamp):
        """
        Saves a single grouped event as a binary image and writes timestamp and image filename to a text file.
        """
        self.max_x = int(self.max_x)
        self.max_y = int(self.max_y)

        img = np.zeros((self.max_y + 1, self.max_x + 1), dtype=np.uint8)
        for event in group_events:
            x, y = event[1:3]
            img[int(y), int(x)] = 255  

        timestamp_str = f"{first_event_timestamp:.6f}".replace('.', '_')  
        image_path = os.path.join(self.output_directory, f"group_{group_index}_binary_image_{timestamp_str}.png")
        cv2.imwrite(image_path, img)

        
        with open(os.path.join(self.output_directory, "timestamps_and_imagename.txt"), "a") as file:
            file.write(f"{first_event_timestamp}, group_{group_index}_binary_image_{timestamp_str}.png\n")

  

# Main
if __name__ == "__main__":
    events_group = EventsGroup()
    data = events_group.read_data()
    events_group.calculate_groups(data)
    events_group.max_x, events_group.max_y = events_group.find_max_coordinates(data)
    events_group.grouped_events = events_group.group_events(data)
    events_group.create_output_directory()
    

    for group_index, group_events in enumerate(events_group.grouped_events):
        first_event_timestamp = group_events[0][0]  # Timestamp of the first event in the group
        events_group.save_image(group_index, group_events, first_event_timestamp)