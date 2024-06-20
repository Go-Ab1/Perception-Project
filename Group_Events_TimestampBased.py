#!/usr/bin/env python3

#But the problem here is that the edges are not as good as the event based approach.

import os   
import cv2
import numpy as np

class EventsGroup:
    
    def __init__(self, data_file="./Dataset/events.txt", time_interval=(0.044065001/4)): # 0.001663 time to get the same number of groups as the event based approach 
        self.data_file = data_file
        self.time_interval = time_interval  
        self.data = []
        self.grouped_events = []
        self.max_x = 0
        self.max_y = 0
        self.num_groups = 0
        self.output_directory = "group_images_timestamp"
        
  
                
    def read_data(self):
      """
      Reads the event data from the data file and stores it in the `data` attribute.
      """
      with open(self.data_file, "r") as file:
        #  print("shape file", file.shape)
        #  for _ in range(106):
        #     next(file)
         
         for line in file:
             
               timestamp, x, y, polarity = line.strip().split()
               timestamp = float(timestamp)  # Convert timestamp to float
               x = int(x)
               y = int(y)
               polarity = int(polarity)
               self.data.append((timestamp, x, y, polarity))

                
    def calculate_groups(self):
        """
        Calculates the number of groups based on the total number of events and the time interval.
        """
        self.data.sort(key=lambda x: x[0])
        start_time = self.data[0][0]
        end_time = self.data[-1][0]
        self.num_groups = int((end_time - start_time) / self.time_interval) + 1
        
    def find_max_coordinates(self):
        """
        Finds the maximum x and y coordinates from the event data.
        """
        self.max_x = max(self.data, key=lambda x: x[1])[1] 
        self.max_y = max(self.data, key=lambda x: x[2])[2]

    def group_events(self):
        """
        Groups the events based on the time interval and stores them in the `grouped_events` attribute.
        """
        self.grouped_events = [[] for _ in range(self.num_groups)]
        for event in self.data:
            group_index = int((event[0] - self.data[0][0]) / self.time_interval)
            self.grouped_events[group_index].append(event)

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
    events_group.read_data()
    events_group.calculate_groups()
    events_group.find_max_coordinates()
    events_group.group_events()
    events_group.create_output_directory()
    

    for group_index, group_events in enumerate(events_group.grouped_events):
        first_event_timestamp = group_events[0][0]  # Timestamp of the first event in the group
        events_group.save_image(group_index, group_events, first_event_timestamp)