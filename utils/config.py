class Config:
    def __init__(self):
        self.directory = 'Dataset/'
        self.events_path_txt = self.directory+'events.txt'
        self.events_image_txt = self.directory+'event_images.txt'
        self.events_group_path = self.directory+'events_grouped_txt/'     
        self.event_path = self.events_group_path + 'group_0_binary_image.png'
        self.image_path =  self.directory+'images/'
        self.text_path = self.directory+'images.txt'
        self.init_timestamp = 0.151394000
        self.init_model_path = self.image_path+'frame_00000003.png'

   # def __init__(self, data_file="./Dataset/events.txt", events_per_group=700):     
    def __str__(self):
        return f"Directory: {self.directory}, Events Path Txt: {self.events_path_txt}, Events Image Txt: {self.events_image_txt}, Events Group Path: {self.events_group_path}, Event Path: {self.event_path}, Image Path: {self.image_path}, Text Path: {self.text_path}, Initial Timestamp: {self.init_timestamp}, Initial Model Path: {self.init_model_path}"

