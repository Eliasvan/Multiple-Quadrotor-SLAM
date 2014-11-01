#!/usr/bin/env python

import rospy

# Import the some messages we're interested in
from sensor_msgs.msg import Image        # for receiving the video feed

# Import some ardrone services
from ardrone_autonomy.srv import LedAnim

# Import some file-operation functions
import os

# Import some C array and OpenCV stuff
import numpy as np
import cv2


class TakeVideo:
    
    def __init__(self, video_display, image_folder, takevideo_max_memory):
        self.image_width, self.image_height, self.image_channels = 640, 360, 3
        
        self.video_display = video_display
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)
        rospy.loginfo("I will use folder '%s' for outputting cam images." % os.path.abspath(image_folder))
        self.image_folder = image_folder
        
        # initialize internal vars
        self.takevideo_button = 0
        self.recording = False
        self.image_counter = 0
        
        # setup messages and services
        video_display.subVideo = rospy.Subscriber('/ardrone/image_raw', Image, self.ReceiveImage)
        self.led_animator = rospy.ServiceProxy('/ardrone/setledanimation', LedAnim, persistent=True)
        
        # allocate memory for image sequence
        self.image_size = self.image_width * self.image_height * self.image_channels
        self.max_num_images = takevideo_max_memory // self.image_size
        rospy.loginfo("Maximum number of images to be captured: %s" % self.max_num_images)
        self.takevideo_max_memory = self.max_num_images * self.image_size
        rospy.loginfo("Allocation memory for image sequence...")
        self.data = np.ones(self.takevideo_max_memory, dtype=np.uint8)
        rospy.loginfo("Done.")
        self.data_pointer = 0
        self.data_timestamps = np.ones(self.max_num_images)
        self.play_leds_anim(True)    # signal ready
    
    def play_leds_anim(self, status):
        animation = 1 if status else 3    # ORANGE for status==False, otherwise GREEN
        freq = 4. if status else 8.
        try:
            resp = self.led_animator(type=animation, freq=freq, duration=1)
            if not resp: rospy.loginfo("led_animator failed with response: %s" % resp)
        except Exception as exc:
            rospy.loginfo("led_animator did not process request: %s" % exc)
    
    def ReceiveImage(self, image):
        if self.recording and self.data_pointer < self.takevideo_max_memory:
            self.data[self.data_pointer:self.data_pointer + self.image_size] = buffer(image.data)
            self.data_pointer += self.image_size
            self.data_timestamps[self.image_counter] = image.header.stamp.to_time()
            self.image_counter += 1
            
            if self.data_pointer >= self.takevideo_max_memory:
                self.video_display.subVideo = rospy.Subscriber('/ardrone/image_raw', Image, self.video_display.ReceiveImage)
                self.recording = False
                self.play_leds_anim(False)    # signal end of max memory
        
        self.video_display.ReceiveImage(image)
    
    def handle(self, take_video_button_new):
        if take_video_button_new == self.takevideo_button:
            return
        
        self.takevideo_button = take_video_button_new
        if not self.takevideo_button:
            return
        
        if self.recording:
            self.recording = False
        elif self.data_pointer < self.takevideo_max_memory:
            self.recording = True
        
        if self.recording:
            rospy.loginfo("Started/resumed recording.")
            self.play_leds_anim(True)    # signal start
        else:
            if self.data_pointer < self.takevideo_max_memory:
                rospy.loginfo("Stopped/paused recording.")
            else:
                rospy.loginfo("Reached max memory limit, record stopped.")
            self.play_leds_anim(False)    # signal end
    
    def close(self):
        if self.image_counter:
            # Save timestamps to disk
            timestamps = '\n'.join(map(str, self.data_timestamps[:self.image_counter])) + '\n'
            open(os.path.join(self.image_folder, "timestamps.txt"), 'w').write(timestamps)
            rospy.loginfo("Wrote timestamps.")
            
            # Save image sequence to disk
            from math import log10
            format_string = "%%0%sd" % (int(log10(self.max_num_images)) + 1)
            rospy.loginfo("Saving images...")
            for i, img in enumerate(self.data.reshape(self.max_num_images, self.image_height, self.image_width, self.image_channels)[:self.image_counter]):
                cv2.imwrite(os.path.join(self.image_folder, "image-%s.jpg" % (format_string % i)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            rospy.loginfo("Done.")
        
        else:
            rospy.loginfo("Nothing recorded.")
        
        # Free memory
        del self.data
        self.data = None
        
        self.led_animator.close()
