#!/usr/bin/env python

import rospy

# Import some ardrone services
from ardrone_autonomy.srv import CamSelect, RecordEnable, LedAnim

# Import some file-operation functions
import os

# Import some C array and OpenCV stuff
import numpy as np
import cv2


class TakePhoto:
    
    def __init__(self, video_display, image_folder_front, image_folder_bottom, takephoto_live, takephoto_usb):
        self.video_display = video_display
        for i, folder in enumerate((image_folder_front, image_folder_bottom)):
            if not os.path.exists(folder):
                os.mkdir(folder)
            print "I will use folder '%s' for outputting %s cam images." % (os.path.abspath(folder), ("front", "bottom")[i])
        self.image_folders = (image_folder_front, image_folder_bottom)
        self.takephoto_live = takephoto_live
        self.takephoto_usb = takephoto_usb
        
        # initialize internal vars
        self.take_photo = 0
        self.cam_id = 0
        self.image_counter = 0
        
        # initialize services
        if takephoto_live: self.cam_selector = rospy.ServiceProxy('/ardrone/setcamchannel', CamSelect, persistent=True)
        if takephoto_usb: self.recorder = rospy.ServiceProxy('/ardrone/setrecord', RecordEnable, persistent=True)
        #self.led_animator = rospy.ServiceProxy('/ardrone/setledanimation', LedAnim, persistent=True)
    
    def save(self):
        frame_id = ("front", "bottom")[self.cam_id]
        rospy.loginfo("=> %s camera" % frame_id)
        
        if self.takephoto_usb:
            try:
                resp = self.recorder(enable=self.take_photo)
                if not resp: rospy.loginfo("recorder failed with response: %s" % resp)
            except Exception as exc:
                rospy.loginfo("recorder did not process request: %s" % exc)
        
        if self.takephoto_live:
            img = np.array(buffer(self.video_display.image.data), dtype=np.uint8).reshape(self.video_display.image.height, self.video_display.image.width, 3)
            cv2.imwrite(os.path.join(self.image_folders[self.cam_id], "%s-%03d.jpg" % (frame_id, self.image_counter)), img)
    
    def handle(self, take_photo_new):
        if take_photo_new == self.take_photo:
            return
        
        self.take_photo = take_photo_new
        if self.take_photo:
            rospy.loginfo("Taking image...")
        self.save()
        
        if self.take_photo:
            #self.led_animator(type=3, freq=4., duration=1)
            self.cam_id = 1 - self.cam_id
            try:
                resp = self.cam_selector(channel=self.cam_id)
                if not resp: rospy.loginfo("cam_selector failed with response: %s" % resp)
            except Exception as exc:
                rospy.loginfo("cam_selector did not process request: %s" % exc)
        else:
            self.image_counter += 1
    
    def close(self):
        if self.takephoto_live: self.cam_selector.close()
        if self.takephoto_usb: self.recorder.close()
        #self.led_animator.close()
