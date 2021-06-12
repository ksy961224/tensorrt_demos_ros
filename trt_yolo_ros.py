#!/usr/bin/env python3
import rospy
from std_msgs.msg import Header
from std_msgs.msg import String


from yolonano_msg.msg import yolonano, distmsg
import numpy as np

# ----------------------------

import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import get_input_shape, TrtYOLO
from module.px2world import bbox_conversion


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()

    

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        
    
        if img is None:
            break
        
        # print(img.shape[:2])
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        cam.img_width : 640
        cam.img_height : 480
     
        img= vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)

        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        yoloros=yolo_ros(boxes, confs, clss)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
        # if rospy.ROSInterruptException:
        #     break



def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (./yolo/%s.trt) not found!' % args.model)
    
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    h, w = get_input_shape(args.model)
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num, args.letter_box)

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        640, 480)
   
    loop_and_detect(cam, trt_yolo, conf_th=0.6, vis=vis)

    cam.release()
    cv2.destroyAllWindows()

class yolo_ros:
    def __init__(self,boxes,confs,clss):
        self.pub=rospy.Publisher('/yolo_nano_0', yolonano, queue_size=10)
        self.pub1=rospy.Publisher('/dist', distmsg, queue_size=1)
        self.pubmsg=yolonano()
        self.pubmsg1=distmsg()
        self.boxes=boxes
        self.clss=clss
        self.confs=confs
        # print(self.boxes)
        rospy.init_node('ros_yolo_nano_0', anonymous=True)
        self.rate=rospy.Rate(15)
        
        self.pubval_xmin=[]
        self.pubval_xmax=[]
        self.pubval_ymin=[]
        self.pubval_ymax=[]
        self.pubval_conf=[]
        self.pubval_clss=[]

        self.y_array=[]
        self.tar_x=0; self.tar_y=0; self.tar_z=0
        self.callback()
        
    
    def callback(self):
       
        idx_array=[]
        x_array=[];y_array=[]
        for i in range(len(self.boxes)):
            if len(self.boxes)>0:
                tar_flag=1
                self.pubval_ymin.append(np.uint32(self.boxes[i][1]))
                self.pubval_ymax.append(np.uint32(self.boxes[i][3]))
                self.pubval_xmin.append(np.uint32(self.boxes[i][0]))
                self.pubval_xmax.append(np.uint32(self.boxes[i][2]))
                self.pubval_conf.append(np.float32(self.confs[i]))
                self.pubval_clss.append(np.uint32(self.clss[i]))
                coord=bbox_conversion(self.pubval_xmin[i],self.pubval_xmax[i],self.pubval_ymin[i],self.pubval_ymax[i])
                cx=(self.pubval_xmin[i]+self.pubval_xmax[i])/2
                if self.pubval_xmin[i]<0.1*640 and self.pubval_xmax[i]<0.4*640:
                    tar_flag=0
                elif self.pubval_xmin[i]>0.6*640 and self.pubval_xmax[i]>0.9*640:
                    tar_flag=0

                if self.pubval_xmin[i]<0.3*640 and self.pubval_xmax[i]<0.35*640:
                    tar_flag=0
                elif self.pubval_xmin[i]>0.6*640 and self.pubval_xmax[i]>0.65*640:
                    tar_flag=0

                if abs(coord[1])<2 and tar_flag==1:
                    idx_array.append(i)
                    x_array.append(coord[0]); y_array.append(coord[1])
                    
                    
                
            
        if len(idx_array)==0:
            tar_x=100;tar_y=100;tar_z=100
        else:
            target_idx=x_array.index(min(x_array))
            tar_x=x_array[target_idx]
            tar_y=y_array[target_idx]

            
            
        # print(tar_x,tar_y)
        self.pubmsg1.tar_x=tar_x
        self.pubmsg1.tar_y=tar_y


        # self.pubmsg.header = 0
        self.pubmsg.bbox_ymin = self.pubval_ymin
        self.pubmsg.bbox_xmin = self.pubval_ymax
        self.pubmsg.bbox_xmax = self.pubval_xmin
        self.pubmsg.bbox_ymax = self.pubval_xmax
        self.pubmsg.classidx  = self.pubval_clss
        self.pubmsg.conf      = self.pubval_conf
        # self.pubmsg.conf = 0
        # self.pubmsg.clss = 0

        self.pub.publish(self.pubmsg)
        self.pub1.publish(self.pubmsg1)
        # rospy.loginfo(self.pubmsg)
        
        



if __name__ == '__main__':
    

    filedir=__file__


    ridx=filedir.rfind('/')
    curdir=filedir[:ridx]
    os.chdir(curdir)
    

    WINDOW_NAME = 'TrtYOLODemo'

    
    
    main()
    # try:
        
        
    # except :
    #     pass
