#!/usr/bin/env python2

'''
Ajou Univ ACL Seo Bomin
bominseo@ajou.ac.kr

Convert image (2D) to world coordinate (3D)
'''

import numpy as np
import math
import os
# import cv2

# import cv_bridge
import time


# class pixel2world:

#     def __init__(self):

#         self.probability_thres = 0.81

#         rospy.init_node('px_2_world')

#         box_sub_topic       = '/darknet_ros/bounding_boxes'
#         # objectCount_topic = '/darknet_ros/found_object'
#         image_topic         = '/vds_node_localhost_2211/image_raw'
#         darknet_image_topic = '/darknet_ros/detection_image'

#         bbox_sub           = rospy.Subscriber(box_sub_topic,BoundingBoxes,self.callbackBBOX)
#         # objectCount_sub    = rospy.Subscriber(objectCount_topic,  ObjectCount,  self.callbackObjectCount)
#         image_sub          = rospy.Subscriber(image_topic,Image,self.callbackImage)
#         darknet_image_sub  = rospy.Subscriber(darknet_image_topic, Image, self.callbackDarkent)


#         self.bboxes_3d_pub           = rospy.Publisher('/vision_bbox',bboxes_3d,queue_size=1)

#         self.traffic_light_state_pub = rospy.Publisher('/traffic_light_state',traffic_light_state,queue_size=1)
#         self.traffic_sign_state_pub = rospy.Publisher('/traffic_sign_state',traffic_light_state,queue_size=1)

#         self.darknetFlag = False
        

# ############################## preprocessing ############################

#     def getNearestBbox(self, bboxes):

#         # bboxes = iter(bboxes)

#         bboxes_array = []
#         bboxes_distance = []
        
#         for bbox in bboxes:
            
#             bboxes_array.append(bbox)
#             distance = bbox.center_x**2 +bbox.center_y**2

#             bboxes_distance.append(distance)
            
#         if len(bboxes_distance) > 1:

#             nearest_bbox_idx = np.where(min(bboxes_distance))
#             nearest_bbox = bboxes_array[nearest_bbox_idx[0][0]]

#         else:
#             nearest_bbox = bboxes





#     def selectBboxSize(self,bbox_class):

#         if bbox_class=='car':

#             Class_height = 1.39

#         elif bbox_class=='bicycle':

#             Class_height = 0.25  

#         elif bbox_class=='person':

#             Class_height = 1.85

#         elif bbox_class.endswith('light'):

#             Class_height = 0.2

#         elif bbox_class.startswith('trafficsign'):

#             Class_height = 0.81

#         elif bbox_class=='traffic sign':

#             Class_height = 0.81

#         elif bbox_class == 'drum':

#             Class_height = 1.07

#         else:

#             Class_height = 999

#         return Class_height


def bbox_conversion(xmin,xmax,ymin,ymax):

    # xmin = bbox.xmin
    # xmax = bbox.xmax
    # ymin = bbox.ymin
    # ymax = bbox.ymax
    # bbox_class = bbox.Class
    
    VEHICLE_LENGTH = 4.7
    REAR2CAMERA = 2.98
    HEIGHT = 0.9
    fov_deg = 52 # deg
    # image_width = 640
    # image_height = 720
    rot_roll  = 90
    rot_pitch = 0
    roll_yaw  = 90

    image_width = 640
    image_height = 480

    camera_translation = np.array([0, 1.35, HEIGHT])
    camera_translation = camera_translation.reshape(1,3)
    camera_rotation  = Rot3d(np.deg2rad(rot_roll), np.deg2rad(rot_pitch), np.deg2rad(roll_yaw))

    c_x = image_width/2
    c_y = image_height/2

    fov_rad = np.deg2rad(fov_deg)
    f_x = c_x/(math.tan(fov_rad/2))
    f_y = c_y/(math.tan(fov_rad/2))
    
    if f_x>=f_y:
        f_max = f_x
    else:
        f_max = f_y

    skew_coeff = 0

    intrinsic_matrix = np.array([[f_y,0,0],[skew_coeff, f_y, 0],[c_x, c_y, 1]])
    # intrinsic_matrix = np.array([[f_max,0,c_x],[0,f_max,c_y],[0,0,1]])

    bbox_TR = np.array([xmax,ymin,1])
    bbox_BL = np.array([xmin,ymax,1])
    bbox_TR = bbox_TR.reshape(1,3)
    bbox_BL = bbox_BL.reshape(1,3)


    intrinsic_inv = np.linalg.pinv(intrinsic_matrix)

    bbox_TR_3d_no_gain = bbox_TR.dot(intrinsic_inv)
    bbox_BL_3d_no_gain = bbox_BL.dot(intrinsic_inv)

    height_norm = abs(bbox_TR_3d_no_gain[0,1] - bbox_BL_3d_no_gain[0,1])
    
    # Class_height = self.selectBboxSize(bbox_class)
    Class_height = 1.39

    arbitrary_gain = Class_height/(height_norm)

    bbox_TR_3d = bbox_TR_3d_no_gain*arbitrary_gain
    bbox_BL_3d = bbox_BL_3d_no_gain*arbitrary_gain


    width_3d  = bbox_TR_3d[0,0] - bbox_BL_3d[0,0]
    length_3d = 0
    height_3d = bbox_BL_3d[0,1]- bbox_TR_3d[0,1]

    center_y = -(bbox_BL_3d[0,0]+ bbox_TR_3d[0,0])/2
    center_z = HEIGHT -(bbox_BL_3d[0,1]+ bbox_TR_3d[0,1])/2
    center_x = (bbox_BL_3d[0,2]+ bbox_TR_3d[0,2])/2 + VEHICLE_LENGTH - REAR2CAMERA


    # return center_x,center_y,center_z, width_3d, length_3d, height_3d, Class_height
    return center_x,center_y,center_z


def Rot3d(roll, pitch, yaw):

    '''
    unit :  deg
    ref : http://www.kwon3d.com/theory/euler/euler_angles.html
    '''

    Rotx = np.array([[1, 0, 0], [0, math.cos(roll), math.sin(roll)],\
        [0, -math.sin(roll), math.cos(roll)]])
    Roty = np.array([[math.cos(pitch), 0, -math.sin(pitch)], [0, 1, 0], \
        [math.sin(pitch), 0, math.cos(pitch)]])
    Rotz = np.array([[math.cos(yaw), math.sin(yaw), 0], \
        [-math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])

    rotation_matrix = Rotz*Roty*Rotx
    
    return rotation_matrix


    # def callbackImage(self,msg):

    #     self.imageTime =msg.header.stamp 

        
    # def callbackBBOX(self,msg):

    #     self.darknetFlag = True

    #     self.bbox_3d_array = bboxes_3d()
    #     self.bbox_3d_array.bboxes_3d = []
    #     self.bbox_3d_array.header.frame_id = 'world'
    #     self.bbox_3d_array.header.stamp = self.imageTime
        
        
    #     self.traffic_light_array = bboxes_3d()
    #     self.traffic_light_array.bboxes_3d = []
    #     self.traffic_light_array.header.frame_id = 'world'
    #     self.traffic_light_array.header.stamp = self.imageTime

    #     self.traffic_sign_array = bboxes_3d()
    #     self.traffic_sign_array.bboxes_3d = []
    #     self.traffic_sign_array.header.frame_id = 'world'
    #     self.traffic_sign_array.header.stamp = self.imageTime        


    #     for bbox in msg.bounding_boxes:

    #         bbox_3d_element = bbox_3d()
    #         bbox_3d_element.header.frame_id = 'world'
    #         bbox_3d_element.header.stamp = self.imageTime
    #         bbox_3d_element.id = bbox.id
    #         bbox_3d_element.Class = bbox.Class

    #         center_x,center_y,center_z, width_3d, length_3d, height_3d, Class_height = \
    #             self.bbox_conversion(bbox)

    #         bbox_3d_element.length = length_3d
    #         bbox_3d_element.width = width_3d
    #         bbox_3d_element.height = height_3d

    #         bbox_3d_element.center_x = center_x
    #         bbox_3d_element.center_y = center_y
    #         bbox_3d_element.center_z = center_z

    #         bbox_3d_element.probability = bbox.probability
    #         # print bbox_3d_element

    #         if (bbox_3d_element.probability > self.probability_thres) and ((bbox_3d_element.Class == 'person') or (bbox_3d_element.Class =='car')) :

    #             self.bbox_3d_array.bboxes_3d.append(bbox_3d_element)
                

    #             # if bbox_3d_element.Class.endswith('light'):

    #             #     self.traffic_light_array.bboxes_3d.append(bbox_3d_element)

    #             # elif bbox_3d_element.Class.startswith('trafficsign'):
    #             #     self.traffic_sign_array.bboxes_3d.append(bbox_3d_element)

    #     print self.bbox_3d_array


    #     self.bboxes_3d_pub.publish(self.bbox_3d_array)

    #     # if len(self.traffic_light_array.bboxes_3d) > 0:

    #     #     traffic_light_state = self.getTrafficLightState(self.traffic_light_array)
    #     #     self.traffic_light_state_pub.publish(traffic_light_state)

    #     # if len(self.traffic_sign_array.bboxes_3d) > 0:
    #     #     traffic_sign_state = self.getTrafficSignState(self.traffic_sign_array)
    #     #     self.traffic_sign_state_pub.publish(traffic_sign_state)

        
    # def callbackDarkent(self,msg):

    #     detection_image = msg.data

    #     # self.getNearestBbox(detection_image,self.bbox_3d_array)


    # def main(self):

    #     print '\n','hello px2world!','\n'
    #     print '\n','Probability thres : ',self.probability_thres,'\n'

    #     while not self.darknetFlag:

    #         print 'Waiting for darknet....'

    #         time.sleep(0.5)
        
        
if __name__ == '__main__':

    # try:

    #     p2w_node = pixel2world()
    #     p2w_node.main()
    #     rospy.spin()

    # except rospy.ROSInterruptException:
    #     pass
    # center_x,center_y,center_z, width_3d, length_3d, height_3d, Class_height = bbox_conversion(284,473,210,360)
    # center_x,center_y,center_z, width_3d, length_3d, height_3d, Class_height = bbox_conversion(201,317,221,305)
    center_x,center_y,center_z, width_3d, length_3d, height_3d, Class_height = bbox_conversion(448,539,225,288)

    print('x={}, y={}, z={}, wid={}, len={}, hei={}'.format(center_x,center_y,center_z,width_3d,length_3d,height_3d))