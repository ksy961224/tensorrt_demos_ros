cmake_minimum_required(VERSION 2.8.3)
project(tensorrt_demos_0)

find_package(catkin REQUIRED COMPONENTS 
rospy
std_msgs
yolonano_msg
)

catkin_package(

)


install(PROGRAMS
   trt_yolo_ros.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch 
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN *~ EXCLUDE
)

catkin_install_python(PROGRAMS
   trt_yolo_ros.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )



