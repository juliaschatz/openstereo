cmake_minimum_required(VERSION 3.13)

project(stereocam)
set(CMAKE_BUILD_TYPE Release)

## Compile as C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11 -Wall -Wextra -Wno-switch -ftree-vectorize -mfpu=neon -fopt-info-vec-all)

find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  roscpp
  rospy
)

include_directories(${catkin_INCLUDE_DIRS})
include_directories(include)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4/")
find_package(OpenCV REQUIRED PATHS /usr/local/lib/cmake/opencv4 NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})

catkin_package(
    CATKIN_DEPENDS std_msgs sensor_msgs
)



add_executable(stereocam_node src/stereocam_node.cpp)
add_dependencies(stereocam_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(stereocam_node ${OpenCV_LIBRARIES})
target_link_libraries(stereocam_node ${catkin_LIBRARIES})