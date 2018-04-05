cmake_minimum_required(VERSION 3.0.2)

project(ev)

set(OKVIS_INSTALLATION "/home/weizhen/Downloads/okvis")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()

# require OpenCV
find_package( OpenCV COMPONENTS core highgui imgproc features2d REQUIRED )
include_directories(BEFORE ${OpenCV_INCLUDE_DIRS})

# require okvis
# set(CMAKE_PREFIX_PATH "/usr/local/lib/")
find_package( okvis REQUIRED)
include_directories(${OKVIS_INCLUDE_DIRS})


# require brisk
find_package( brisk 2 REQUIRED)
include_directories(${BRISK_INCLUDE_DIRS})

# require ceres
list(APPEND CMAKE_PREFIX_PATH ${OKVIS_INSTALLATION})
find_package( Ceres REQUIRED )
include_directories(${CERES_INCLUDE_DIRS})

# require OpenGV
find_package(opengv REQUIRED)

# VISensor, if available
list(APPEND CMAKE_MODULE_PATH ${OKVIS_INSTALLATION}/cmake)
find_package(VISensor)
if(VISENSORDRIVER_FOUND)
  message(STATUS "Found libvisensor.")
else()
  message(STATUS "libvisensor not found")
endif()

find_package(glog REQUIRED)

find_package(Qt5Core)

include_directories(${PROJECT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include)


# only for editting, delete all non-executables afterwards
add_executable(${PROJECT_NAME} "main.cpp"
    "${PROJECT_SOURCE_DIR}/event.cpp"
    "${PROJECT_SOURCE_DIR}/ThreadedEventIMU.cpp"
    "${PROJECT_SOURCE_DIR}/Frontend.cpp"
    "${CMAKE_SOURCE_DIR}/include/util/utils.cpp")

target_link_libraries(${PROJECT_NAME}
    Qt5::Core
    glog::glog
    ${OpenCV_LIBS}
    ${OKVIS_LIBRARIES})

if(${VISENSORDRIVER_FOUND})
    target_link_libraries(ev ${VISensorDriver_LIBRARY})
endif()
