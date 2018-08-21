cmake_minimum_required(VERSION 3.0.2)

project(ev)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()

# require OpenCV
find_package( OpenCV COMPONENTS core highgui imgproc features2d REQUIRED )
include_directories(BEFORE ${OpenCV_INCLUDE_DIRS})

# require Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# require gsl
find_package(GSL REQUIRED)

# require OpenGV
find_package(opengv REQUIRED)

find_package(glog REQUIRED)

#find_package(Gnuplot REQUIRED)

FIND_PACKAGE(Boost COMPONENTS filesystem system iostreams REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

include_directories(${PROJECT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE LIBS "*.cc" "*.h" "*.*sh")

add_library(${PROJECT_NAME} SHARED ${LIBS})

add_executable(event_based "main.cpp")

target_link_libraries(event_based
    ${PROJECT_NAME}
    glog::glog
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
    GSL::gsl
    GSL::gslcblas)
