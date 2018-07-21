cmake_minimum_required(VERSION 3.0.2)

project(ev)

set(DEFAULT_BUILD_TYPE "Release")
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

# require OpenMP
OPTION (USE_OpenMP "Use OpenMP to enamble <omp.h>" OFF)

find_package(GSL REQUIRED)

#if(USE_OpenMP)
#  find_package(OpenMP REQUIRED)
#endif(USE_OpenMP)

if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

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

#find_package(Gnuplot REQUIRED)

FIND_PACKAGE(Boost COMPONENTS filesystem system iostreams REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

include_directories(${PROJECT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include)

# Add GLFW lib (with disabled spam)
option(GLFW_BUILD_EXAMPLES "" OFF)
option(GLFW_BUILD_TESTS "" OFF)
option(GLFW_BUILD_DOCS "" OFF)
option(GLFW_INSTALL "" OFF)
option(BUILD_SHARED_LIBS "" ON)
add_subdirectory(libs/glfw ${CMAKE_BINARY_DIR}/libs/glfw)

add_subdirectory(libs/glad ${CMAKE_BINARY_DIR}/libs/glad)

# only for editting, delete all non-executables afterwards
add_library(${PROJECT_NAME} SHARED
${PROJECT_SOURCE_DIR}/event.cpp
${PROJECT_SOURCE_DIR}/ThreadedEventIMU.cpp
${PROJECT_SOURCE_DIR}/Frontend.cpp
include/util/utils.cpp
include/KeyFrame.cc
include/Map.cc
include/MapPoint.cc
include/Tracking.cc
include/Frame.cc
include/Converter.cc
include/Optimizer.cc
include/MapDrawer.h
include/MapDrawer.cc
shaders/shader.vsh
shaders/shader.fsh
include/Shader.h
include/Shader.cc)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_SOURCE_DIR}/libs)

add_executable(event_based
    main.cpp
)

target_link_libraries(${PROJECT_NAME}
    glad
    glfw)

target_link_libraries(event_based PUBLIC
    ${PROJECT_NAME}
    glog::glog
    ${OpenCV_LIBS}
    ${OKVIS_LIBRARIES}
    ${Boost_LIBRARIES}
    GSL::gsl
    GSL::gslcblas)

if(${VISENSORDRIVER_FOUND})
    target_link_libraries(${PROJECT_NAME} ${VISensorDriver_LIBRARY} )
endif()
