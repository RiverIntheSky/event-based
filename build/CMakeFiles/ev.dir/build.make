# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/weizhen/ev

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/weizhen/ev/build

# Include any dependencies generated for this target.
include CMakeFiles/ev.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ev.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ev.dir/flags.make

CMakeFiles/ev.dir/main.cpp.o: CMakeFiles/ev.dir/flags.make
CMakeFiles/ev.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/weizhen/ev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ev.dir/main.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ev.dir/main.cpp.o -c /home/weizhen/ev/main.cpp

CMakeFiles/ev.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ev.dir/main.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/weizhen/ev/main.cpp > CMakeFiles/ev.dir/main.cpp.i

CMakeFiles/ev.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ev.dir/main.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/weizhen/ev/main.cpp -o CMakeFiles/ev.dir/main.cpp.s

CMakeFiles/ev.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/ev.dir/main.cpp.o.requires

CMakeFiles/ev.dir/main.cpp.o.provides: CMakeFiles/ev.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/ev.dir/build.make CMakeFiles/ev.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/ev.dir/main.cpp.o.provides

CMakeFiles/ev.dir/main.cpp.o.provides.build: CMakeFiles/ev.dir/main.cpp.o


CMakeFiles/ev.dir/src/event.cpp.o: CMakeFiles/ev.dir/flags.make
CMakeFiles/ev.dir/src/event.cpp.o: ../src/event.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/weizhen/ev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ev.dir/src/event.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ev.dir/src/event.cpp.o -c /home/weizhen/ev/src/event.cpp

CMakeFiles/ev.dir/src/event.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ev.dir/src/event.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/weizhen/ev/src/event.cpp > CMakeFiles/ev.dir/src/event.cpp.i

CMakeFiles/ev.dir/src/event.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ev.dir/src/event.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/weizhen/ev/src/event.cpp -o CMakeFiles/ev.dir/src/event.cpp.s

CMakeFiles/ev.dir/src/event.cpp.o.requires:

.PHONY : CMakeFiles/ev.dir/src/event.cpp.o.requires

CMakeFiles/ev.dir/src/event.cpp.o.provides: CMakeFiles/ev.dir/src/event.cpp.o.requires
	$(MAKE) -f CMakeFiles/ev.dir/build.make CMakeFiles/ev.dir/src/event.cpp.o.provides.build
.PHONY : CMakeFiles/ev.dir/src/event.cpp.o.provides

CMakeFiles/ev.dir/src/event.cpp.o.provides.build: CMakeFiles/ev.dir/src/event.cpp.o


CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.o: CMakeFiles/ev.dir/flags.make
CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.o: ../src/ThreadedEventIMU.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/weizhen/ev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.o -c /home/weizhen/ev/src/ThreadedEventIMU.cpp

CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/weizhen/ev/src/ThreadedEventIMU.cpp > CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.i

CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/weizhen/ev/src/ThreadedEventIMU.cpp -o CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.s

CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.o.requires:

.PHONY : CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.o.requires

CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.o.provides: CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.o.requires
	$(MAKE) -f CMakeFiles/ev.dir/build.make CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.o.provides.build
.PHONY : CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.o.provides

CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.o.provides.build: CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.o


CMakeFiles/ev.dir/src/Frontend.cpp.o: CMakeFiles/ev.dir/flags.make
CMakeFiles/ev.dir/src/Frontend.cpp.o: ../src/Frontend.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/weizhen/ev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/ev.dir/src/Frontend.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ev.dir/src/Frontend.cpp.o -c /home/weizhen/ev/src/Frontend.cpp

CMakeFiles/ev.dir/src/Frontend.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ev.dir/src/Frontend.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/weizhen/ev/src/Frontend.cpp > CMakeFiles/ev.dir/src/Frontend.cpp.i

CMakeFiles/ev.dir/src/Frontend.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ev.dir/src/Frontend.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/weizhen/ev/src/Frontend.cpp -o CMakeFiles/ev.dir/src/Frontend.cpp.s

CMakeFiles/ev.dir/src/Frontend.cpp.o.requires:

.PHONY : CMakeFiles/ev.dir/src/Frontend.cpp.o.requires

CMakeFiles/ev.dir/src/Frontend.cpp.o.provides: CMakeFiles/ev.dir/src/Frontend.cpp.o.requires
	$(MAKE) -f CMakeFiles/ev.dir/build.make CMakeFiles/ev.dir/src/Frontend.cpp.o.provides.build
.PHONY : CMakeFiles/ev.dir/src/Frontend.cpp.o.provides

CMakeFiles/ev.dir/src/Frontend.cpp.o.provides.build: CMakeFiles/ev.dir/src/Frontend.cpp.o


CMakeFiles/ev.dir/include/util/utils.cpp.o: CMakeFiles/ev.dir/flags.make
CMakeFiles/ev.dir/include/util/utils.cpp.o: ../include/util/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/weizhen/ev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/ev.dir/include/util/utils.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ev.dir/include/util/utils.cpp.o -c /home/weizhen/ev/include/util/utils.cpp

CMakeFiles/ev.dir/include/util/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ev.dir/include/util/utils.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/weizhen/ev/include/util/utils.cpp > CMakeFiles/ev.dir/include/util/utils.cpp.i

CMakeFiles/ev.dir/include/util/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ev.dir/include/util/utils.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/weizhen/ev/include/util/utils.cpp -o CMakeFiles/ev.dir/include/util/utils.cpp.s

CMakeFiles/ev.dir/include/util/utils.cpp.o.requires:

.PHONY : CMakeFiles/ev.dir/include/util/utils.cpp.o.requires

CMakeFiles/ev.dir/include/util/utils.cpp.o.provides: CMakeFiles/ev.dir/include/util/utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/ev.dir/build.make CMakeFiles/ev.dir/include/util/utils.cpp.o.provides.build
.PHONY : CMakeFiles/ev.dir/include/util/utils.cpp.o.provides

CMakeFiles/ev.dir/include/util/utils.cpp.o.provides.build: CMakeFiles/ev.dir/include/util/utils.cpp.o


CMakeFiles/ev.dir/ev_automoc.cpp.o: CMakeFiles/ev.dir/flags.make
CMakeFiles/ev.dir/ev_automoc.cpp.o: ev_automoc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/weizhen/ev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/ev.dir/ev_automoc.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ev.dir/ev_automoc.cpp.o -c /home/weizhen/ev/build/ev_automoc.cpp

CMakeFiles/ev.dir/ev_automoc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ev.dir/ev_automoc.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/weizhen/ev/build/ev_automoc.cpp > CMakeFiles/ev.dir/ev_automoc.cpp.i

CMakeFiles/ev.dir/ev_automoc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ev.dir/ev_automoc.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/weizhen/ev/build/ev_automoc.cpp -o CMakeFiles/ev.dir/ev_automoc.cpp.s

CMakeFiles/ev.dir/ev_automoc.cpp.o.requires:

.PHONY : CMakeFiles/ev.dir/ev_automoc.cpp.o.requires

CMakeFiles/ev.dir/ev_automoc.cpp.o.provides: CMakeFiles/ev.dir/ev_automoc.cpp.o.requires
	$(MAKE) -f CMakeFiles/ev.dir/build.make CMakeFiles/ev.dir/ev_automoc.cpp.o.provides.build
.PHONY : CMakeFiles/ev.dir/ev_automoc.cpp.o.provides

CMakeFiles/ev.dir/ev_automoc.cpp.o.provides.build: CMakeFiles/ev.dir/ev_automoc.cpp.o


# Object files for target ev
ev_OBJECTS = \
"CMakeFiles/ev.dir/main.cpp.o" \
"CMakeFiles/ev.dir/src/event.cpp.o" \
"CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.o" \
"CMakeFiles/ev.dir/src/Frontend.cpp.o" \
"CMakeFiles/ev.dir/include/util/utils.cpp.o" \
"CMakeFiles/ev.dir/ev_automoc.cpp.o"

# External object files for target ev
ev_EXTERNAL_OBJECTS =

ev: CMakeFiles/ev.dir/main.cpp.o
ev: CMakeFiles/ev.dir/src/event.cpp.o
ev: CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.o
ev: CMakeFiles/ev.dir/src/Frontend.cpp.o
ev: CMakeFiles/ev.dir/include/util/utils.cpp.o
ev: CMakeFiles/ev.dir/ev_automoc.cpp.o
ev: CMakeFiles/ev.dir/build.make
ev: /usr/local/lib/libglog.a
ev: /usr/local/lib/libokvis_util.a
ev: /usr/local/lib/libokvis_kinematics.a
ev: /usr/local/lib/libokvis_time.a
ev: /usr/local/lib/libokvis_cv.a
ev: /usr/local/lib/libokvis_common.a
ev: /usr/local/lib/libokvis_ceres.a
ev: /usr/local/lib/libokvis_timing.a
ev: /usr/local/lib/libokvis_matcher.a
ev: /usr/local/lib/libokvis_frontend.a
ev: /usr/local/lib/libokvis_multisensor_processing.a
ev: /usr/local/lib/libvisensor.so
ev: /usr/local/lib/libokvis_frontend.a
ev: /usr/local/lib/libokvis_ceres.a
ev: /usr/local/lib/libokvis_common.a
ev: /usr/local/lib/libokvis_cv.a
ev: /usr/local/lib/libokvis_kinematics.a
ev: /usr/local/lib/libokvis_time.a
ev: /usr/local/lib/libokvis_timing.a
ev: /usr/local/lib/libokvis_matcher.a
ev: /usr/local/lib/libokvis_util.a
ev: /home/weizhen/Documents/okvis/build/brisk/src/brisk_external-build/lib/libbrisk.a
ev: /home/weizhen/Documents/okvis/build/brisk/src/brisk_external-build/lib/libagast.a
ev: /opt/ros/kinetic/lib/libopencv_stitching3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_superres3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_videostab3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_photo3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_aruco3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_bgsegm3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_bioinspired3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_ccalib3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_cvv3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_dpm3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_face3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_fuzzy3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_hdf3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_img_hash3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_line_descriptor3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_optflow3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_reg3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_rgbd3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_saliency3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_stereo3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_structured_light3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_viz3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_phase_unwrapping3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_surface_matching3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_tracking3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_datasets3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_plot3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_text3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_dnn3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_xfeatures2d3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_ml3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_shape3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_video3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_ximgproc3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_calib3d3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_features2d3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_highgui3.so.3.3.1
ev: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.5.1
ev: /opt/ros/kinetic/lib/libopencv_videoio3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_flann3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_xobjdetect3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_imgcodecs3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_objdetect3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_xphoto3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_imgproc3.so.3.3.1
ev: /opt/ros/kinetic/lib/libopencv_core3.so.3.3.1
ev: /usr/local/lib/libceres.a
ev: /usr/lib/x86_64-linux-gnu/libglog.so
ev: /usr/lib/x86_64-linux-gnu/libgflags.so
ev: /usr/local/lib/libmetis.so
ev: /usr/local/lib/libspqr.a
ev: /usr/lib/x86_64-linux-gnu/libtbb.so
ev: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
ev: /usr/local/lib/libcholmod.a
ev: /usr/local/lib/libccolamd.a
ev: /usr/local/lib/libcamd.a
ev: /usr/local/lib/libcolamd.a
ev: /usr/local/lib/libamd.a
ev: /usr/lib/liblapack.so
ev: /usr/lib/libf77blas.so
ev: /usr/lib/libatlas.so
ev: /usr/lib/libf77blas.so
ev: /usr/lib/libatlas.so
ev: /usr/local/lib/libsuitesparseconfig.a
ev: /usr/lib/x86_64-linux-gnu/librt.so
ev: /usr/local/lib/libcxsparse.a
ev: /home/weizhen/Documents/okvis/build/opengv/src/opengv/lib/libopengv.a
ev: /usr/local/lib/libglog.a
ev: CMakeFiles/ev.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/weizhen/ev/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable ev"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ev.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ev.dir/build: ev

.PHONY : CMakeFiles/ev.dir/build

CMakeFiles/ev.dir/requires: CMakeFiles/ev.dir/main.cpp.o.requires
CMakeFiles/ev.dir/requires: CMakeFiles/ev.dir/src/event.cpp.o.requires
CMakeFiles/ev.dir/requires: CMakeFiles/ev.dir/src/ThreadedEventIMU.cpp.o.requires
CMakeFiles/ev.dir/requires: CMakeFiles/ev.dir/src/Frontend.cpp.o.requires
CMakeFiles/ev.dir/requires: CMakeFiles/ev.dir/include/util/utils.cpp.o.requires
CMakeFiles/ev.dir/requires: CMakeFiles/ev.dir/ev_automoc.cpp.o.requires

.PHONY : CMakeFiles/ev.dir/requires

CMakeFiles/ev.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ev.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ev.dir/clean

CMakeFiles/ev.dir/depend:
	cd /home/weizhen/ev/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/weizhen/ev /home/weizhen/ev /home/weizhen/ev/build /home/weizhen/ev/build /home/weizhen/ev/build/CMakeFiles/ev.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ev.dir/depend

