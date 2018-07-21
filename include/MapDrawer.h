#pragma once

#include "Shader.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "GLFW/linmath.h"
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "parameters.h"
#include "Map.h"
#include "Tracking.h"

namespace ev {
class Map;
class Tracking;

class MapDrawer {
public:
    MapDrawer(Parameters* param_, Map* map_, Tracking* tracking_): param(param_), map(map_), tracking(tracking_) {}

    void drawMapPoints();
//    void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    Parameters* param;
    Map* map;
    Tracking* tracking;
    // assign this member to frame??
    cv::Mat framePartition;
};
}
