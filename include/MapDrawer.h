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
class Frame;

typedef struct {
    float x, y, p, t;
} event;

class MapDrawer {
public:
    MapDrawer(Parameters* param_, Map* map_, Tracking* tracking_): param(param_), map(map_), tracking(tracking_) {}

    void setUp();

    // patches
    void setUpPatchShader();

    // full screen quad
    void setUpQuadShader();
    void drawQuad();

    // gaussian blur
    void setUpGaussianBlurShader();
    void gaussianBlur(GLuint& imageFBO, GLuint& imageTex, GLuint& blurredFBO, GLuint& blurredTex, glm::vec2 dir);

    // events
    void setUpEventShader();
    void updateEvents();
    void setUpSquareShader();
    void setUpSummationShader();

    // shader
    void setUp2DRect(GLuint& FBO, GLuint& tex);
    void setUpShader(GLuint& shader, const char* filename);
    void setUpSampler2D(GLuint& FBO, GLuint& tex);

    void drawMapPoints();
//    void framebuffer_size_callback(GLFWwindow* window, int width, int height);

    GLuint patchVAO, patchVBO, patchEBO, patchVS, patchFS, patchShader,
           patchFramebuffer, patchOcclusion;
    GLint apos_location, model_location, view_location, projection_location;

    GLuint quadVAO, quadVBO, quadEBO, quadVS, quadFS, quadShader;
    GLint quad_tex_location;

    GLuint eventVAO, eventVBO, eventVS, eventFS, eventShader, warpFramebuffer, warppedImage;
    GLint event_apos_location, w_location, v_location, camera_matrix_location,
          near_plane_location, occlusion_map_location, event_projection_location;

    GLuint blurShader, blurFramebuffer, blurredImage;
    GLint blur_apos_location, atex_location, dir_location;

    GLuint squareShader, squareFramebuffer, squaredImage;
    GLint square_tex_location, square_apos_location;

    GLuint sumShader, sumFramebuffer, sumImage;
    GLint sum_tex_location, sum_apos_location;

    std::string shaderFilePath;
    GLFWwindow* window;
    Parameters* param;
    Frame* frame;
    Map* map;
    Tracking* tracking;
    // assign this member to frame??
    cv::Mat framePartition;
};
}
