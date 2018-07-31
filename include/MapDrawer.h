#pragma once

#include "Shader.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <memory>

#include "GLFW/linmath.h"
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
    void updateFrame();
    void setUpSquareShader();
    void setUpSummationShader();

    float tracking_cost_func(cv::Mat& w, cv::Mat& v);
    float cost_func(cv::Mat& w, cv::Mat& v);
    float cost_func(GLuint& fbo);

    // shader
    void setUp2DRect(GLuint& FBO, GLuint& tex);
    void setUpShader(GLuint& shader, const char* filename);
    void setUp2DMultisample(GLuint& FBO, GLuint& tex);

    // map
    void draw_map();
    void visualize_map();
    void update_map_texture();
    void draw_map_texture();

    // optimization pipeline
    void optimize_gsl(double ss, int nv, double (*f)(const gsl_vector*, void*), void *params,
                             gsl_multimin_fminimizer* s, gsl_vector* x, double* res, size_t iter);
    void initialize_map();
    static double initialize_cost_func(const gsl_vector *vec, void *params);
    static double optimize_cost_func(const gsl_vector *vec, void *params);
    static double frame_cost_func(const gsl_vector *vec, void *params);
    static double tracking_cost_func(const gsl_vector *vec, void *params);
    float initialize_map_draw(cv::Mat& nws, std::vector<float>& inv_d_ws, cv::Mat& w, cv::Mat& v);
    float tracking_draw(cv::Mat& w, cv::Mat& v);
    void optimize_map();
    void optimize_frame();
    void track();

    float optimize_map_draw(cv::Mat& nws, std::vector<float>& inv_d_ws, cv::Mat& w, cv::Mat& v);

    void drawMapPoints();
//    void framebuffer_size_callback(GLFWwindow* window, int width, int height);

    GLuint patchVAO, patchVBO, patchEBO, patchVS, patchFS, patchShader,
           patchFramebuffer, patchOcclusion;
    GLint apos_location, model_location, view_location, projection_location;

    GLuint quadVAO, quadVBO, quadEBO, quadVS, quadFS, quadShader;
    GLint quad_tex_location;

    GLuint eventVAO, eventVBO, eventShader, warpFramebuffer, warppedImage;
    GLint event_apos_location, w_location, v_location, camera_matrix_location, wc1c2_location,
          tc1c2_location, near_plane_location, occlusion_map_location, event_projection_location;

    GLuint blurShader, blurFramebuffer, blurredImage;
    GLint blur_apos_location, atex_location, dir_location;

    GLuint squareShader, squareFramebuffer, squaredImage;
    GLint square_tex_location, square_apos_location;

    GLuint sumShader, sumFramebuffer, sumImage;
    GLint sum_tex_location, sum_apos_location;

    GLuint tmpFramebuffer, tmpImage;

    std::string shaderFilePath;
    GLFWwindow* window;
    Parameters* param;
    std::shared_ptr<Frame> frame;
    Map* map;
    Tracking* tracking;
    // assign this member to frame??
    cv::Mat framePartition;
};
}
