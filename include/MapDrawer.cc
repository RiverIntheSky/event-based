#include "MapDrawer.h"
#include <chrono>

namespace ev {
typedef void (*framebuffer_size_callback)(void);

void MapDrawer::drawMapPoints() {
    // fixed size
    int size[] = {param->height, param->width, 3};
    framePartition = cv::Mat(3, size, CV_32F);

    setUp();

    while(!glfwWindowShouldClose(window)) {

        if (map->isDirty) {

            updateFrame();
            if (map->mspMapPoints.empty()) {

                initialize_map();
                visualize_map();
            }
            glFinish();

            map->isDirty = false;

            if (tracking->newFrame) {

                track();
                visualize_map();
                glFinish();
                tracking->newFrame = false;
            }
        }

        std::this_thread::yield();
    }
    glfwTerminate();
}

void MapDrawer::setUp() {

    if (!glfwInit())
        exit(EXIT_FAILURE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(param->width, param->height, "window", NULL, NULL);
    if (!window) {
        exit(EXIT_FAILURE);
        glfwTerminate();
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        exit(EXIT_FAILURE);
    glViewport(0, 0, param->width, param->height);
//    glfwSetFramebufferSizeCallback(window, &MapDrawer::framebuffer_size_callback, this);
//    glfwSetKeyCallback(window, key_callback);

    glClearColor(0.f, 0.f, 0.f, 1.0f);

    shaderFilePath = __FILE__;
    shaderFilePath = shaderFilePath.substr(0, shaderFilePath.find_last_of("/\\"));
    shaderFilePath = shaderFilePath.substr(0, shaderFilePath.find_last_of("/\\"));
    shaderFilePath += "/shaders/";

    setUpPatchShader();
    setUpEventShader();
    setUpQuadShader();
    setUpGaussianBlurShader();
    setUpContrastShader();
    setUpCompareShader();
    setUpSummationShader();
    setUp2DRect(tmpFramebuffer, tmpImage);
    setUp2DRect(mapFramebuffer, mapImage);
}

void MapDrawer::setUpPatchShader() {

    GLfloat points[] = {
            -0.1f,  0.1f,
             0.1f,  0.1f,
             0.1f, -0.1f,
            -0.1f, -0.1f
    };
    unsigned int indices[] = {
        3, 1, 0,
        3, 2, 1
    };

    // create vao
    {
        glGenVertexArrays(1, &patchVAO);
        glBindVertexArray(patchVAO);

        glGenBuffers(1, &patchVBO);
        glBindBuffer(GL_ARRAY_BUFFER, patchVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);

        glGenBuffers(1, &patchEBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, patchEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    }

    setUpShader(patchShader, "patch");
    setUp2DRect(patchFramebuffer, patchOcclusion);

    // set up uniforms and attributes
    {
        apos_location = glGetAttribLocation(patchShader, "aPos");
        glEnableVertexAttribArray(apos_location);
        glVertexAttribPointer(apos_location, 2, GL_FLOAT, GL_FALSE, 0 * sizeof(float), (void*)0);

        glm::mat4 projection;

        projection[0][0] = 2.f * param->fx / param->width;

        projection[1][1] = 2.f * param->fy / param->height;

        projection[2][0] = 1.f - 2.f * param->cx / param->width;
        projection[2][1] = 2.f * param->cy / param->height - 1.f;
        projection[2][2] = (param->zfar + param->znear) / (param->znear - param->zfar);
        projection[2][3] = -1.f;

        projection[3][2] = 2.f * param->zfar * param->znear / (param->znear - param->zfar);
        projection[3][3] = 0.f;

        model_location = glGetUniformLocation(patchShader, "model");
        view_location = glGetUniformLocation(patchShader, "view");
        projection_location = glGetUniformLocation(patchShader, "projection");
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, glm::value_ptr(projection));
    }
}

void MapDrawer::setUpEventShader() {

    setUpShader(eventShader, "event");
    setUp2DRect(warpFramebuffer, warppedImage);

    // set up uniforms and attributes
    {
        event_apos_location = glGetAttribLocation(eventShader, "aPos");


        w_location = glGetUniformLocation(eventShader, "w");
        v_location = glGetUniformLocation(eventShader, "v");
        camera_matrix_location = glGetUniformLocation(eventShader, "cameraMatrix");
        glm::mat3 K;
        K[0][0] = param->fx;
        K[1][1] = param->fy;
        K[2][0] = param->cx;
        K[2][1] = param->cy;
        glUniformMatrix3fv(camera_matrix_location, 1, GL_FALSE, glm::value_ptr(K));

        wc1c2_location = glGetUniformLocation(eventShader, "wc1c2");
        tc1c2_location = glGetUniformLocation(eventShader, "tc1c2");

        near_plane_location = glGetUniformLocation(eventShader, "nearPlane");
        glUniform1f(near_plane_location, param->znear);

        occlusion_map_location = glGetUniformLocation(eventShader, "patchTexture");
        glUniform1i(occlusion_map_location, 0);

        use_polarity_location = glGetUniformLocation(eventShader, "usePolarity");
        glUniform1i(use_polarity_location, true);
    }
}

void MapDrawer::setUpQuadShader() {

    GLfloat quad[] = {
            -1.f,  1.f,
             1.f,  1.f,
             1.f, -1.f,
            -1.f, -1.f
    };

    unsigned int indices[] = {
        3, 1, 0,
        3, 2, 1
    };

    // create vao
    {
        glGenVertexArrays(1, &quadVAO);
        glBindVertexArray(quadVAO);

        glGenBuffers(1, &quadVBO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

        glGenBuffers(1, &quadEBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    }

    setUpShader(quadShader, "screen");

    {
        atex_location = glGetUniformLocation(quadShader, "tex");
        glUniform1i(atex_location, 0);
    }
}

void MapDrawer::setUpSummationShader() {
    setUp2DRect(sumFramebuffer, sumImage);
    setUpShader(sumShader, "summation");

    // set up uniforms and attributes
    {
        sum_tex_location = glGetUniformLocation(sumShader, "tex");
        glUniform1i(atex_location, 0);
        sum_apos_location = glGetAttribLocation(blurShader, "aPos");
        glEnableVertexAttribArray(blur_apos_location);
        glVertexAttribPointer(blur_apos_location, 2, GL_FLOAT, GL_FALSE, 0 * sizeof(float), (void*)0);
    }
}

void MapDrawer::setUpGaussianBlurShader() {
    setUp2DRect(blurFramebuffer, blurredImage);
    setUpShader(blurShader, "gaussian_blur");

    // set up uniforms and attributes
    {
        atex_location = glGetUniformLocation(blurShader, "tex");
        glUniform1i(atex_location, 0);
        dir_location = glGetUniformLocation(blurShader, "dir");
        blur_apos_location = glGetAttribLocation(blurShader, "aPos");
        glEnableVertexAttribArray(blur_apos_location);
        glVertexAttribPointer(blur_apos_location, 2, GL_FLOAT, GL_FALSE, 0 * sizeof(float), (void*)0);
    }
}

void MapDrawer::setUpContrastShader() {
    setUp2DRect(contrastFramebuffer, contrastImage);
    setUpShader(contrastShader, "contrast");

    // set up uniforms and attributes
    {
        contrast_atex_location = glGetUniformLocation(contrastShader, "tex");
        glUniform1i(contrast_atex_location, 0);
        mean_location = glGetUniformLocation(contrastShader, "mean");
        glUniform1f(mean_location, 0.f);
        contrast_apos_location = glGetAttribLocation(contrastShader, "aPos");
        glEnableVertexAttribArray(contrast_apos_location);
        glVertexAttribPointer(contrast_apos_location, 2, GL_FLOAT, GL_FALSE, 0 * sizeof(float), (void*)0);
    }
}

void MapDrawer::setUpCompareShader() {
    setUpShader(compareShader, "compare");

    // set up textures
    {
        glGenTextures(1, &texFrame);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_RECTANGLE, texFrame);
        glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB32F, param->width, param->height, 0,
                     GL_RGB, GL_FLOAT, 0);
        tex_frame_location = glGetUniformLocation(compareShader, "tex0");
        glUniform1i(tex_frame_location, 0);

        glGenTextures(1, &texMap);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_RECTANGLE, texMap);
        glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB32F, param->width, param->height, 0,
                     GL_RGB, GL_FLOAT, 0);
        tex_map_location = glGetUniformLocation(compareShader, "tex1");
        glUniform1i(tex_map_location, 1);

        glActiveTexture(GL_TEXTURE0);
    }
}

void MapDrawer::setUp2DRect(GLuint& FBO, GLuint& tex) {
    glGenFramebuffers(1, &FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_RECTANGLE, tex);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB32F, param->width, param->height, 0,
                 GL_RGB, GL_FLOAT, 0);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, tex, 0);
}

void MapDrawer::setUp2DMultisample(GLuint& FBO, GLuint& tex) {
    glGenFramebuffers(1, &FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);

    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA32F, param->width, param->height, false);
    glBindTexture(GL_TEXTURE_2D, 0);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, tex, 0);
}

void MapDrawer::setUpShader(GLuint& shader, const char* filename) {    
    shader = glCreateProgram();

    std::string f(filename);
    std::string vsh_path = shaderFilePath + f + ".vsh",
                gsh_path = shaderFilePath + f + ".gsh",
                fsh_path = shaderFilePath + f + ".fsh";
    const char* vsh_file = vsh_path.c_str();
    const char* gsh_file = gsh_path.c_str();
    const char* fsh_file = fsh_path.c_str();
    GLuint vsh, gsh, fsh;

    if (createShader(GL_VERTEX_SHADER, vsh_file, vsh))
        glAttachShader(shader, vsh);
    if (createShader(GL_GEOMETRY_SHADER, gsh_file, gsh))
        glAttachShader(shader, gsh);
    if (createShader(GL_FRAGMENT_SHADER, fsh_file, fsh))
        glAttachShader(shader, fsh);

    glLinkProgram(shader);
    glUseProgram(shader);
    glDeleteShader(vsh);
    glDeleteShader(gsh);
    glDeleteShader(fsh);
}

void MapDrawer::drawQuad() {
    glBindVertexArray(quadVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void MapDrawer::gaussianBlur(GLuint& imageTex, glm::vec2 dir) {
    glBindFramebuffer(GL_FRAMEBUFFER, blurFramebuffer);
    glBindTexture(GL_TEXTURE_RECTANGLE, imageTex);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(blurShader);
    glUniform2fv(dir_location, 1, glm::value_ptr(dir));

    drawQuad();
}

void MapDrawer::gaussianBlur(GLuint& fbo, GLuint& tex) {
    gaussianBlur(tex, glm::vec2(0, 1));
    std::swap(blurFramebuffer, fbo);
    std::swap(blurredImage, tex);
    gaussianBlur(tex, glm::vec2(1, 0));
    std::swap(blurFramebuffer, fbo);
    std::swap(blurredImage, tex);
}

float MapDrawer::contrast(GLuint fbo) {
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tmpFramebuffer);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBlitFramebuffer(0, 0, param->width, param->height, 0, 0, param->width, param->height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

    gaussianBlur(tmpImage, glm::vec2(0, 1));
    std::swap(blurFramebuffer, tmpFramebuffer);
    std::swap(blurredImage, tmpImage);
    gaussianBlur(tmpImage, glm::vec2(1, 0));

    glUseProgram(contrastShader);
    glUniform1f(mean_location, mean(blurredImage));         
    glBindFramebuffer(GL_FRAMEBUFFER, contrastFramebuffer);
    glBindTexture(GL_TEXTURE_RECTANGLE, blurredImage);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(contrastShader);
    drawQuad();

    float c =  -mean(contrastImage);
//    LOG(INFO) << c;
    return c;
}

float MapDrawer::sum(GLuint& tex) {
    glBindFramebuffer(GL_FRAMEBUFFER, tmpFramebuffer);
    glBindTexture(GL_TEXTURE_RECTANGLE, tex);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(quadShader);
    drawQuad();

    float sum;
    glUseProgram(sumShader);
    float currentW = param->width,
          currentH = param->height;
    while (currentW > 1) {
        glBindFramebuffer(GL_FRAMEBUFFER, sumFramebuffer);
        glBindTexture(GL_TEXTURE_RECTANGLE, tmpImage);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        std::swap(tmpFramebuffer, sumFramebuffer);
        std::swap(tmpImage, sumImage);

        glViewport(0, 0, currentW, currentH);
        currentW = std::ceil(currentW / 2);
        currentH = std::ceil(currentH / 2);
    }

    glReadPixels(0, 0, 1, 1, GL_RED, GL_FLOAT, &sum);
    glViewport(0, 0, param->width, param->height);
    return sum;
}

float MapDrawer::overlap(GLuint& fbo1, GLuint& tex1, GLuint& fbo2, GLuint& tex2) {
    gaussianBlur(fbo1, tex1);
    gaussianBlur(fbo2, tex2);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(compareShader);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE, tex1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_RECTANGLE, tex2);
    drawQuad();
    glActiveTexture(GL_TEXTURE0);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tmpFramebuffer);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBlitFramebuffer(0, 0, param->width, param->height, 0, 0, param->width, param->height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

//    drawImage(tmpImage);

    float* count = (float *)malloc(3 * sizeof(float));
    glUseProgram(sumShader);
    float currentW = param->width,
          currentH = param->height;
    while (currentW > 1) {
        glBindFramebuffer(GL_FRAMEBUFFER, sumFramebuffer);
        glBindTexture(GL_TEXTURE_RECTANGLE, tmpImage);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        std::swap(tmpFramebuffer, sumFramebuffer);
        std::swap(tmpImage, sumImage);

        glViewport(0, 0, currentW, currentH);
        currentW = std::ceil(currentW / 2);
        currentH = std::ceil(currentH / 2);
    }

    glReadPixels(0, 0, 1, 1, GL_RGB, GL_FLOAT, count);
    glViewport(0, 0, param->width, param->height);

    return count[1]/count[0];
}

float MapDrawer::overlap(cv::Mat& Rwc, cv::Mat& twc, cv::Mat& w, cv::Mat& v){
    // warp to tmpFramebuffer
    set_use_polarity(false);
    warp(Rwc, twc, w, v);
    draw_map_texture(Rwc, twc, mapFramebuffer);

    return overlap(tmpFramebuffer, tmpImage, mapFramebuffer, mapImage);

}

// store to tmp
void MapDrawer::warp(cv::Mat& Rwc, cv::Mat& twc, cv::Mat& w, cv::Mat& v) {
    draw_map_patch(Rwc, twc);

    glBindFramebuffer(GL_FRAMEBUFFER, warpFramebuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_RECTANGLE, patchOcclusion);

    glUseProgram(eventShader);

    glm::vec3 w_ = Converter::toGlmVec3(w);
    glUniform3fv(w_location, 1, glm::value_ptr(w_));
    glm::vec3 v_ = Converter::toGlmVec3(v);
    glUniform3fv(v_location, 1, glm::value_ptr(v_));
    glm::vec3 wc1c2, tc1c2;
    glUniform3fv(wc1c2_location, 1, glm::value_ptr(wc1c2));
    glUniform3fv(tc1c2_location, 1, glm::value_ptr(tc1c2));

    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    glBindVertexArray(frame->vao);
    glDrawArrays(GL_POINTS, 0, frame->events());

    glDisable(GL_BLEND);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, tmpFramebuffer);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, warpFramebuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBlitFramebuffer(0, 0, param->width, param->height, 0, 0, param->width, param->height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

    //gaussianBlur(tmpFramebuffer, tmpImage);

//    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, warpFramebuffer);
//    glBindFramebuffer(GL_READ_FRAMEBUFFER, blurFramebuffer);
//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//    glBlitFramebuffer(0, 0, param->width, param->height, 0, 0, param->width, param->height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

float MapDrawer::mean(GLuint& tex) {
    return sum(tex) / (param->width * param->height);
}

void MapDrawer::set_use_polarity(bool b){
    glUseProgram(eventShader);
    glUniform1i(use_polarity_location, b);
}

void MapDrawer::updateFrame() {
    frame = tracking->getCurrentFrame();
    std::vector<event> events;
    events.reserve(param->window_size);
    auto t0 = frame->mTimeStamp;
    for (auto e: frame->vEvents) {
        event e_;
        e_.x = float(e->measurement.x);
        e_.y = float(e->measurement.y);
        e_.p = float(e->measurement.p);
        e_.t = float((e->timeStamp - t0).toSec());
        events.push_back(e_);
    }

    // create vao
    {
        glGenVertexArrays(1, &frame->vao);
        glBindVertexArray(frame->vao);

        glGenBuffers(1, &frame->vbo);
        glBindBuffer(GL_ARRAY_BUFFER, frame->vbo);
        glBufferData(GL_ARRAY_BUFFER, param->window_size * sizeof(event), &events[0], GL_STATIC_DRAW);

        glEnableVertexAttribArray(event_apos_location);
        glVertexAttribPointer(event_apos_location, 4, GL_FLOAT, GL_FALSE, 0 * sizeof(event), (void*)0);
    }

}

float MapDrawer::cost_func(cv::Mat& w, cv::Mat& v) {

    glBindTexture(GL_TEXTURE_RECTANGLE, patchOcclusion);

    glUseProgram(eventShader);

    glm::vec3 w_ = Converter::toGlmVec3(w);
    glUniform3fv(w_location, 1, glm::value_ptr(w_));
    glm::vec3 v_ = Converter::toGlmVec3(v);
    glUniform3fv(v_location, 1, glm::value_ptr(v_));
    glm::vec3 wc1c2, tc1c2;
    glUniform3fv(wc1c2_location, 1, glm::value_ptr(wc1c2));
    glUniform3fv(tc1c2_location, 1, glm::value_ptr(tc1c2));

    glBindFramebuffer(GL_FRAMEBUFFER, warpFramebuffer);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    glBindVertexArray(frame->vao);
    glDrawArrays(GL_POINTS, 0, frame->events());

    glDisable(GL_BLEND);

    return contrast(warpFramebuffer);
}

void MapDrawer::draw_map_texture(cv::Mat& Rwc, cv::Mat& twc, GLuint& fbo) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for (auto kf: map->getAllKeyFrames()) {
        cv::Mat Rwc1 = kf->getRotation();
        cv::Mat twc1 = kf->getTranslation();
        glm::mat4 view = toView(Rwc1, twc1);

        glUseProgram(patchShader);
        glBindVertexArray(patchVAO);
        glBindFramebuffer(GL_FRAMEBUFFER, patchFramebuffer);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUniformMatrix4fv(view_location, 1, GL_FALSE, glm::value_ptr(view));
        glEnable(GL_DEPTH_TEST);

        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        // change to observation
        for (auto mpPoint: map->mspMapPoints) {

            cv::Mat nw = mpPoint->getNormal();
            cv::Mat nc = Rwc1.t() * nw;
            float inv_d_c = 1.f/(1.f/mpPoint->d + twc1.dot(nw));
            glm::vec3 color = glm::vec3((nc.at<float>(0)+1)/2, (-nc.at<float>(1)+1)/2, inv_d_c*param->znear);
            glUniform3fv(glGetUniformLocation(patchShader, "aColor"), 1, glm::value_ptr(color));

            // model matrix of the plane
            cv::Mat pos = mpPoint->getWorldPos();
            pos = pos /(-mpPoint->d * pos.dot(nw)); /* n'x + d = 0 */
            glm::mat4 model = glm::translate(glm::mat4(), Converter::toGlmVec3(pos));
            glm::vec3 n = glm::vec3(0.f, 0.f, 1.f);
            glm::vec3 n_ = Converter::toGlmVec3(nw);
            model = glm::rotate(model, glm::acos(glm::dot(n, n_)), glm::cross(n, n_));
            glUniformMatrix4fv(model_location, 1, GL_FALSE, glm::value_ptr(model));
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        }

        // set uniform
        {
            glUseProgram(eventShader);

            glm::vec3 w_ = Converter::toGlmVec3(kf->getAngularVelocity());
            glUniform3fv(w_location, 1, glm::value_ptr(w_));

            glm::vec3 v_ = Converter::toGlmVec3(kf->getLinearVelocity());
            glUniform3fv(v_location, 1, glm::value_ptr(v_));

            glm::vec3 wc1c2_ = Converter::toGlmVec3(rotm2axang(Rwc1.t() * Rwc));
            glUniform3fv(wc1c2_location, 1, glm::value_ptr(wc1c2_));


            glm::vec3 tc1c2_ = Converter::toGlmVec3(Rwc1.t() * (twc - twc1));
            glUniform3fv(tc1c2_location, 1, glm::value_ptr(tc1c2_));

//            LOG(INFO) << kf->getAngularVelocity();
//            LOG(INFO) << kf->getLinearVelocity();
//            LOG(INFO) << Rwc;
//            LOG(INFO) << twc;
//            LOG(INFO) << Rwc1;
//            LOG(INFO) << Rwc1.t();
//            LOG(INFO) << twc1;
//            LOG(INFO) << rotm2axang(Rwc1.t() * Rwc);
//            LOG(INFO) << Rwc1.t() * (twc - twc1);
        }

        // draw
        {
            glBindTexture(GL_TEXTURE_RECTANGLE, patchOcclusion);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);

            glEnable(GL_BLEND);
            glDisable(GL_DEPTH_TEST);
            glBindVertexArray(kf->vao);
            glDrawArrays(GL_POINTS, 0, param->window_size);
            glDisable(GL_BLEND);
        }
    }
}

void MapDrawer::draw_map_texture(GLuint& fbo) {
    cv::Mat twc = frame->getTranslation();
    cv::Mat Rwc = frame->getRotation();
    draw_map_texture(Rwc, twc, fbo);
}

float MapDrawer::tracking_cost_func(cv::Mat& Rwc, cv::Mat& twc, cv::Mat& w, cv::Mat& v) {

    draw_map_patch(Rwc, twc);
    draw_map_texture(Rwc, twc, warpFramebuffer);

    glBindTexture(GL_TEXTURE_RECTANGLE, patchOcclusion);

    glUseProgram(eventShader);

    glm::vec3 w_ = Converter::toGlmVec3(w);
    glUniform3fv(w_location, 1, glm::value_ptr(w_));
    glm::vec3 v_ = Converter::toGlmVec3(v);
    glUniform3fv(v_location, 1, glm::value_ptr(v_));
    glm::vec3 wc1c2, tc1c2;
    glUniform3fv(wc1c2_location, 1, glm::value_ptr(wc1c2));
    glUniform3fv(tc1c2_location, 1, glm::value_ptr(tc1c2));

    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    glBindVertexArray(frame->vao);
    glDrawArrays(GL_POINTS, 0, frame->events());

    glDisable(GL_BLEND);
    // compute cost
    return contrast(warpFramebuffer);
}

float MapDrawer::tracking_cost_func(cv::Mat& w, cv::Mat& v) {

    glBindFramebuffer(GL_FRAMEBUFFER, warpFramebuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // draw map on the current frame
    glBindTexture(GL_TEXTURE_RECTANGLE, mapImage);
    glUseProgram(quadShader);
    drawQuad();

    glBindTexture(GL_TEXTURE_RECTANGLE, patchOcclusion);

    glUseProgram(eventShader);

    glm::vec3 w_ = Converter::toGlmVec3(w);
    glUniform3fv(w_location, 1, glm::value_ptr(w_));
    glm::vec3 v_ = Converter::toGlmVec3(v);
    glUniform3fv(v_location, 1, glm::value_ptr(v_));
    glm::vec3 wc1c2, tc1c2;
    glUniform3fv(wc1c2_location, 1, glm::value_ptr(wc1c2));
    glUniform3fv(tc1c2_location, 1, glm::value_ptr(tc1c2));

    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    glBindVertexArray(frame->vao);
    glDrawArrays(GL_POINTS, 0, frame->events());

    glDisable(GL_BLEND);
    // compute cost
    return contrast(warpFramebuffer);
}

float MapDrawer::initialize_map_draw(cv::Mat& nws, std::vector<float>& inv_d_ws, cv::Mat& w, cv::Mat& v) {
    // camera view matrix
    cv::Mat R = frame->getRotation();
    cv::Mat t = frame->getTranslation();
    glm::mat4 view = toView(R, t);

    glUseProgram(patchShader);
    glBindVertexArray(patchVAO);
    glBindFramebuffer(GL_FRAMEBUFFER, patchFramebuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUniformMatrix4fv(view_location, 1, GL_FALSE, glm::value_ptr(view));
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    int i = 0;
    for (auto mpPoint: frame->mvpMapPoints) {
        // color stores normal and distance information of the plane
        // -1 < x, y < 1, -1/znear < inverse_d < 1/zfar ??
        cv::Mat nw = nws.col(i);
        cv::Mat nc = R.t() * nw;
        float inv_d_c = 1.f/(1.f/inv_d_ws[i] + t.dot(nw));
        glm::vec3 color = glm::vec3((nc.at<float>(0)+1)/2, (-nc.at<float>(1)+1)/2, inv_d_c*param->znear);
        glUniform3fv(glGetUniformLocation(patchShader, "aColor"), 1, glm::value_ptr(color));

        // model matrix of the plane
        cv::Mat pos = mpPoint->getWorldPos();
        pos = pos /(-inv_d_ws[i] * pos.dot(nw)); /* n'x + d = 0 */
        glm::mat4 model = glm::translate(glm::mat4(), Converter::toGlmVec3(pos));
        glm::vec3 n = glm::vec3(0.f, 0.f, 1.f);
        glm::vec3 n_ = Converter::toGlmVec3(nw);
        model = glm::rotate(model, glm::acos(glm::dot(n, n_)), glm::cross(n, n_));
        glUniformMatrix4fv(model_location, 1, GL_FALSE, glm::value_ptr(model));
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        i++;
    }

    float cost = cost_func(w, v);
    return cost;
}

void MapDrawer::draw_map_patch(cv::Mat& Rwc, cv::Mat& twc) {
    glm::mat4 view = toView(Rwc, twc);

    glUseProgram(patchShader);
    glBindVertexArray(patchVAO);
    glBindFramebuffer(GL_FRAMEBUFFER, patchFramebuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUniformMatrix4fv(view_location, 1, GL_FALSE, glm::value_ptr(view));
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    for (auto mpPoint: map->mspMapPoints) {
        // color stores normal and distance information of the plane
        // -1 < x, y < 1, -1/znear < inverse_d < 1/zfar ??

        cv::Mat nw = mpPoint->getNormal();
        cv::Mat nc = Rwc.t() * nw;
        float inv_d_c = 1.f/(1.f/mpPoint->d + twc.dot(nw));
        glm::vec3 color = glm::vec3((nc.at<float>(0)+1)/2, (-nc.at<float>(1)+1)/2, inv_d_c*param->znear);
        glUniform3fv(glGetUniformLocation(patchShader, "aColor"), 1, glm::value_ptr(color));

        // model matrix of the plane
        cv::Mat pos = mpPoint->getWorldPos();
        glm::mat4 model = glm::translate(glm::mat4(), Converter::toGlmVec3(pos));
        glm::vec3 n = glm::vec3(0.f, 0.f, 1.f);
        glm::vec3 n_ = Converter::toGlmVec3(nw);
        model = glm::rotate(model, glm::acos(glm::dot(n, n_)), glm::cross(n, n_));
        glUniformMatrix4fv(model_location, 1, GL_FALSE, glm::value_ptr(model));
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }
}

void MapDrawer::draw_map_patch() {
    cv::Mat t = frame->getTranslation();
    cv::Mat R = frame->getRotation();
    draw_map_patch(R, t);
}

float MapDrawer::optimize_map_draw(paramSet* p, cv::Mat& nws, std::vector<float>& depths,
                                   cv::Mat& Rwc_w, cv::Mat& twc, cv::Mat& ws, cv::Mat& vs) {
    glBindFramebuffer(GL_FRAMEBUFFER, mapFramebuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    int nFs = ws.cols;
    int fi = 0;

    // pose of current frame
    cv::Mat R_w = Rwc_w.col(nFs - 1);
    cv::Mat R = axang2rotm(R_w);
    cv::Mat t = twc.col(nFs - 1);
    glm::mat4 view;

    for (auto kf: *(p->KFs)) {
        // pose of keyframe
        cv::Mat R_w1 = Rwc_w.col(fi);
        cv::Mat R1 = axang2rotm(R_w1);
        cv::Mat t1 = twc.col(fi);
        view = toView(R1, t1);
        glUseProgram(patchShader);
        glBindVertexArray(patchVAO);
        glBindFramebuffer(GL_FRAMEBUFFER, patchFramebuffer);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUniformMatrix4fv(view_location, 1, GL_FALSE, glm::value_ptr(view));
        glEnable(GL_DEPTH_TEST);

        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        int mi = 0;
        for (auto pMP: (*p->MPs)) {
            if (pMP->getObservations().count(kf) != 0) {
                cv::Mat nw = nws.col(mi);
                cv::Mat nc = R1.t() * nw;
                float d = 1.f/(1.f/depths[mi] + t1.dot(nw));
                glm::vec3 color = glm::vec3((nc.at<float>(0)+1)/2, (-nc.at<float>(1)+1)/2, d*param->znear);
                glUniform3fv(glGetUniformLocation(patchShader, "aColor"), 1, glm::value_ptr(color));

                cv::Mat pos = pMP->getWorldPos();
                pos = pos /(-depths[mi] * pos.dot(nw));
                glm::mat4 model = glm::translate(glm::mat4(), Converter::toGlmVec3(pos));
                glm::vec3 n = glm::vec3(0.f, 0.f, 1.f);
                glm::vec3 n_ = Converter::toGlmVec3(nw);
                model = glm::rotate(model, glm::acos(glm::dot(n, n_)), glm::cross(n, n_));
                glUniformMatrix4fv(model_location, 1, GL_FALSE, glm::value_ptr(model));
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            }
            mi++;
        }


        // set uniform
        {
            glUseProgram(eventShader);

            glm::vec3 w_ = Converter::toGlmVec3(ws.col(fi));
            glUniform3fv(w_location, 1, glm::value_ptr(w_));

            glm::vec3 v_ = Converter::toGlmVec3(vs.col(fi));
            glUniform3fv(v_location, 1, glm::value_ptr(v_));

            glm::vec3 wc1c2_ = Converter::toGlmVec3(rotm2axang(R1.t() * R));
            glUniform3fv(wc1c2_location, 1, glm::value_ptr(wc1c2_));

            glm::vec3 tc1c2_ = Converter::toGlmVec3(R1.t() * (t - t1));
            glUniform3fv(tc1c2_location, 1, glm::value_ptr(tc1c2_));

//            LOG(INFO) << ws.col(fi);
//            LOG(INFO) << vs.col(fi);
//            LOG(INFO) << R;
//            LOG(INFO) << t;
//            LOG(INFO) << R1;
//            LOG(INFO) << R1.t();
//            LOG(INFO) << t1;
//            LOG(INFO) <<rotm2axang(R1.t() * R);
//            LOG(INFO) << R1.t() * (t - t1);
        }

        // draw
        {
            glBindTexture(GL_TEXTURE_RECTANGLE, patchOcclusion);
            glBindFramebuffer(GL_FRAMEBUFFER, mapFramebuffer);

            glEnable(GL_BLEND);
            glDisable(GL_DEPTH_TEST);
            glBindVertexArray(kf->vao);
            glDrawArrays(GL_POINTS, 0, param->window_size);
            glDisable(GL_BLEND);
        }
        fi++;    
    }

//    draw_map_texture(R, t, mapFramebuffer);

    view = toView(R, t);

    glUseProgram(patchShader);
    glBindVertexArray(patchVAO);
    glBindFramebuffer(GL_FRAMEBUFFER, patchFramebuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUniformMatrix4fv(view_location, 1, GL_FALSE, glm::value_ptr(view));
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    int mi = 0;
    for (auto pMP: (*p->MPs)) {
        cv::Mat nw = nws.col(mi);
        cv::Mat nc = R.t() * nw;
        float d = 1.f/(1.f/depths[mi] + t.dot(nw));
        glm::vec3 color = glm::vec3((nc.at<float>(0)+1)/2, (-nc.at<float>(1)+1)/2, d*param->znear);
        glUniform3fv(glGetUniformLocation(patchShader, "aColor"), 1, glm::value_ptr(color));

        cv::Mat pos = pMP->getWorldPos();
        pos = pos /(-depths[mi] * pos.dot(nw));
        glm::mat4 model = glm::translate(glm::mat4(), Converter::toGlmVec3(pos));
        glm::vec3 n = glm::vec3(0.f, 0.f, 1.f);
        glm::vec3 n_ = Converter::toGlmVec3(nw);
        model = glm::rotate(model, glm::acos(glm::dot(n, n_)), glm::cross(n, n_));
        glUniformMatrix4fv(model_location, 1, GL_FALSE, glm::value_ptr(model));
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        mi++;
    }

    // set uniform
    {
        glUseProgram(eventShader);

        glm::vec3 w_ = Converter::toGlmVec3(ws.col(fi));
        glUniform3fv(w_location, 1, glm::value_ptr(w_));

        glm::vec3 v_ = Converter::toGlmVec3(vs.col(fi));
        glUniform3fv(v_location, 1, glm::value_ptr(v_));

        glm::vec3 wc1c2_, tc1c2_;
        glUniform3fv(wc1c2_location, 1, glm::value_ptr(wc1c2_));
        glUniform3fv(tc1c2_location, 1, glm::value_ptr(tc1c2_));

    }

    // draw
    {
        glUseProgram(eventShader);
        glBindTexture(GL_TEXTURE_RECTANGLE, patchOcclusion);
        glBindFramebuffer(GL_FRAMEBUFFER, tmpFramebuffer);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        glBindVertexArray(frame->vao);
        glDrawArrays(GL_POINTS, 0, param->window_size);
        glDisable(GL_BLEND);
    }

    if (p->optimize) {
        glBindFramebuffer(GL_FRAMEBUFFER, warpFramebuffer);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_BLEND);

        glBindTexture(GL_TEXTURE_RECTANGLE, tmpImage);
        glUseProgram(quadShader);
        drawQuad();
        glBindTexture(GL_TEXTURE_RECTANGLE, mapImage);
        drawQuad();
        glDisable(GL_BLEND);

//        drawImage(warppedImage);

        return contrast(warpFramebuffer);
    } else {        
        return 0.f;
    }
}

void MapDrawer::visualize_map(){
    draw_map_texture(tmpFramebuffer);
    draw_map_patch();
    set_use_polarity(false);

    {
        glBindFramebuffer(GL_FRAMEBUFFER, tmpFramebuffer);

        glBindTexture(GL_TEXTURE_RECTANGLE, patchOcclusion);
        glUseProgram(eventShader);

        glm::vec3 w_ = Converter::toGlmVec3(frame->getAngularVelocity());
        glUniform3fv(w_location, 1, glm::value_ptr(w_));
        glm::vec3 v_ = Converter::toGlmVec3(frame->getLinearVelocity());
        glUniform3fv(v_location, 1, glm::value_ptr(v_));

        glm::vec3 wc1c2_, tc1c2_;
        glUniform3fv(wc1c2_location, 1, glm::value_ptr(wc1c2_));
        glUniform3fv(tc1c2_location, 1, glm::value_ptr(tc1c2_));

        glEnable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);

        glBindVertexArray(frame->vao);
        glDrawArrays(GL_POINTS, 0, frame->events());

        glDisable(GL_BLEND);
    }

    {
        gaussianBlur(tmpImage, glm::vec2(0, 1));
        std::swap(blurFramebuffer, tmpFramebuffer);
        std::swap(blurredImage, tmpImage);
        gaussianBlur(tmpImage, glm::vec2(1, 0));
    }

    {
        cv::Mat R = frame->getRotation();
        cv::Mat t = frame->getTranslation();
        glm::mat4 view = toView(R, t);

        glUseProgram(patchShader);
        glBindVertexArray(patchVAO);

        glUniformMatrix4fv(view_location, 1, GL_FALSE, glm::value_ptr(view));

        glBindFramebuffer(GL_FRAMEBUFFER, tmpFramebuffer);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        for (auto mpPoint: map->mspMapPoints) {
            // color stores normal and distance information of the plane
            // -1 < x, y < 1, -1/znear < inverse_d < 1/zfar ??

            cv::Mat nw = mpPoint->getNormal();
            cv::Mat nc = R.t() * nw;
            glm::vec3 color = glm::vec3((nc.at<float>(0)+1)/2, (-nc.at<float>(1)+1)/2, -nc.at<float>(2));
            glUniform3fv(glGetUniformLocation(patchShader, "aColor"), 1, glm::value_ptr(color));

            // model matrix of the plane
            cv::Mat pos = mpPoint->getWorldPos();
            glm::mat4 model = glm::translate(glm::mat4(), Converter::toGlmVec3(pos));
            glm::vec3 n = glm::vec3(0.f, 0.f, 1.f);
            glm::vec3 n_ = Converter::toGlmVec3(nw);
            model = glm::rotate(model, glm::acos(glm::dot(n, n_)), glm::cross(n, n_));
            glUniformMatrix4fv(model_location, 1, GL_FALSE, glm::value_ptr(model));
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        }
        glDisable(GL_DEPTH_TEST);

        glEnable(GL_BLEND);
        glBindTexture(GL_TEXTURE_RECTANGLE, blurredImage);
        glUseProgram(quadShader);
        drawQuad();
        glDisable(GL_BLEND);
    }
    drawImage(tmpImage);
}

void MapDrawer::drawImage(GLuint& image) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_RECTANGLE, image);
    glUseProgram(quadShader);
    drawQuad();
    glfwSwapBuffers(window);
    glfwPollEvents();
}

glm::mat4 MapDrawer::toView(cv::Mat& Rwc, cv::Mat& twc) {
    glm::mat4 view;
    cv::Mat axang = rotm2axang(Rwc);
    glm::vec3 axang_ = -Converter::toGlmVec3(axang);
    float angle = glm::length(axang_);
    if (std::abs(angle) > 1e-6) {
        glm::vec3 axis = glm::normalize(axang_);
        view = glm::rotate(view, angle, axis);
    }
    view = glm::translate(view, -Converter::toGlmVec3(twc));
    return view;
}

glm::mat4 MapDrawer::toView(cv::Mat& Twc) {
    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
    cv::Mat twc = Twc.rowRange(0,3).col(3);
    return toView(Rwc, twc);
}

bool MapDrawer::inFrame(cv::Mat Xw, cv::Mat& Rwc, cv::Mat& twc) {
    cv::Mat Xc = Rwc.t() * (Xw - twc);
    if (Xc.at<float>(2) < param->znear)
        return false;
    cv::Mat xc = param->K_n * Xc;
    float x = xc.at<float>(0) / xc.at<float>(2);
    float y = xc.at<float>(1) / xc.at<float>(2);
    return (x >= 0 && x  < param->width && y >= 0 && y < param->height);
}

}
