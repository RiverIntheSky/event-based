#include "MapDrawer.h"
#include <chrono>
namespace ev {
typedef void (*framebuffer_size_callback)(void);
void MapDrawer::drawMapPoints() {
    // fixed size
    int size[] = {param->height, param->width, 3};
    framePartition = cv::Mat(3, size, CV_32F);

    setUp();

    LOG(INFO) << "drawwwwwwwwwwwwwwwwwwwww";

    while(!glfwWindowShouldClose(window)) {
        if (map->isDirty) {

            frame = tracking->getCurrentFrame().get();

            // camera view matrix
            cv::Mat t = frame->getTranslation();
            glm::mat4 view = glm::translate(glm::mat4(), Converter::toGlmVec3(t));
            cv::Mat R = frame->getRotation();
            cv::Mat axang = rotm2axang(R);
            glm::vec3 axang_ = Converter::toGlmVec3(axang);
            float angle = glm::length(axang_);
            if (std::abs(angle) > 1e-6) {
                glm::vec3 axis = glm::normalize(axang_);
                view = glm::rotate(view, -angle, axis);
            }

            glUseProgram(patchShader);
            glUniformMatrix4fv(view_location, 1, GL_FALSE, glm::value_ptr(view));
            glEnable(GL_DEPTH_TEST);

// draw plane only when facing camera??
//            glEnable(GL_CULL_FACE);
//            glCullFace(GL_BACK);

            glBindVertexArray(patchVAO);
            glBindFramebuffer(GL_FRAMEBUFFER, patchFramebuffer);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            for (auto mpPoint: map->mspMapPoints) {

                // color stores normal and distance information of the plane
                // -1 < x, y < 1, -1/znear < inverse_d < 1/zfar ??
                cv::Mat nw = mpPoint->getNormal();
                cv::Mat nc = frame->getRotation().t() * nw;
                float inverse_Depth = 1.f/(1.f/mpPoint->d + frame->getTranslation().dot(nw));
                glm::vec3 color = glm::vec3((nc.at<double>(0)+1)/2, (-nc.at<double>(1)+1)/2, (inverse_Depth*param->znear+1)/2);
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

            updateEvents();

            glBindTexture(GL_TEXTURE_RECTANGLE, patchOcclusion);
            glUseProgram(eventShader);

            glm::vec3 w = Converter::toGlmVec3(frame->getAngularVelocity());
            glUniform3fv(w_location, 1, glm::value_ptr(w));
            glm::vec3 v = Converter::toGlmVec3(frame->getLinearVelocity());
            glUniform3fv(v_location, 1, glm::value_ptr(v));

            glBindFramebuffer(GL_FRAMEBUFFER, warpFramebuffer);
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE);
            glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glDisable(GL_DEPTH_TEST);

            glDrawArrays(GL_POINTS, 0, frame->events());

            glDisable(GL_BLEND);

            gaussianBlur(warpFramebuffer, warppedImage, blurFramebuffer, blurredImage, glm::vec2(0, 1));
            std::swap(blurFramebuffer, warpFramebuffer);
            std::swap(warppedImage, blurredImage);
            gaussianBlur(warpFramebuffer, warppedImage, blurFramebuffer, blurredImage, glm::vec2(1, 0));

            glEnable(GL_BLEND);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glBindTexture(GL_TEXTURE_RECTANGLE, blurredImage);
            drawQuad();
            glBindTexture(GL_TEXTURE_RECTANGLE, patchOcclusion);
            drawQuad();
            glDisable(GL_BLEND);

            glfwSwapBuffers(window);
            glfwPollEvents();

            // we could also read GL_DEPTH_COMPONENT here, but we don't,
            // since the depth value might change after the warp, but distance to the plane won't

            std::vector<GLfloat> data(param->width * param->height * 3);
            glReadPixels(0, 0, param->width, param->height, GL_RGB, GL_FLOAT, &data[0]);
            for (int j = param->height-1; j >= 0; j--) {
                for (int i = 0; i < param->width; i++) {
//                    LOG(INFO) << data[3*(i+param->width*j)];
//                    LOG(INFO) << data[3*(i+param->width*j)+1];
//                    LOG(INFO) << data[3*(i+param->width*j)+2];
                    framePartition.at<float>(j, i, 0) = data[3*(i+param->width*j)] * 2 - 1; // x
                    framePartition.at<float>(j, i, 1) = data[3*(i+param->width*j)+1] * 2 - 1; // y
                    framePartition.at<float>(j, i, 2) = (data[3*(i+param->width*j)+2] * 2 - 1) * param->zfar; // d
                }
            }
//            map->isDirty = false;
        }
//        std::this_thread::yield();
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

    glGenVertexArrays(1, &patchVAO);
    glBindVertexArray(patchVAO);

    glGenBuffers(1, &patchVBO);
    glBindBuffer(GL_ARRAY_BUFFER, patchVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);

    glGenBuffers(1, &patchEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, patchEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    std::string patch_vsh_path = shaderFilePath + "patch.vsh",
            patch_fsh_path = shaderFilePath + "patch.fsh";
    const char* patchVSFile = patch_vsh_path.c_str();
    const char* patchFSFile = patch_fsh_path.c_str();
    patchVS = createShader(GL_VERTEX_SHADER, patchVSFile);
    patchFS = createShader(GL_FRAGMENT_SHADER, patchFSFile);

    patchShader = glCreateProgram();
    glAttachShader(patchShader, patchVS);
    glAttachShader(patchShader, patchFS);

    glLinkProgram(patchShader);
    glUseProgram(patchShader);
    glDeleteShader(patchVS);
    glDeleteShader(patchFS);

    apos_location = glGetAttribLocation(patchShader, "aPos");
    glEnableVertexAttribArray(apos_location);
    glVertexAttribPointer(apos_location, 2, GL_FLOAT, GL_FALSE, 0 * sizeof(float), (void*)0);

    // patch framebuffer
    glGenFramebuffers(1, &patchFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, patchFramebuffer);
    glGenTextures(1, &patchOcclusion);
    glBindTexture(GL_TEXTURE_RECTANGLE, patchOcclusion);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB8, param->width, param->height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, 0);

    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_RECTANGLE,
                           patchOcclusion,
                           0);

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

void MapDrawer::setUpEventShader() {
    glGenVertexArrays(1, &eventVAO);
    glBindVertexArray(eventVAO);

    glGenBuffers(1, &eventVBO);
    glBindBuffer(GL_ARRAY_BUFFER, eventVBO);
    glBufferData(GL_ARRAY_BUFFER, param->window_size * sizeof(event), NULL, GL_DYNAMIC_DRAW);

    std::string event_vsh_path = shaderFilePath + "event.vsh",
                event_fsh_path = shaderFilePath + "event.fsh";
    const char* eventVSFile = event_vsh_path.c_str();
    const char* eventFSFile = event_fsh_path.c_str();
    eventVS = createShader(GL_VERTEX_SHADER, eventVSFile);
    eventFS = createShader(GL_FRAGMENT_SHADER, eventFSFile);

    eventShader = glCreateProgram();
    glAttachShader(eventShader, eventVS);
    glAttachShader(eventShader, eventFS);

    glLinkProgram(eventShader);
    glUseProgram(eventShader);
    glDeleteShader(eventVS);
    glDeleteShader(eventFS);

    event_apos_location = glGetAttribLocation(eventShader, "aPos");
    glEnableVertexAttribArray(event_apos_location);
    glVertexAttribPointer(event_apos_location, 4, GL_FLOAT, GL_FALSE, 0 * sizeof(event), (void*)0);

    w_location = glGetUniformLocation(eventShader, "w");
    v_location = glGetUniformLocation(eventShader, "v");
    camera_matrix_location = glGetUniformLocation(eventShader, "cameraMatrix");
    glm::mat3 K;
    K[0][0] = param->fx;
    K[1][1] = param->fy;
    K[2][0] = param->cx;
    K[2][1] = param->cy;
    glUniformMatrix3fv(camera_matrix_location, 1, GL_FALSE, glm::value_ptr(K));

    near_plane_location = glGetUniformLocation(eventShader, "nearPlane");
    glUniform1f(near_plane_location, param->znear);

    occlusion_map_location = glGetUniformLocation(eventShader, "patchTexture");
    glUniform1i(occlusion_map_location, 0);

    glGenFramebuffers(1, &warpFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, warpFramebuffer);
    glGenTextures(1, &warppedImage);
    glBindTexture(GL_TEXTURE_RECTANGLE, warppedImage);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB8, param->width, param->height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, 0);

    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_RECTANGLE,
                           warppedImage,
                           0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
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

    glGenVertexArrays(1, &quadVAO);
    glBindVertexArray(quadVAO);

    glGenBuffers(1, &quadVBO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

    glGenBuffers(1, &quadEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

//    std::string quad_vsh_path = shaderFilePath + "screen.vsh",
//                quad_fsh_path = shaderFilePath + "screen.fsh";
//    const char* quadVSFile = quad_vsh_path.c_str();
//    const char* quadFSFile = quad_fsh_path.c_str();
//    quadVS = createShader(GL_VERTEX_SHADER, quadVSFile);
//    quadFS = createShader(GL_FRAGMENT_SHADER, quadFSFile);

//    quadShader = glCreateProgram();
//    glAttachShader(quadShader, quadVS);
//    glAttachShader(quadShader, quadFS);

//    glLinkProgram(quadShader);
//    glUseProgram(quadShader);
//    glDeleteShader(quadVS);
//    glDeleteShader(quadFS);

//    atex_location = glGetUniformLocation(quadShader, "tex");
//    glUniform1i(atex_location, 0);
}

void MapDrawer::setUpGaussianBlurShader() {
    glGenFramebuffers(1, &blurFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, blurFramebuffer);
    glGenTextures(1, &blurredImage);
    glBindTexture(GL_TEXTURE_RECTANGLE, blurredImage);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB8, param->width, param->height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, 0);

    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_RECTANGLE,
                           blurredImage,
                           0);

    std::string blur_vsh_path = shaderFilePath + "gaussian_blur.vsh",
            blur_fsh_path = shaderFilePath + "gaussian_blur.fsh";
    const char* blurVSFile = blur_vsh_path.c_str();
    const char* blurFSFile = blur_fsh_path.c_str();
    blurVS = createShader(GL_VERTEX_SHADER, blurVSFile);
    blurFS = createShader(GL_FRAGMENT_SHADER, blurFSFile);

    blurShader = glCreateProgram();
    glAttachShader(blurShader, blurVS);
    glAttachShader(blurShader, blurFS);

    glLinkProgram(blurShader);
    glUseProgram(blurShader);
    glDeleteShader(blurVS);
    glDeleteShader(blurFS);

    atex_location = glGetUniformLocation(blurShader, "tex");
    glUniform1i(atex_location, 0);
    dir_location = glGetUniformLocation(blurShader, "dir");
    blur_apos_location = glGetAttribLocation(blurShader, "aPos");
    glEnableVertexAttribArray(blur_apos_location);
    glVertexAttribPointer(blur_apos_location, 2, GL_FLOAT, GL_FALSE, 0 * sizeof(float), (void*)0);
}

void MapDrawer::drawQuad() {
//    glUseProgram(quadShader);
    glBindVertexArray(quadVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void MapDrawer::gaussianBlur(GLuint& imageFBO, GLuint& imageTex, GLuint& blurredFBO, GLuint& blurredTex, glm::vec2 dir) {
    glBindFramebuffer(GL_FRAMEBUFFER, blurredFBO);
    glBindTexture(GL_TEXTURE_RECTANGLE, imageTex);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(blurShader);
    glUniform2fv(dir_location, 1, glm::value_ptr(dir));

    drawQuad();
}

void MapDrawer::updateEvents() {

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

    glBindVertexArray(eventVAO);
    glBindBuffer(GL_ARRAY_BUFFER, eventVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, events.size() * sizeof(event), &events[0]);
}

//void MapDrawer::framebuffer_size_callback(GLFWwindow* window, int width, int height){
//    //lock framebuffer size
//    glViewport(0, 0, param->width, param->height);
//}
}
