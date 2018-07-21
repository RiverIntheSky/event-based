#include "MapDrawer.h"

namespace ev {
typedef void (*framebuffer_size_callback)(void);
void MapDrawer::drawMapPoints() {
    // fixed size
    int size[] = {param->height, param->width, 3};
    framePartition = cv::Mat(3, size, CV_32F);
    GLfloat points[] = {
            -0.1f,  0.1f,
             0.1f,  0.1f,
             0.1f, -0.1f,
            -0.1f, -0.1f
    };
    unsigned int indices[] = {
        0, 1, 3,
        1, 2, 3
    };
    GLFWwindow* window;
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

    GLuint VAO, VBO, EBO, vertexShader, fragmentShader, shaderProgram;
    GLint apos_location, model_location, view_location, projection_location;

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);

    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    std::string file_path(__FILE__);
    file_path = file_path.substr(0, file_path.find_last_of("/\\"));
    file_path = file_path.substr(0, file_path.find_last_of("/\\"));
    file_path += "/shaders/";
    std::string vsh_path = file_path + "shader.vsh",
                fsh_path = file_path + "shader.fsh";
    const char* vertexShaderFile = vsh_path.c_str();
    const char* fragmentShaderFile = fsh_path.c_str();
    vertexShader = createShader(GL_VERTEX_SHADER, vertexShaderFile);
    fragmentShader = createShader(GL_FRAGMENT_SHADER, fragmentShaderFile);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    apos_location = glGetAttribLocation(shaderProgram, "aPos");
    glEnableVertexAttribArray(apos_location);
    glVertexAttribPointer(apos_location, 2, GL_FLOAT, GL_FALSE, 0 * sizeof(float), (void*)0);

    glEnable(GL_DEPTH_TEST);
    // draw plane only when facing camera??
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glm::vec3 color;
    glm::mat4 projection;

    projection[0][0] = 2.f * param->fx / param->width;

    projection[1][1] = -2.f * param->fy / param->height;

    projection[2][0] = 1.f - 2.f * param->cx / param->width;
    projection[2][1] = 2.f * param->cy / param->height - 1.f;
    projection[2][2] = (param->zfar + param->znear) / (param->znear - param->zfar);
    projection[2][3] = -1.f;

    projection[3][2] = 2.f * param->zfar * param->znear / (param->znear - param->zfar);
    projection[3][3] = 0.f;

    model_location = glGetUniformLocation(shaderProgram, "model");
    view_location = glGetUniformLocation(shaderProgram, "view");
    projection_location = glGetUniformLocation(shaderProgram, "projection");
    glUniformMatrix4fv(projection_location, 1, GL_FALSE, glm::value_ptr(projection));

    LOG(INFO) << "drawwwwwwwwwwwwwwwwwwwww";

    while(!glfwWindowShouldClose(window)) {
        if (map->isDirty) {
                LOG(INFO) << "dirty";
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glUseProgram(shaderProgram);

            // camera view matrix
            auto frame = tracking->getCurrentFrame();
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

            glUniformMatrix4fv(view_location, 1, GL_FALSE, glm::value_ptr(view));

            for (auto mpPoint: map->mspMapPoints) {
                cv::Mat nw = mpPoint->getNormal();

                // color stores normal and distance information of the plane
                // -1 < x, y < 1, -zfar < d < zfar ??
                color = glm::vec3((nw.at<double>(0)+1)/2, (nw.at<double>(1)+1)/2, (mpPoint->d/param->zfar+1)/2);
                glUniform3fv(glGetUniformLocation(shaderProgram, "aColor"), 1, glm::value_ptr(color));

                // model matrix of the plane
                cv::Mat pos = mpPoint->getWorldPos();
                glm::mat4 model = glm::translate(glm::mat4(), Converter::toGlmVec3(pos));
                glm::vec3 n = glm::vec3(0.f, 0.f, 1.f);
                glm::vec3 n_ = Converter::toGlmVec3(nw);
                model = glm::rotate(model, glm::acos(glm::dot(n, n_)), glm::cross(n, n_));
                glUniformMatrix4fv(model_location, 1, GL_FALSE, glm::value_ptr(model));

                glBindVertexArray(VAO);
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            }

            glfwSwapBuffers(window);
            glfwPollEvents();

            // we could also read GL_DEPTH_COMPONENT here, but we don't,
            // since the depth value might change after the warp, but distance to the plane won't
            std::vector<GLfloat> data(param->width * param->height * 3);
            glReadPixels(0, 0, param->width, param->height, GL_RGB, GL_FLOAT, &data[0]);
            for (int j = param->height-1; j >= 0; j--) {
                for (int i = 0; i < param->width; i++) {
                    framePartition.at<float>(j, i, 0) = data[3*(i+param->width*j)] * 2 - 1; // x
                    framePartition.at<float>(j, i, 1) = data[3*(i+param->width*j)+1] * 2 - 1; // y
                    framePartition.at<float>(j, i, 2) = (data[3*(i+param->width*j)+2] * 2 - 1) * param->zfar; // d
                }
            }
            map->isDirty = false;
        }
        std::this_thread::yield();
    }
    glfwTerminate();
}

//void MapDrawer::framebuffer_size_callback(GLFWwindow* window, int width, int height){
//    //lock framebuffer size
//    glViewport(0, 0, param->width, param->height);
//}
}
