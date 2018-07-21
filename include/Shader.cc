#include "Shader.h"
#include <iostream>

namespace ev {
std::string ReadShaderFile(const char* pFileName){
    std::ifstream f(pFileName);
    std::string outFile;

    if (f.is_open()) {
        std::string line;
        while (getline(f, line)) {
            outFile.append(line);
            outFile.append("\n");
        }
        f.close();
    } else
        std::cerr << "No such file"<<std::endl;

    return outFile;
}

GLuint createShader(GLenum type, const char* pFileName) {
    GLuint shader = glCreateShader(type);
    std::string src_ = ReadShaderFile(pFileName);
    const char* src = src_.c_str();
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);

    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    return shader;
}
}
