#include "Shader.h"
#include <iostream>

namespace ev {
bool ReadShaderFile(const char* pFileName, std::string& outFile){
    std::ifstream f(pFileName);

    if (f.is_open()) {
        std::string line;
        while (getline(f, line)) {
            outFile.append(line);
            outFile.append("\n");
        }
        f.close();
        return true;
    }
    return false;
}

bool createShader(GLenum type, const char* pFileName, GLuint& shader) {
    shader = glCreateShader(type);
    std::string src_;
    if (ReadShaderFile(pFileName, src_)) {
        const char* src = src_.c_str();
        glShaderSource(shader, 1, &src, NULL); glCompileShader(shader);

        int success;
        char infoLog[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if(!success)
        {
            glGetShaderInfoLog(shader, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
        }
        return true;
    }
    return false;
}
}
