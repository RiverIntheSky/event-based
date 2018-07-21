#pragma once

#include <glad/glad.h>
#include <fstream>
#include <stdio.h>

namespace ev {
std::string ReadShaderFile(const char* pFileName);
GLuint createShader(GLenum type, const char* pFileName);
}
