#pragma once

#include <glad/glad.h>
#include <fstream>
#include <stdio.h>

namespace ev {
bool ReadShaderFile(const char* pFileName, std::string& outFile);
bool createShader(GLenum type, const char* pFileName, GLuint &shader);
}
