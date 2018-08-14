#version 430 core

in float vColor;

out vec4 fColor;

void main()
{
    fColor = vec4(vec3(vColor), 1.f);
}
