#version 330 core

in float gColor;

out vec4 fColor;

void main()
{
    fColor = vec4(vec3(gColor), 1.f);
}
