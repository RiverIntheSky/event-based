#version 430 core

in vec3 vColor;

out vec4 fColor;

void main()
{
    fColor = vec4(vColor, 1.0);
}
