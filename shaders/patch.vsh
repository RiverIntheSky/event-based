#version 430 core

in vec2 aPos;

uniform vec3 aColor;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 vColor;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 0.f, 1.f);
    vColor = aColor;
}
