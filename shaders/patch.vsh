#version 430 core

in vec2 aPos;

uniform vec3 aColor;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 vColor;

void main()
{
    float z = -model[3][2];
    gl_Position = projection * view * model * vec4(aPos*z, 0.f, 1.f);
//    gl_Position = vec4(aPos*10, 0.f, 1.f);
    vColor = aColor;
}
