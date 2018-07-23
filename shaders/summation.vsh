#version 430 core

in vec2 aPos;

void main()
{
    gl_Position = vec4((aPos+1)/2-1, 0.f, 1.f);
}
