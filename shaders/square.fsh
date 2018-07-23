#version 430 core

uniform sampler2DRect tex;

out vec4 fColor;

void main()
{
    float color = texelFetch(tex, ivec2(gl_FragCoord.xy)).r;
    color = pow(color, 2);
    fColor = vec4(vec3(color), 1.f);
}
