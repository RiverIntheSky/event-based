#version 430 core

uniform sampler2DRect tex;
uniform float mean;

out vec4 fColor;

void main()
{
    float color = texelFetch(tex, ivec2(gl_FragCoord.xy)).r;
//    color = pow(clamp(color, -1.f, 1.f), 2);
    color = pow(clamp(color - mean, -1.f, 1.f), 2);
    fColor = vec4(vec3(color), 1.f);
}
