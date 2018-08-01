#version 430 core

uniform sampler2DRect tex;

out vec4 fColor;

void main()
{
    fColor = 4.f * texture(tex, gl_FragCoord.xy * 2);
}
