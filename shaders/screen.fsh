#version 330 core

uniform sampler2DRect tex;

out vec4 fColor;

void main()
{
    fColor = texture(tex, gl_FragCoord.xy);
}
