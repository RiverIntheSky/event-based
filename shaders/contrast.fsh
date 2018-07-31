#version 430 core

uniform sampler2DRect tex;

out vec4 fColor;

void main()
{
    float o = 2./3;
    float w1 = -2.25;
    float w2 = 9;
    fColor = texture(tex, gl_FragCoord.xy) * w2;
    fColor += texture(tex, vec2(gl_FragCoord.x + o, gl_FragCoord.y + o)) * w1;
    fColor += texture(tex, vec2(gl_FragCoord.x + o, gl_FragCoord.y - o)) * w1;
    fColor += texture(tex, vec2(gl_FragCoord.x - o, gl_FragCoord.y - o)) * w1;
    fColor += texture(tex, vec2(gl_FragCoord.x - o, gl_FragCoord.y + o)) * w1;
}
