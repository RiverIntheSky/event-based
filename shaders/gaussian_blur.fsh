#version 430 core

uniform sampler2DRect tex;
uniform vec2 dir;

out vec4 fColor;

// http://rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/
// sigma = 1; kerner_size = 7;
vec4 blur7(sampler2DRect image, vec2 coord, vec2 dir) {
    vec2 off = vec2(0.548137238122394, 2.075858180021243);
    vec2 weight = vec2(0.441561369202342, 0.058438630797658);
    vec4 color = vec4(0.0);
    color += texture(image, coord + off[0] * dir) * weight[0];
    color += texture(image, coord + off[1] * dir) * weight[1];
    color += texture(image, coord - off[0] * dir) * weight[0];
    color += texture(image, coord - off[1] * dir) * weight[1];
    return color;
}

void main()
{
    fColor = blur7(tex, gl_FragCoord.xy, dir);
}
