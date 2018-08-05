#version 430 core

uniform sampler2DRect tex0; // frame
uniform sampler2DRect tex1; // map

out vec4 fColor;

void main()
{
    fColor = vec4(0.);
    // white: match; red: mismatch
    if (abs(texture(tex0, gl_FragCoord.xy).r) > .1) {
        fColor.r = 1.;
        if (abs(texture(tex1, gl_FragCoord.xy).r) > .0) {
            fColor.g = 1.;
            fColor.b = 1.;
        }
    }
}
