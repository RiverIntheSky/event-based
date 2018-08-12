#version 330 core
layout (points) in;
layout (points, max_vertices = 1) out;

in VS_OUT {
    float vColor;
    int width;
    int height;
} gs_in[];

out float gColor;

void main() {
    int w = gs_in[0].width;
    int h = gs_in[0].height;
    float c = gs_in[0].vColor;
    float x = gl_in[0].gl_Position.x;
    float y = gl_in[0].gl_Position.y;
    float x1 = floor(x);
    float x2 = x1 + 1;
    float y1 = floor(y);
    float y2 = y1 + 1;

    gl_Position = gl_in[0].gl_Position;
    gColor = c;
    EmitVertex();

    EndPrimitive();
}
