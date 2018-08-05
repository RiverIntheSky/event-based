#version 330 core
layout (points) in;
layout (points, max_vertices = 4) out;

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

    gl_Position = vec4(2*x1/w-1, 1-2*y1/h, 0.f, 1.f);
    gColor = (x2-x)*(y2-y)*c;
    EmitVertex();

    gl_Position = vec4(2*x1/w-1, 1-2*y2/h, 0.f, 1.f);
    gColor = -(x2-x)*(y1-y)*c;
    EmitVertex();

    gl_Position = vec4(2*x2/w-1, 1-2*y1/h, 0.f, 1.f);
    gColor = -(x1-x)*(y2-y)*c;
    EmitVertex();

    gl_Position = vec4(2*x2/w-1, 1-2*y2/h, 0.f, 1.f);
    gColor = (x1-x)*(y1-y)*c;
    EmitVertex();
    EndPrimitive();
}
