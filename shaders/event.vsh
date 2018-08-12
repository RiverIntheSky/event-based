#version 330 core

in vec4 aPos;

uniform vec3 w;
uniform vec3 v;
uniform float t;

uniform mat3 cameraMatrix;

uniform vec3 wc1c2;
uniform vec3 tc1c2;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out VS_OUT {
    float vColor;
    int width;
    int height;
} vs_out;

mat3 rotationMatrix(vec3 axis, float angle)
{
    if (abs(length(axis)) < 1e-6)
        return mat3(1.f);
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y + axis.z * s,  oc * axis.z * axis.x - axis.y * s,
                oc * axis.x * axis.y - axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z + axis.x * s,
                oc * axis.z * axis.x + axis.y * s,  oc * axis.y * axis.z - axis.x * s,  oc * axis.z * axis.z + c           );
}

void main()
{
  gl_Position = projection * model* vec4(aPos.x + (1 - t) * v.x / 0.55 * aPos.w, -aPos.y, -aPos.w * t * 2, 1.f);
vs_out.width = 240;
vs_out.height = 180;
    vs_out.vColor = 0.1 * (aPos.z * 2 - 1);

}
