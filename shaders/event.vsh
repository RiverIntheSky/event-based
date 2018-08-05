#version 330 core

in vec4 aPos;

uniform vec3 w;
uniform vec3 v;
uniform sampler2DRect patchTexture;
uniform mat3 cameraMatrix;
uniform vec3 wc1c2;
uniform vec3 tc1c2;
uniform float nearPlane;
uniform bool usePolarity;

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

void discard_() {
    gl_Position = vec4(-1.f, -1.f, 0.f, 0.f);
    vs_out.vColor = 0.f;
}

void main()
{
    vs_out.width = textureSize(patchTexture).x;
    vs_out.height = textureSize(patchTexture).y;

    vec3 Xc1_ = vec3(aPos.x, aPos.y, 1.f);
    vec3 xc1_ = cameraMatrix * Xc1_;

    vec3 pixel = texture(patchTexture, vec2(xc1_.x, vs_out.height - xc1_.y)).xyz;

    float nx = pixel.x * 2 - 1;
    float ny = pixel.y * 2 - 1;
    if (nx*nx+ny*ny > 1) {
        discard_();
    } else  {
        float nz = sqrt(1 - nx * nx - ny * ny);
        vec3 nc1 = vec3(nx, ny, nz);
        mat3 Rc2c1 = rotationMatrix(wc1c2, -length(wc1c2));
        vec3 nc2 = Rc2c1 * nc1;
        if (nc2.z < 0) {
            discard_();
        } else {
            float t = aPos.w;
            mat3 Rc1_c1 = rotationMatrix(w, -t * length(w)); // Rc1_c1
            float dc1 = pixel.z / nearPlane;
            float dc2 = 1.f / (1.f/dc1 + dot(tc1c2, nc1));
            if (dc2 < nearPlane) {
                discard_();
            } else {
                mat3 Hc1_c1 = Rc1_c1 * (mat3(1.f) + outerProduct(v * t * dc1, nc1));
                mat3 Hc2c1 = Rc2c1 * (mat3(1.f)+ outerProduct(tc1c2 * dc1, nc1));
                vec3 Xc2 = Hc2c1 * inverse(Hc1_c1) * vec3(aPos.x, -aPos.y, -1.f);
                vec3 xc2 = cameraMatrix * vec3(-Xc2.x / Xc2.z,
                                               Xc2.y / Xc2.z,
                                               1.f);
                gl_Position = vec4(xc2.x, xc2.y, 0.f, 1.f);
                if (usePolarity) {
                    vs_out.vColor = 0.1 * (aPos.z * 2 - 1);
                } else {
                    vs_out.vColor = 0.1;
                }
            }
        }
    }

}
