#version 430 core

in vec4 aPos;

uniform vec3 w;
uniform vec3 v;
uniform sampler2DRect patchTexture;
uniform mat3 cameraMatrix;
uniform vec3 wc1c2;
uniform vec3 tc1c2;
uniform float nearPlane;

out float vColor;

mat3 rotationMatrix(vec3 axis, float angle)
{
    if (abs(length(axis)) < 1e-6)
        return mat3(1.f);
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c           );
}

void main()
{
    int width = textureSize(patchTexture).x;
    int height = textureSize(patchTexture).y;

    vec3 Xc1_ = vec3(aPos.x, aPos.y, 1.f);
    vec3 xc1_ = cameraMatrix * Xc1_;
    float polarity = aPos.z * 2 - 1;
    float t = aPos.w;
    vec3 pixel = texture(patchTexture, vec2(xc1_.x, height - xc1_.y)).xyz;
    float dc1 = pixel.z / nearPlane;
    float nx = pixel.x * 2 - 1;
    float ny = pixel.y * 2 - 1;
    float nz = sqrt(1 - nx * nx - ny * ny);
    mat3 Rc1_c1 = rotationMatrix(w, t * length(w)); // Rc1_c1
    vec3 nc1 = vec3(nx, ny, nz);
    mat3 Hc1_c1 = Rc1_c1 * (mat3(1.f) + outerProduct(v * t * dc1, nc1));

    vec3 Xc1 = inverse(Hc1_c1) * vec3(aPos.x, -aPos.y, -1.f); // keyframe projection
    mat3 Rc2c1 = rotationMatrix(-wc1c2, length(wc1c2));
    vec3 nc2 = Rc2c1 * nc1;
    float dc2 = 1.f / (1.f/dc1 + dot(tc1c2, nc1);
    mat3 Hc2c1 = Rc2c1 * (mat3(1.f) + outerProduct(-tc1c2 * dc2, nc2));
    vec3 Xc2 = inverse(Hc2c1) * Xc1;
    vec3 xc2 = cameraMatrix * vec3(-Xc2.x / Xc2.z,
                                            Xc2.y / Xc2.z,
                                            1.f);
    gl_Position = vec4((xc2.x-width/2)/(width/2), (height/2-xc2.y)/(height/2), 0.f, 1.f);
    vColor = polarity * 0.1;

}
