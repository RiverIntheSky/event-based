#version 330 core

in vec2 aPos;

uniform vec3 w;
uniform vec3 v;
uniform sampler2DRect patchTexture;
uniform mat3 cameraMatrix;
uniform mat4 projection;
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

    vec3 worldPos = vec3(aPos.x, -aPos.y, 1.f);
//    vec3 framePos = cameraMatrix * worldPos;
//    float polarity = aPos.z * 2 - 1;
//    float t = aPos.w;
//    vec3 pixel = texture(patchTexture, vec2(framePos.x, height - framePos.y)).xyz;
//    float inverseDepth = (pixel.z * 2 - 1) / nearPlane;
//    float nx = pixel.x * 2 - 1;
//    float ny = pixel.y * 2 - 1;
//    float nz = sqrt(1 - nx * nx - ny * ny);
//    mat3 R = rotationMatrix(w, -t * length(w));
//    mat3 H = R * (mat3(1.f) + outerProduct(v * t * inverseDepth, vec3(nx, ny, nz)));
//    vec3 newWorldPos = inverse(H) * worldPos;
    vec3 newWorldPos =  worldPos;
    vec3 newFramePos = cameraMatrix * vec3(newWorldPos.x / newWorldPos.z,
                                          -newWorldPos.y / newWorldPos.z,
                                           1.f);
    gl_Position = vec4((newFramePos.x-width/2)/(width/2), (height/2-newFramePos.y)/(height/2), 0.f, 1.f);
//gl_Position = vec4((newFramePos.x-120.f)/(120.f), (90.f-newFramePos.y)/(90.f), 0.f, 1.f);
//    gl_Position = vec4(aPos.x, -aPos.y, 1.f, 1.f);
    float polarity = 0.1f;

    vColor = polarity;
}
