#include "util.hpp"
#include "Camera.hpp"

Camera::Camera()
{
    pos = vec3(0.0f);
    forward = vec3(0.0f);
    right = vec3(0.0f);
    up = vec3(0.0f);
    cameraMultiplyFactor = vec3(0.0f);
    cameraAddFactor = vec3(0.0f);
}