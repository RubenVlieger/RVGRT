#pragma once
#include "util.hpp"


class Camera 
{
    public:
    glm::vec3 pos;
    glm::vec3 forward;
    glm::vec3 right;
    glm::vec3 up;
    glm::vec2 cameraMultiplyFactor;
    glm::vec2 cameraAddFactor;

    glm::vec3 transform(float x, float y);
    Camera();
};