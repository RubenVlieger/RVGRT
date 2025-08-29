#pragma once
#include "util.hpp"


struct hitInfo
{
    bool hit = false;
    glm::vec3 pos = glm::vec3(0, 0, 0);
    glm::vec3 normal = glm::vec3(0, 0, 0);
    glm::vec3 color = glm::vec3(0, 0, 0);
};