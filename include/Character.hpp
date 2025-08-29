#pragma once
#include "util.hpp"
#include "Camera.hpp"
#include "bitset"
class Character 
{
    public:
    
    Camera camera;
    bool lockMouse;
    glm::vec3 position;
    glm::vec3 velocity;
    glm::dvec3 direction;

    double yaw;
    double pitch;
    double FOV;
    
    float speed;
    float speedDropoff;
    float jumpSpeed;
    float sensitivity;
    float gravityAmount;
    bool IsKeyDown(char key);
    void Update();
    Character();
};