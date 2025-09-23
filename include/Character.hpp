#pragma once
#include "util.hpp"
#include "Camera.hpp"
#include "bitset"
class Character 
{
    public:
    glm::mat4 viewMatrix;
    glm::mat4 projectionMatrix;
    glm::mat4 viewProjectionMatrix;   
    glm::mat4 unjitteredViewProjectionMatrix;
    glm::mat4 prevUnjitteredViewProjectionMatrix;
    glm::mat4 inverseViewProjectionMatrix;
    glm::mat4 prevViewProjectionMatrix;   // Previous frame's combined matrix
    float nearPlane;
    float farPlane;
    float FOV;

    float jitterX;
    float jitterY;


    Camera camera;
    bool lockMouse;
    glm::vec3 position;
    glm::vec3 velocity;
    glm::dvec3 direction;

    float yaw;
    float pitch;
    
    float speed;
    float speedDropoff;
    float jumpSpeed;
    float sensitivity;
    float gravityAmount;
    bool IsKeyDown(char key);
    void Update(unsigned int frameCount);
    Character();
};