#pragma once
#include "util.hpp"
#include "Camera.hpp"
#include <bitset>

struct BodyPart {
    glm::mat4 modelMatrix;
    glm::mat4 inverseModelMatrix;
};

class Character 
{
public:
    // Model Parts for GPU AABB testing
    BodyPart boundingBox;
    BodyPart head;
    BodyPart trunk;
    BodyPart leftArm;
    BodyPart rightArm;
    BodyPart leftLeg;
    BodyPart rightLeg;

    // Animation state
    float walkPhase;
    float walkSwingAmount;

    // View & Projection Matrices
    glm::mat4 viewMatrix;
    glm::mat4 projectionMatrix;
    glm::mat4 viewProjectionMatrix;   
    glm::mat4 unjitteredViewProjectionMatrix;
    glm::mat4 prevUnjitteredViewProjectionMatrix;
    glm::mat4 inverseViewProjectionMatrix;
    glm::mat4 prevViewProjectionMatrix;

    glm::mat4 lastRenderedViewProjectionMatrix; 
    
    // Camera params
    float nearPlane;
    float farPlane;
    float FOV;

    float jitterX;
    float jitterY;

    Camera camera;
    bool lockMouse;
    
    // Physical state
    glm::vec3 position; // Eye/Camera position of the character
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
    
private:
    void UpdateTransformations();
};