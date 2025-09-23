#include "util.hpp"
#include "Camera.hpp"
#include "Character.hpp"
#include "State.hpp"
#include <numbers>

#include <glm/gtc/matrix_transform.hpp>

static const float g_JitterSequence[8][2] =
{
    { -1.0f/8.0f, -1.0f/8.0f }, {  1.0f/8.0f,  3.0f/8.0f },
    {  5.0f/8.0f, -3.0f/8.0f }, { -3.0f/8.0f,  5.0f/8.0f },
    { -7.0f/8.0f, -5.0f/8.0f }, {  3.0f/8.0f,  7.0f/8.0f },
    {  7.0f/8.0f, -7.0f/8.0f }, { -5.0f/8.0f,  1.0f/8.0f }
};


glm::dvec3 calcDirfromSphere(double pitch, double yaw) 
{
    const float pih = std::numbers::pi_v<float> * 0.5f;
    glm::vec4 sins = sin(glm::vec4(yaw, yaw + pih, pitch, pitch+pih));
    return normalize(glm::vec3(-sins[0] * -sins[3], 
                          -sins[2],
                          -sins[1] *  sins[3]));
}

Character::Character() 
{
    velocity = glm::vec3(0.0f, 0.0f, 0.0f);
    position = glm::vec3(128.0f, 350.0f, 128.0f);

    viewMatrix = glm::mat4(1.0f);
    projectionMatrix = glm::mat4(1.0f);
    viewProjectionMatrix = glm::mat4(1.0f);
    unjitteredViewProjectionMatrix = glm::mat4(1.0f);
    prevViewProjectionMatrix = glm::mat4(1.0f);
    prevUnjitteredViewProjectionMatrix = glm::mat4(1.0f);

    nearPlane = 0.1f;
    farPlane = 50000.0f;    


    FOV = 60.0f;

    yaw = -0.7f;
    pitch = -std::numbers::pi - 0.3f;
    direction = calcDirfromSphere(pitch, yaw);

    speed = 30.0f;
    speedDropoff = 0.95f;
    jumpSpeed = -30.0f;
    sensitivity = 0.015f;
    gravityAmount = 0.0f;
    lockMouse = false;
}
void Character::Update(unsigned int frameCount) 
{
    prevViewProjectionMatrix = viewProjectionMatrix;
    prevUnjitteredViewProjectionMatrix = unjitteredViewProjectionMatrix;
    if (!lockMouse) 
    {
        yaw += State::state.deltaXMouse.exchange(0) * sensitivity * State::state.deltaTime * FOV;
        pitch += State::state.deltaYMouse.exchange(0) * sensitivity * State::state.deltaTime * FOV;

        yaw = fmod(yaw, std::numbers::pi * 2.0f);
        pitch = clamp(pitch, -4.5f, -1.65f);  
        direction = calcDirfromSphere(pitch, yaw);
    }
    vec3 inputs = vec3((IsKeyDown(0x44) ? 1.0f : 0.0f) + (IsKeyDown(0x41) ? -1.0f : 0.0f), //D - A
                       (IsKeyDown(0x20) ? 1.0f : 0.0f) + (IsKeyDown(0x5A) ? -1.0f : 0.0f), // space - z
                       (IsKeyDown(0x57) ? 1.0f : 0.0f) + (IsKeyDown(0x53) ? -1.0f : 0.0f)) * speed; // W  - S

    velocity += inputs.x * glm::cross((vec3)direction, vec3(0.0f, 1.0f, 0.0f)) + inputs.z * (vec3)direction;
    velocity *= speedDropoff;

    vec3 jump = vec3(0.0f, 1.0f, 0.0f) * -inputs.y * jumpSpeed;
    vec3 gravity = vec3(0.0f, 1.0f, 0.0f) * gravityAmount;

    vec3 addVector = (velocity + jump + gravity) * State::state.deltaTime;

    position = glm::mix(position, position + addVector, 0.5f);
    vec3 dirright = normalize(cross((vec3)direction, vec3(0.f, 1.f, 0.f))); //direction.z, 0, direction.x
    vec3 dirup = normalize(cross((vec3)direction, dirright));

    const float FOVFACTOR = (1.0f) / tanf((70.0f * 3.1415f / 180.0f) / 2.0f);

    viewMatrix = glm::lookAt(
        position,                 // Camera position
        position + (vec3)direction,     // Target position
        glm::vec3(0.0f, 1.0f, 0.0f) // Up vector
    );

    projectionMatrix = glm::perspective(
        glm::radians(FOV), // FOV in radians
        (float)State::state.dispWIDTH / (float)State::state.dispHEIGHT,
        nearPlane,
        farPlane
    );
    unjitteredViewProjectionMatrix = projectionMatrix * viewMatrix;

    jitterX = g_JitterSequence[frameCount % 8][0] * 0.5;
    jitterY =  g_JitterSequence[frameCount % 8][1] * 0.5;
    float clipSpaceJitterX = jitterX / (0.5f * State::state.dispWIDTH);
    float clipSpaceJitterY = jitterY / (0.5f * State::state.dispHEIGHT);
    projectionMatrix[2][0] += clipSpaceJitterX;
    projectionMatrix[2][1] += clipSpaceJitterY;

    viewProjectionMatrix = projectionMatrix * viewMatrix;

    inverseViewProjectionMatrix = glm::inverse(unjitteredViewProjectionMatrix);

    camera.pos = position;
    camera.forward = vec3(direction);
    camera.right = dirright;
    camera.up = dirup;

    float fovFactor = (float)tan(FOV * std::numbers::pi / 180.0);
    vec2 rcpAspectRatio = vec2((float)State::state.dispWIDTH / (float)State::state.dispHEIGHT, 1.0f);
    vec2 rcpTwoImageSizes = 2.0f / vec2((float)State::state.dispWIDTH, (float)State::state.dispHEIGHT);

    camera.cameraAddFactor = rcpAspectRatio * -fovFactor;
    camera.cameraMultiplyFactor = fovFactor * rcpAspectRatio * rcpTwoImageSizes;

    State::state.deltaXMouse.store(0);
    State::state.deltaYMouse.store(0);
}


bool Character::IsKeyDown(char keycode)
{
    return State::state.IsKeyDown(keycode);
}