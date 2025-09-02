#include "util.hpp"
#include "Camera.hpp"
#include "Character.hpp"
#include "State.hpp"
#include <numbers>

// void print(vec3 v)
// {
//     cout << v.x << " " << v.y << " " << v.z << endl; 
// }

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
    position = glm::vec3(128.0f, 466.0f, 128.0f);

    FOV = 60.0f;

    yaw = -0.7f;
    pitch = -std::numbers::pi - 0.3f;
    direction = calcDirfromSphere(pitch, yaw);

    speed = 2.0f;
    speedDropoff = 0.95f;
    jumpSpeed = -30.0f;
    sensitivity = 0.015f;
    gravityAmount = 0.0f;
    lockMouse = false;
}
void Character::Update() 
{
    if (!lockMouse) {
        //vec3 deltaRotation = vec3(Game::window.mouseState.delta.first, 0.0f, Game::window.mouseState.delta.second) * deltaTime * sensitivity;
        //print(deltaRotation);

        yaw += State::state.deltaXMouse.exchange(0) * sensitivity * State::state.deltaTime * FOV;
        pitch += State::state.deltaYMouse.exchange(0) * sensitivity * State::state.deltaTime * FOV;

        yaw = fmod(yaw, std::numbers::pi * 2.0f);
        pitch = clamp(pitch, -4.5, -1.65);  
        direction = calcDirfromSphere(pitch, yaw);
    }

    //print(direction);

    vec3 inputs = vec3((IsKeyDown(0x44) ? 1.0f : 0.0f) + (IsKeyDown(0x41) ? -1.0f : 0.0f), //D - A
                       (IsKeyDown(0x20) ? 1.0f : 0.0f) + (IsKeyDown(0x5A) ? -1.0f : 0.0f), // space - z
                       (IsKeyDown(0x57) ? 1.0f : 0.0f) + (IsKeyDown(0x53) ? -1.0f : 0.0f)) * speed; // W  - S

    velocity += inputs.x * glm::cross((vec3)direction, vec3(0.0f, 1.0f, 0.0f)) + inputs.z * (vec3)direction;
    velocity *= speedDropoff;

    vec3 jump = vec3(0.0f, 1.0f, 0.0f) * -inputs.y * jumpSpeed;
    vec3 gravity = vec3(0.0f, 1.0f, 0.0f) * gravityAmount;

    vec3 addVector = (velocity + jump + gravity) * State::state.deltaTime;
    std::cout << camera.pos.y << std::endl;
    position = glm::mix(position, position + addVector, 0.5f);
    vec3 dirright = normalize(cross((vec3)direction, vec3(0.f, 1.f, 0.f))); //direction.z, 0, direction.x
    vec3 dirup = normalize(cross((vec3)direction, dirright));

    const float FOVFACTOR = (1.0f) / tanf((70.0f * 3.1415f / 180.0f) / 2.0f);

    camera.pos = position;
    camera.forward = vec3(direction);
    camera.right = dirright;
    camera.up = dirup;

    float fovFactor = (float)tan(FOV * std::numbers::pi / 180.0);
    vec2 rcpAspectRatio = vec2((float)State::state.dispWIDTH / (float)State::state.dispHEIGHT, 1.0f);
    vec2 rcpTwoImageSizes = 2.0f / vec2((float)State::state.dispWIDTH, (float)State::state.dispHEIGHT);

    camera.cameraAddFactor = rcpAspectRatio * -fovFactor;
    camera.cameraMultiplyFactor = fovFactor * rcpAspectRatio * rcpTwoImageSizes;

    State::state.deltaXMouse = 0.0f;
    State::state.deltaYMouse = 0.0f;
}


bool Character::IsKeyDown(char keycode)
{
    return State::state.IsKeyDown(keycode);
}