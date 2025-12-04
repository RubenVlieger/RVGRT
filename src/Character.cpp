#include "util.hpp"
#include "Camera.hpp"
#include "Character.hpp"
#include "State.hpp" 
#include "platform/Platform.hpp"
#include <numbers>

#include <glm/gtc/matrix_transform.hpp>

float halton(int index, int base) {
    float f = 1.0f;
    float r = 0.0f;
    while (index > 0) {
        f = f / (float)base;
        r = r + f * (float)(index % base);
        index = index / base;
    }
    return r * 0.5f;
}
Character::Character()
    : viewMatrix(1.0f),
      projectionMatrix(1.0f),
      viewProjectionMatrix(1.0f),
      unjitteredViewProjectionMatrix(1.0f),
      prevUnjitteredViewProjectionMatrix(1.0f),
      inverseViewProjectionMatrix(1.0f),
      prevViewProjectionMatrix(1.0f),
      nearPlane(0.1f),
      farPlane(1000.0f),
      FOV(70.0f),
      jitterX(0.0f),
      jitterY(0.0f),
      lockMouse(false),
      position(508.0f, 156.0f, 408.0f), // Start in the middle of the world
      velocity(0.0f),
      direction(0.0f, 0.0f, -1.0f),
      yaw(std::numbers::pi_v<float>),
      pitch(-std::numbers::pi_v<float> * 0.5f),
      speed(0.05f),
      speedDropoff(0.92f),
      jumpSpeed(2.0f),
      sensitivity(0.00007f),
      gravityAmount(0.0f)
      
{
    lastRenderedViewProjectionMatrix = glm::mat4(1.0f);
    // Constructor body can be empty if all initialization is done above
}

glm::dvec3 calcDirfromSphere(double pitch, double yaw) 
{
    const float pih = std::numbers::pi_v<float> * 0.5f;
    glm::vec4 sins = (glm::vec4(sin(yaw), sin(yaw + pih), sin(pitch), sin(pitch+pih)));
    return normalize(glm::vec3(-sins[0] * -sins[3], 
                          -sins[2],
                          -sins[1] *  sins[3]));
}

void Character::Update(unsigned int frameCount) 
{
    // Get the platform pointer from the global state
    Platform* platform = State::state.platform.get();
    if (!platform) return; // Guard against calls before platform is initialized
    prevViewProjectionMatrix = viewProjectionMatrix;
    prevUnjitteredViewProjectionMatrix = unjitteredViewProjectionMatrix;


    vec3 inputs = vec3((platform->IsKeyDown('D') ? 1.0f : 0.0f) + (platform->IsKeyDown('A') ? -1.0f : 0.0f),
                       (platform->IsKeyDown(' ') ? 1.0f : 0.0f) + (platform->IsKeyDown('Z') ? -1.0f : 0.0f),
                       (platform->IsKeyDown('W') ? 1.0f : 0.0f) + (platform->IsKeyDown('S') ? -1.0f : 0.0f)) * speed;

    
    if(platform->IsKeyDown(0x38)){
        inputs *= 0.3f;   
    }     

    if (!lockMouse) 
    {
        // Access members through the platform pointer
        float deltayaw = platform->deltaXMouse.exchange(0) * sensitivity * platform->deltaTime * FOV;
        float deltapitch = platform->deltaYMouse.exchange(0) * sensitivity * platform->deltaTime * FOV;

        if(platform->IsKeyDown(0x38)){
            deltayaw *= 0.4f;
            deltapitch *= 0.4f;
        }     

        yaw += deltayaw ;
        pitch += deltapitch;

        yaw = fmod(yaw, std::numbers::pi * 2.0f);
        pitch = clamp(pitch, -4.5f, -1.65f);  
        direction = calcDirfromSphere(pitch, yaw);
    }


    velocity += inputs.x * glm::cross((vec3)direction, vec3(0.0f, 1.0f, 0.0f)) + inputs.z * (vec3)direction;
    velocity *= speedDropoff;

    vec3 jump = vec3(0.0f, 1.0f, 0.0f) * inputs.y * jumpSpeed;
    vec3 gravity = vec3(0.0f, 1.0f, 0.0f) * gravityAmount;



    vec3 addVector = (velocity + jump + gravity) * platform->deltaTime;



    position = glm::mix(position, position + addVector, 0.5f);
    vec3 dirright = normalize(cross((vec3)direction, vec3(0.f, 1.f, 0.f)));
    vec3 dirup = normalize(cross((vec3)direction, dirright));
    
    viewMatrix = glm::lookAt(
        position,                 
        position + (vec3)direction,     
        glm::vec3(0.0f, 1.0f, 0.0f) 
    );

    projectionMatrix = glm::perspective(
        glm::radians(FOV),
        (float)State::dispWIDTH / (float)State::dispHEIGHT,
        nearPlane,
        farPlane
    );
    unjitteredViewProjectionMatrix = projectionMatrix * viewMatrix;

    int jitterIndex = (frameCount % 16) + 1;
    jitterX = halton(jitterIndex, 2) - 0.5f;
    jitterY = halton(jitterIndex, 3) - 0.5f;

    float clipSpaceJitterX = jitterX / (0.5f * State::dispWIDTH);
    float clipSpaceJitterY = jitterY / (0.5f * State::dispHEIGHT);
    projectionMatrix[2][0] += clipSpaceJitterX;
    projectionMatrix[2][1] += clipSpaceJitterY;

    viewProjectionMatrix = projectionMatrix * viewMatrix;
    inverseViewProjectionMatrix = glm::inverse(unjitteredViewProjectionMatrix);

    camera.pos = position;
    camera.forward = vec3(direction);
    camera.right = dirright;
    camera.up = dirup;

    platform->deltaXMouse.store(0);
    platform->deltaYMouse.store(0);
}


bool Character::IsKeyDown(char keycode)
{
    // Access IsKeyDown via the platform pointer
    if(State::state.platform) {
        return State::state.platform->IsKeyDown(keycode);
    }
    return false;
}