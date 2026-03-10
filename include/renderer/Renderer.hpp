#pragma once

#include "glm/mat4x4.hpp"
class Character;

// A pure abstract base class for any rendering engine.
class Renderer
{
public:
    virtual ~Renderer() = default;

    // A generic draw call that all renderers must implement.
    virtual void Draw(
        const Character& character,
        unsigned int frameCount
    ) = 0;
    
    // Returns the final platform-specific output texture (e.g. ID3D12Resource* or id<MTLTexture>)
    virtual void* GetOutputTexture() = 0;
    
    // Any other common renderer functions can go here.
};