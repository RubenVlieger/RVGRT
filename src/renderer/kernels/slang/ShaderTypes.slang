#pragma once

// Shader types for Slang - self-contained, no C++ header dependencies
// Layout must match C++ ShaderTypesCore.h

struct CameraData {
    float3 position;
    float3 forward;
    float3 right;
    float3 up;

    float4x4 unjitteredViewProjection;
    float4x4 prevUnjitteredViewProjection;

    float2 jitter;
    float2 padding;
};

struct CharacterGPUData {
    int numCharacters;
    int padding[3];
    float4 characterCenters[16];
    float4x4 invBoundingBoxes[16];
    float4x4 invBodyParts[96];
};

struct FrameData {
    float3 sunDirection;
    float time;
    float deltaTime;
    int3 worldOrigin;
};

struct ExposureData {
    float sceneLuminance;
    float padding[3];
};

struct SectorInfo {
    uint32_t baseBrickIndex;
    uint32_t flags;
    uint64_t brickMask;
};

struct BrickWorkItem {
    uint32_t sectorIndex;
    uint32_t localBrickIndex;
    uint64_t occupancyOffset;
    uint64_t dataOffset;
};

struct SectorWorkItem {
    int32_t worldX;
    int32_t worldY;
    int32_t worldZ;
    uint32_t wrappedIdx;
};

struct GlyphInstance {
    float2 screenPos;
    float2 screenSize;
    float2 atlasUVMin;
    float2 atlasUVMax;
    float4 color;
    float softness;
    float sceneDepth;
    uint32_t flags;
    uint32_t _pad;
};

struct TextOverlayData {
    uint32_t numGlyphs;
    uint32_t numTilesX;
    uint32_t numTilesY;
    uint32_t screenWidth;
    uint32_t screenHeight;
};
