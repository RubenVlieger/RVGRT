#pragma once

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* id;
#endif

#include <cstdint>
#include "renderer/ShaderTypes.h"

class MaterialMap {
public:
    MaterialMap();
    ~MaterialMap();

    void GenerateDynamic(); 

    // Getters for the new XBrickMap pipeline
    id GetIndirectionTexture(); // Points to index in SectorBuffer
    id GetSectorBuffer();       // Array of SectorInfo
    id GetOccupancyBuffer();    // Array of uint64_t (Brick masks)
    id GetDataBuffer();         // Array of uint8_t (Material IDs)

private:
    id _device;
    
    // 1. Level 1: 3D Texture (R32Uint). Value = Index into SectorBuffer
    id _indirectionTexture; 

    // 2. Level 2: Buffer of SectorInfo structs
    id _sectorBuffer;

    // 3. Level 3: Buffer of uint64_t masks. 
    id _occupancyBuffer;

    // 4. Level 4: The raw material data.
    id _dataBuffer;

    // Compute Pipelines
    id _psoAnalyze; 
    id _psoFill;   
};