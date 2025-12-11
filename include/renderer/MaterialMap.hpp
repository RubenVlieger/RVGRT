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

    // These names must match MaterialMap.mm implementation
    id GetIndirectionTexture(); 
    id GetGeoBuffer();            
    id GetMatBuffer(); 

private:
    id _device;
    id _indirectionTexture; 

    id _geoBuffer;
    id _matBuffer;

    id _psoAnalyze; 
    id _psoFill;   
    
    id _psoJFAInit;
    id _psoJFAStep;
    id _psoJFACommit;
};