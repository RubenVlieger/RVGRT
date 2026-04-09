#pragma once

#include "renderer/ShaderTypes.h"
#include "renderer/SystemConfig.h"
#include "renderer/FontAtlas.hpp"
#include <vector>
#include <string>

#ifdef __APPLE__
#import <Metal/Metal.h>
#endif

struct GlyphInstance;

class TextRenderer {
public:
    TextRenderer();
    ~TextRenderer() = default;

    bool Initialize(id<MTLDevice> device, FontAtlas& fontAtlas);

    void BeginFrame(uint32_t screenWidth, uint32_t screenHeight);
    void AddText(const std::string& text, float x, float y, float scale,
                 simd_float4 color, float softness = 0.1f, bool depthTest = false, float sceneDepth = 1e30f);
    void AddRect(float x, float y, float w, float h,
                 simd_float4 color, float sceneDepth = 1e30f);
    void EndFrame();

    void UpdateBuffers(id<MTLDevice> device);

    id<MTLTexture> GetAtlasTexture() const { return _atlasTexture; }
    id<MTLBuffer> GetGlyphBuffer() const { return _glyphBuffer; }
    id<MTLBuffer> GetTileBuffer() const { return _tileBuffer; }
    id<MTLBuffer> GetOverlayDataBuffer() const { return _overlayDataBuffer; }
    uint32_t GetNumGlyphs() const { return static_cast<uint32_t>(_glyphs.size()); }

private:
    struct TileData {
        uint32_t count;
        uint32_t indices[TEXT_MAX_GLYPHS_PER_TILE];
    };

    void BuildTileCoverage();

    FontAtlas* _fontAtlas;
    id<MTLTexture> _atlasTexture;
    id<MTLBuffer> _glyphBuffer;
    id<MTLBuffer> _tileBuffer;
    id<MTLBuffer> _overlayDataBuffer;

    std::vector<GlyphInstance> _glyphs;
    std::vector<TileData> _tiles;
    uint32_t _screenWidth;
    uint32_t _screenHeight;
    uint32_t _tilesX;
    uint32_t _tilesY;
    bool _buffersDirty;
};