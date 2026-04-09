#import <Metal/Metal.h>
#include "renderer/TextRenderer.hpp"
#include <cstring>
#include <cmath>

TextRenderer::TextRenderer()
    : _fontAtlas(nullptr)
    , _atlasTexture(nil)
    , _glyphBuffer(nil)
    , _tileBuffer(nil)
    , _overlayDataBuffer(nil)
    , _screenWidth(0)
    , _screenHeight(0)
    , _tilesX(0)
    , _tilesY(0)
    , _buffersDirty(false) {
}

bool TextRenderer::Initialize(id<MTLDevice> device, FontAtlas& fontAtlas) {
    _fontAtlas = &fontAtlas;

    const std::vector<uint8_t>& pixels = fontAtlas.GetAtlasPixels();
    uint32_t atlasWidth = fontAtlas.GetAtlasWidth();
    uint32_t atlasHeight = fontAtlas.GetAtlasHeight();

    MTLTextureDescriptor* texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                        width:atlasWidth
                                                                                       height:atlasHeight
                                                                                    mipmapped:NO];
    texDesc.usage = MTLTextureUsageShaderRead;
    texDesc.storageMode = MTLStorageModeShared;

    _atlasTexture = [device newTextureWithDescriptor:texDesc];
    if (!_atlasTexture) return false;

    MTLRegion region = MTLRegionMake2D(0, 0, atlasWidth, atlasHeight);
    [_atlasTexture replaceRegion:region
                     mipmapLevel:0
                       withBytes:pixels.data()
                     bytesPerRow:atlasWidth];

    NSUInteger maxGlyphBufferSize = sizeof(GlyphInstance) * TEXT_MAX_GLYPHS;
    _glyphBuffer = [device newBufferWithLength:maxGlyphBufferSize
                                        options:MTLResourceStorageModeShared];
    if (!_glyphBuffer) return false;

    NSUInteger maxTileBufferSize = sizeof(TileData) * 1024;
    _tileBuffer = [device newBufferWithLength:maxTileBufferSize
                                      options:MTLResourceStorageModeShared];
    if (!_tileBuffer) return false;

    _overlayDataBuffer = [device newBufferWithLength:sizeof(TextOverlayData)
                                            options:MTLResourceStorageModeShared];
    if (!_overlayDataBuffer) return false;

    return true;
}

void TextRenderer::BeginFrame(uint32_t screenWidth, uint32_t screenHeight) {
    _screenWidth = screenWidth;
    _screenHeight = screenHeight;
    _glyphs.clear();

    _tilesX = (screenWidth + TEXT_TILE_SIZE - 1) / TEXT_TILE_SIZE;
    _tilesY = (screenHeight + TEXT_TILE_SIZE - 1) / TEXT_TILE_SIZE;
}

void TextRenderer::AddText(const std::string& text, float x, float y, float scale,
                            simd_float4 color, float softness, bool depthTest, float sceneDepth) {
    if (!_fontAtlas || !_fontAtlas->IsValid()) return;
    if (_glyphs.size() + text.size() > TEXT_MAX_GLYPHS) return;

    float cursorX = x;

    for (char c : text) {
        if (c == '\n') {
            cursorX = x;
            y += _fontAtlas->GetGlyphMetric('M').sizeY * scale * 1.2f;
            continue;
        }

        const GlyphMetric& metric = _fontAtlas->GetGlyphMetric(c);

        if (c != ' ' && static_cast<unsigned char>(c) >= TEXT_FIRST_CHAR &&
            static_cast<unsigned char>(c) < TEXT_FIRST_CHAR + TEXT_NUM_CHARS) {

            float glyphScreenW = metric.sizeX * scale;
            float glyphScreenH = metric.sizeY * scale;

            if (_glyphs.size() < TEXT_MAX_GLYPHS) {
                GlyphInstance gi;
                gi.screenPos = simd_make_float2(cursorX + metric.bearingX * scale,
                                                  y + metric.bearingY * scale);
                gi.screenSize = simd_make_float2(glyphScreenW, glyphScreenH);
                gi.atlasUVMin = simd_make_float2(metric.uvMinX, metric.uvMinY);
                gi.atlasUVMax = simd_make_float2(metric.uvMaxX, metric.uvMaxY);
                gi.color = color;
                gi.softness = softness;
                gi.sceneDepth = sceneDepth;
                gi.flags = depthTest ? 1u : 0u;
                gi._pad = 0;
                _glyphs.push_back(gi);

                if (_glyphs.size() <= 5) {
                    NSLog(@"AddText: char='%c' cursorX=%.1f screenPos=(%.1f,%.1f) advance=%.1f size=%.1fx%.1f",
                          c, cursorX, gi.screenPos.x, gi.screenPos.y, metric.advance, glyphScreenW, glyphScreenH);
                }
            }
        }

        cursorX += metric.advance * scale;
    }
}

void TextRenderer::AddRect(float x, float y, float w, float h,
                           simd_float4 color, float sceneDepth) {
    if (_glyphs.size() >= TEXT_MAX_GLYPHS) return;

    GlyphInstance gi;
    gi.screenPos = simd_make_float2(x, y);
    gi.screenSize = simd_make_float2(w, h);
    gi.atlasUVMin = simd_make_float2(0.0f, 0.0f);
    gi.atlasUVMax = simd_make_float2(1.0f, 1.0f);
    gi.color = color;
    gi.softness = 0.5f;
    gi.sceneDepth = sceneDepth;
    gi.flags = GLYPH_FLAG_SOLID_RECT;
    gi._pad = 0;
    _glyphs.push_back(gi);
}

void TextRenderer::EndFrame() {
    BuildTileCoverage();
    _buffersDirty = true;
}

void TextRenderer::BuildTileCoverage() {
    uint32_t totalTiles = _tilesX * _tilesY;
    _tiles.clear();
    _tiles.resize(totalTiles);

    for (uint32_t i = 0; i < _glyphs.size(); i++) {
        const GlyphInstance& gi = _glyphs[i];

        float minX = gi.screenPos.x;
        float minY = gi.screenPos.y;
        float maxX = minX + gi.screenSize.x;
        float maxY = minY + gi.screenSize.y;

        uint32_t startTileX = static_cast<uint32_t>(std::max(0.0f, minX)) / TEXT_TILE_SIZE;
        uint32_t startTileY = static_cast<uint32_t>(std::max(0.0f, minY)) / TEXT_TILE_SIZE;
        uint32_t endTileX = static_cast<uint32_t>(std::min((float)_screenWidth - 1.0f, maxX)) / TEXT_TILE_SIZE;
        uint32_t endTileY = static_cast<uint32_t>(std::min((float)_screenHeight - 1.0f, maxY)) / TEXT_TILE_SIZE;

        for (uint32_t ty = startTileY; ty <= endTileY && ty < _tilesY; ty++) {
            for (uint32_t tx = startTileX; tx <= endTileX && tx < _tilesX; tx++) {
                uint32_t tileIdx = ty * _tilesX + tx;
                if (_tiles[tileIdx].count < TEXT_MAX_GLYPHS_PER_TILE) {
                    _tiles[tileIdx].indices[_tiles[tileIdx].count++] = i;
                }
            }
        }
    }

    // Debug: print tile 0
    if (_tiles.size() > 0 && _tiles[0].count > 0) {
        fprintf(stderr, "TileCoverage: tile[0] has %u glyphs:", _tiles[0].count);
        for (uint32_t i = 0; i < _tiles[0].count; i++) {
            uint32_t gIdx = _tiles[0].indices[i];
            fprintf(stderr, " g%u@%.0f", gIdx, _glyphs[gIdx].screenPos.x);
        }
        fprintf(stderr, "\n");
    }
}

void TextRenderer::UpdateBuffers(id<MTLDevice> device) {
    if (!_buffersDirty) return;
    _buffersDirty = false;

    // Always update overlay data (even when no glyphs)
    TextOverlayData overlayData;
    overlayData.numGlyphs = static_cast<uint32_t>(_glyphs.size());
    overlayData.numTilesX = _tilesX;
    overlayData.numTilesY = _tilesY;
    overlayData.screenWidth = _screenWidth;
    overlayData.screenHeight = _screenHeight;
    memcpy([_overlayDataBuffer contents], &overlayData, sizeof(TextOverlayData));

    fprintf(stderr, "UpdateBuffers: numGlyphs=%u tiles=%ux%u screen=%ux%u\n",
            overlayData.numGlyphs, overlayData.numTilesX, overlayData.numTilesY,
            overlayData.screenWidth, overlayData.screenHeight);

    if (_glyphs.size() > 0) {
        NSUInteger requiredSize = sizeof(GlyphInstance) * _glyphs.size();
        if (_glyphBuffer.length < requiredSize) {
            NSUInteger newSize = requiredSize * 2;
            _glyphBuffer = [device newBufferWithLength:newSize
                                               options:MTLResourceStorageModeShared];
        }
        memcpy([_glyphBuffer contents], _glyphs.data(), requiredSize);

        // Verify: read back first glyph to confirm GPU buffer has correct data
        GlyphInstance* gpuData = (GlyphInstance*)[_glyphBuffer contents];
        for (int j = 0; j < 5 && j < (int)_glyphs.size(); j++) {
            fprintf(stderr, "GPU Buffer glyph[%d]: pos=(%.1f,%.1f) size=(%.1f,%.1f) uv=(%.3f,%.3f)-(%.3f,%.3f)\n",
                    j, gpuData[j].screenPos.x, gpuData[j].screenPos.y,
                    gpuData[j].screenSize.x, gpuData[j].screenSize.y,
                    gpuData[j].atlasUVMin.x, gpuData[j].atlasUVMin.y,
                    gpuData[j].atlasUVMax.x, gpuData[j].atlasUVMax.y);
        }
    } else {
        // Clear the glyph buffer to prevent stale data
        if (_glyphBuffer.length > 0) {
            memset([_glyphBuffer contents], 0, _glyphBuffer.length);
        }
    }

    NSUInteger tileDataSize = sizeof(TileData) * _tiles.size();
    if (_tileBuffer.length < tileDataSize) {
        NSUInteger newSize = tileDataSize * 2;
        _tileBuffer = [device newBufferWithLength:newSize
                                          options:MTLResourceStorageModeShared];
    }
    if (_tiles.size() > 0) {
        memcpy([_tileBuffer contents], _tiles.data(), tileDataSize);

        // Verify tile buffer
        uint32_t* tileData = (uint32_t*)[_tileBuffer contents];
        fprintf(stderr, "Tile Buffer verify: tile[0] count=%u indices=[", tileData[0]);
        for (uint32_t i = 0; i < _tiles[0].count && i < 5; i++) {
            fprintf(stderr, "%u ", tileData[1 + i]);
        }
        fprintf(stderr, "]\n");
    }
}