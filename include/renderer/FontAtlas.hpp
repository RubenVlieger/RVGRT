#pragma once

#include "renderer/SystemConfig.h"
#include <cstdint>
#include <vector>
#include <string>

struct GlyphMetric {
    float uvMinX;
    float uvMinY;
    float uvMaxX;
    float uvMaxY;
    float sizeX;
    float sizeY;
    float advance;
    float bearingX;
    float bearingY;
};

class FontAtlas {
public:
    FontAtlas();
    ~FontAtlas() = default;

    bool Initialize(const std::string& fontPath, float fontSize);
    bool InitializeWithSystemFont(float fontSize);

    const std::vector<uint8_t>& GetAtlasPixels() const { return _atlasPixels; }
    uint32_t GetAtlasWidth() const { return _atlasWidth; }
    uint32_t GetAtlasHeight() const { return _atlasHeight; }

    const GlyphMetric& GetGlyphMetric(char c) const;
    bool IsValid() const { return _initialized; }

private:
    bool GenerateSDFAtlas(const std::vector<uint8_t>& fontData, float fontSize);
    void ComputeSDF(const uint8_t* bitmap, int bmWidth, int bmHeight,
                    uint8_t* sdfOutput, int sdfWidth, int sdfHeight,
                    int spread);

    std::vector<uint8_t> _atlasPixels;
    uint32_t _atlasWidth;
    uint32_t _atlasHeight;
    GlyphMetric _metrics[TEXT_NUM_CHARS];
    bool _initialized;
};