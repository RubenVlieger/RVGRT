#include "renderer/FontAtlas.hpp"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <fstream>

FontAtlas::FontAtlas()
    : _atlasWidth(TEXT_ATLAS_WIDTH)
    , _atlasHeight(TEXT_ATLAS_HEIGHT)
    , _initialized(false) {
    _atlasPixels.resize(_atlasWidth * _atlasHeight, 0);
    memset(_metrics, 0, sizeof(_metrics));
}

bool FontAtlas::Initialize(const std::string& fontPath, float fontSize) {
    std::ifstream file(fontPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return InitializeWithSystemFont(fontSize);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> fontData(fileSize);
    if (!file.read(reinterpret_cast<char*>(fontData.data()), static_cast<std::streamsize>(fileSize))) {
        return InitializeWithSystemFont(fontSize);
    }

    return GenerateSDFAtlas(fontData, fontSize);
}

bool FontAtlas::InitializeWithSystemFont(float fontSize) {
#ifdef __APPLE__
    const char* systemPaths[] = {
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Geneva.ttf",
        "/System/Library/Fonts/NewYork.ttf",
        "/System/Library/Fonts/Keyboard.ttf",
        "/System/Library/Fonts/SFCompact.ttf",
        nullptr
    };
#elif _WIN32
    const char* systemPaths[] = {
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        nullptr
    };
#else
    const char* systemPaths[] = { nullptr };
#endif

    for (int i = 0; systemPaths[i] != nullptr; i++) {
        std::ifstream file(systemPaths[i], std::ios::binary | std::ios::ate);
        if (!file.is_open()) continue;

        size_t fileSize = static_cast<size_t>(file.tellg());
        file.seekg(0, std::ios::beg);
        std::vector<uint8_t> fontData(fileSize);
        if (!file.read(reinterpret_cast<char*>(fontData.data()), static_cast<std::streamsize>(fileSize))) continue;

        if (GenerateSDFAtlas(fontData, fontSize)) {
            return true;
        }
    }

    return false;
}

bool FontAtlas::GenerateSDFAtlas(const std::vector<uint8_t>& fontData, float fontSize) {
    stbtt_fontinfo fontInfo;
    int offset = stbtt_GetFontOffsetForIndex(fontData.data(), 0);
    if (offset < 0) offset = 0;

    if (!stbtt_InitFont(&fontInfo, fontData.data(), offset)) {
        return false;
    }

    float scaleFactor = stbtt_ScaleForPixelHeight(&fontInfo, fontSize);
    int padding = TEXT_SDF_SPREAD;
    unsigned char onedge_value = 128;
    float pixel_dist_scale = 18.0f;

    // Clear atlas
    memset(_atlasPixels.data(), 0, _atlasWidth * _atlasHeight);

    // Pack glyphs into atlas manually
    int cursorX = 0;
    int cursorY = 0;
    int rowHeight = 0;

    for (int i = 0; i < TEXT_NUM_CHARS; i++) {
        int codepoint = TEXT_FIRST_CHAR + i;

        int sdfWidth, sdfHeight, xoff, yoff;
        unsigned char* sdfBitmap = stbtt_GetCodepointSDF(
            &fontInfo, scaleFactor, codepoint, padding,
            onedge_value, pixel_dist_scale,
            &sdfWidth, &sdfHeight, &xoff, &yoff);

        if (!sdfBitmap) {
            // Empty glyph (like space)
            int advance, lsb;
            stbtt_GetCodepointHMetrics(&fontInfo, codepoint, &advance, &lsb);
            _metrics[i].advance = advance * scaleFactor;
            _metrics[i].bearingX = lsb * scaleFactor;
            _metrics[i].bearingY = 0;
            _metrics[i].sizeX = 0;
            _metrics[i].sizeY = 0;
            _metrics[i].uvMinX = 0;
            _metrics[i].uvMinY = 0;
            _metrics[i].uvMaxX = 0;
            _metrics[i].uvMaxY = 0;
            continue;
        }

        // Pack into atlas
        if (cursorX + sdfWidth > (int)_atlasWidth) {
            cursorX = 0;
            cursorY += rowHeight;
            rowHeight = 0;
        }

        if (cursorY + sdfHeight > (int)_atlasHeight) {
            stbtt_FreeSDF(sdfBitmap, nullptr);
            return false;
        }

        // Copy SDF bitmap into atlas
        for (int y = 0; y < sdfHeight; y++) {
            memcpy(&_atlasPixels[(cursorY + y) * _atlasWidth + cursorX],
                   &sdfBitmap[y * sdfWidth], sdfWidth);
        }

        stbtt_FreeSDF(sdfBitmap, nullptr);

        // Store metrics
        int advance, lsb;
        stbtt_GetCodepointHMetrics(&fontInfo, codepoint, &advance, &lsb);

        _metrics[i].uvMinX = (float)cursorX / _atlasWidth;
        _metrics[i].uvMinY = (float)cursorY / _atlasHeight;
        _metrics[i].uvMaxX = (float)(cursorX + sdfWidth) / _atlasWidth;
        _metrics[i].uvMaxY = (float)(cursorY + sdfHeight) / _atlasHeight;
        _metrics[i].sizeX = (float)sdfWidth;
        _metrics[i].sizeY = (float)sdfHeight;
        _metrics[i].advance = advance * scaleFactor;
        _metrics[i].bearingX = (float)xoff;
        _metrics[i].bearingY = (float)yoff;

        if (i < 5) {
            fprintf(stderr, "FontAtlas: char='%c' advance=%.1f size=%dx%d uv=(%.3f,%.3f)-(%.3f,%.3f)\n",
                  codepoint, advance * scaleFactor, sdfWidth, sdfHeight,
                  _metrics[i].uvMinX, _metrics[i].uvMinY,
                  _metrics[i].uvMaxX, _metrics[i].uvMaxY);
        }

        cursorX += sdfWidth + 1;
        rowHeight = std::max(rowHeight, sdfHeight + 1);
    }

    _initialized = true;
    return true;
}

const GlyphMetric& FontAtlas::GetGlyphMetric(char c) const {
    int idx = static_cast<unsigned char>(c) - TEXT_FIRST_CHAR;
    if (idx < 0 || idx >= TEXT_NUM_CHARS) {
        idx = 0;
    }
    return _metrics[idx];
}