#include "shader_macros.h"

#if defined(PLATFORM_METAL)
#include <metal_stdlib>
using namespace metal;
#endif

#include "cumath.h"
#include "renderer/ShaderTypes.h"
#include "renderer/SystemConfig.h"

KERNEL(TextOverlay)(
    PARAM_TEXTURE_READWRITE(tex2d_f32_rw, texComposite, 0),
    PARAM_TEXTURE_READ(tex2d_f32_r, texDepth, 1),
    PARAM_TEXTURE_READ(tex2d_f32_s, texAtlas, 2),

    PARAM_BUFFER(GlyphInstance, glyphs, 0),
    PARAM_BUFFER(TextOverlayData, overlayData, 1),
    PARAM_BUFFER(uint32_t, tileData, 2),

    DECLARE_GID()
)
{
#if defined(PLATFORM_METAL)
    if (gid.x >= overlayData->screenWidth || gid.y >= overlayData->screenHeight) return;
#else
    CHECK_BOUNDS(overlayData->screenWidth, overlayData->screenHeight);
    int2 gid = GET_GID();
#endif

    uint32_t numGlyphs = overlayData->numGlyphs;
    if (numGlyphs == 0) return;

    uint32_t tilesX = overlayData->numTilesX;
    uint32_t tilesY = overlayData->numTilesY;

    uint32_t tileX = (uint32_t)gid.x / TEXT_TILE_SIZE;
    uint32_t tileY = (uint32_t)gid.y / TEXT_TILE_SIZE;

    if (tileX >= tilesX || tileY >= tilesY) return;

    uint32_t tileIdx = tileY * tilesX + tileX;

    uint32_t tileOffset = tileIdx * (1 + TEXT_MAX_GLYPHS_PER_TILE);
    uint32_t tileCount = tileData[tileOffset];

    if (tileCount == 0) return;

    float4 color = TEX_READ_2D(texComposite, gid);

    float depth = TEX_READ_2D(texDepth, gid).r;

    DECLARE_SAMPLER(sLinear, linear, clamp_to_edge);

    for (uint32_t i = 0; i < tileCount; i++) {
        uint32_t glyphIdx = tileData[tileOffset + 1 + i];
        if (glyphIdx >= numGlyphs) continue;

        GlyphInstance g = glyphs[glyphIdx];

        if ((float)gid.x < g.screenPos.x || (float)gid.x >= g.screenPos.x + g.screenSize.x) continue;
        if ((float)gid.y < g.screenPos.y || (float)gid.y >= g.screenPos.y + g.screenSize.y) continue;

        if ((g.flags & 1u) != 0 && depth < g.sceneDepth) continue;

        float2 localPos = (AS_FLOAT2(gid) - AS_FLOAT2(g.screenPos)) / AS_FLOAT2(g.screenSize);
        float2 uv = AS_FLOAT2(g.atlasUVMin) + localPos * (AS_FLOAT2(g.atlasUVMax) - AS_FLOAT2(g.atlasUVMin));

        uv = MATH_CLAMP(uv, AS_FLOAT2(g.atlasUVMin), AS_FLOAT2(g.atlasUVMax));

#if defined(PLATFORM_METAL)
        float dist = texAtlas.sample(sLinear, uv).r;
#else
        float dist = TEX_SAMPLE_2D(texAtlas, uv).r;
#endif

        float softness = MATH_MAX(g.softness, 0.01f);
        float alpha = MATH_SATURATE((dist - 0.5f + softness) / (2.0f * softness));

        if (alpha > 0.001f) {
            float srcAlpha = alpha * g.color.w;
            color = make_float4(
                color.x * (1.0f - srcAlpha) + g.color.x * srcAlpha,
                color.y * (1.0f - srcAlpha) + g.color.y * srcAlpha,
                color.z * (1.0f - srcAlpha) + g.color.z * srcAlpha,
                1.0f
            );
        }
    }

    TEX_WRITE_2D(texComposite, color, gid);
}