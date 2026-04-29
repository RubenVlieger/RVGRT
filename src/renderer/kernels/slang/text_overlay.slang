// Text Overlay
// Renders SDF text and solid rectangles over the composite image.

#include "ShaderTypes.slang"
#include "Settings.slang"

RWTexture2D<float4> texComposite;
Texture2D<float4> texDepth;
Texture2D<float4> texAtlas;
SamplerState atlasSampler;

StructuredBuffer<GlyphInstance> glyphs;
StructuredBuffer<TextOverlayData> overlayData;
StructuredBuffer<uint> tileData;

[shader("compute")]
[numthreads(16, 16, 1)]
void TextOverlay(uint3 gid : SV_DispatchThreadID)
{
    if (gid.x >= overlayData[0].screenWidth || gid.y >= overlayData[0].screenHeight) return;

    uint32_t numGlyphs = overlayData[0].numGlyphs;
    if (numGlyphs == 0) return;

    uint32_t tilesX = overlayData[0].numTilesX;
    uint32_t tilesY = overlayData[0].numTilesY;

    uint32_t tileX = uint32_t(gid.x) / TEXT_TILE_SIZE;
    uint32_t tileY = uint32_t(gid.y) / TEXT_TILE_SIZE;

    if (tileX >= tilesX || tileY >= tilesY) return;

    uint32_t tileIdx = tileY * tilesX + tileX;

    uint32_t tileOffset = tileIdx * (1 + TEXT_MAX_GLYPHS_PER_TILE);
    uint32_t tileCount = tileData[tileOffset];

    if (tileCount == 0) return;

    float4 color = texComposite[gid.xy];

    float depth = texDepth[gid.xy].r;

    for (uint32_t i = 0; i < tileCount; i++) {
        uint32_t glyphIdx = tileData[tileOffset + 1 + i];
        if (glyphIdx >= numGlyphs) continue;

        GlyphInstance g = glyphs[glyphIdx];

        if ((float)gid.x < g.screenPos.x || (float)gid.x >= g.screenPos.x + g.screenSize.x) continue;
        if ((float)gid.y < g.screenPos.y || (float)gid.y >= g.screenPos.y + g.screenSize.y) continue;

        if ((g.flags & 1u) != 0 && depth < g.sceneDepth) continue;

        // Solid rectangle mode: fill AABB with flat color, no SDF sampling
        if ((g.flags & GLYPH_FLAG_SOLID_RECT) != 0) {
            float srcAlpha = g.color.w;
            color = float4(
                color.x * (1.0f - srcAlpha) + g.color.x * srcAlpha,
                color.y * (1.0f - srcAlpha) + g.color.y * srcAlpha,
                color.z * (1.0f - srcAlpha) + g.color.z * srcAlpha,
                1.0f
            );
            continue;
        }

        float2 localPos = (float2(gid.xy) - g.screenPos) / g.screenSize;
        float2 uv = g.atlasUVMin + localPos * (g.atlasUVMax - g.atlasUVMin);

        uv = clamp(uv, g.atlasUVMin, g.atlasUVMax);

        float dist = texAtlas.Sample(atlasSampler, uv).r;

        float softness = max(g.softness, 0.01f);
        float alpha = saturate((dist - 0.5f + softness) / (2.0f * softness));

        if (alpha > 0.001f) {
            float srcAlpha = alpha * g.color.w;
            color = float4(
                color.x * (1.0f - srcAlpha) + g.color.x * srcAlpha,
                color.y * (1.0f - srcAlpha) + g.color.y * srcAlpha,
                color.z * (1.0f - srcAlpha) + g.color.z * srcAlpha,
                1.0f
            );
        }
    }

    texComposite[gid.xy] = color;
}
