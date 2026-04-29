#pragma once

// Raytracing helpers for Slang shaders
// Accesses global resources directly

#include "Math.slang"

static const half3 TINT_GRASS   = half3(0.48h, 0.65h, 0.36h);
static const half3 TINT_FOLIAGE = half3(0.28h, 0.70h, 0.17h);
static const half3 TINT_NONE    = half3(1.0h, 1.0h, 1.0h);

[ForceInline]
inline half3 sampleTexture(
    Texture2DArray<float> textureAtlas,
    SamplerState atlasSampler,
    half2 uv,
    uint8_t matID,
    half3 normal,
    float distSq)
{
    int face = (normal.y > 0.5h) ? 0 : ((normal.y < -0.5h) ? 1 : 2);
    int texIndex = TEX_STONE;
    half3 tint = TINT_NONE;

    switch(matID) {
        case MAT_GRASS:
            if (face == 0) {       texIndex = TEX_GRASS_TOP; tint = TINT_GRASS; }
            else if (face == 1) {  texIndex = TEX_DIRT; }
            else {                 texIndex = TEX_GRASS_SIDE; }
            break;
        case MAT_DIRT:      texIndex = TEX_DIRT; break;
        case MAT_STONE:     texIndex = TEX_STONE; break;
        case MAT_COBBLE:    texIndex = TEX_COBBLE; break;
        case MAT_BEDROCK:   texIndex = TEX_BEDROCK; break;
        case MAT_SAND:      texIndex = TEX_SAND; break;
        case MAT_GRAVEL:    texIndex = TEX_GRAVEL; break;
        case MAT_CLAY:      texIndex = TEX_CLAY; break;
        case MAT_SOULSAND:  texIndex = TEX_SOULSAND; break;
        case MAT_NETHERRACK:texIndex = TEX_NETHERRACK; break;
        case MAT_GLOWSTONE: texIndex = TEX_GLOWSTONE; break;
        case MAT_PLANKS:    texIndex = TEX_PLANKS; break;
        case MAT_BRICK:     texIndex = TEX_BRICK; break;
        case MAT_MOSSY:     texIndex = TEX_MOSSY; break;
        case MAT_OBSIDIAN:  texIndex = TEX_OBSIDIAN; break;
        case MAT_LOG:
            if (face == 0 || face == 1) texIndex = TEX_LOG_TOP;
            else texIndex = TEX_LOG_SIDE;
            break;
        case MAT_LEAVES:
            texIndex = TEX_LEAVES;
            tint = TINT_FOLIAGE;
            break;
        case MAT_GLASS:     texIndex = TEX_GLASS; break;
        case MAT_WOOL:      texIndex = TEX_WOOL_WHITE; break;
        case MAT_SNOW:      texIndex = TEX_SNOW; break;
        case MAT_ICE:       texIndex = TEX_ICE; break;
        case MAT_COAL_ORE:  texIndex = TEX_COAL_ORE; break;
        case MAT_IRON_ORE:  texIndex = TEX_IRON_ORE; break;
        case MAT_GOLD_ORE:  texIndex = TEX_GOLD_ORE; break;
        case MAT_DIAM_ORE:  texIndex = TEX_DIAM_ORE; break;
        case MAT_IRON_BLK:  texIndex = TEX_IRON_BLK; break;
        case MAT_GOLD_BLK:  texIndex = TEX_GOLD_BLK; break;
        case MAT_DIAM_BLK:  texIndex = TEX_DIAM_BLK; break;
        case MAT_TNT:
            if (face == 0) texIndex = TEX_TNT_TOP;
            else if (face == 1) texIndex = TEX_TNT_BOT;
            else texIndex = TEX_TNT_SIDE;
            break;
        case MAT_SANDSTONE:
            if (face == 0) texIndex = TEX_SANDSTONE_TOP;
            else if (face == 1) texIndex = TEX_SANDSTONE_BOT;
            else texIndex = TEX_SANDSTONE_SID;
            break;
        case MAT_PUMPKIN:
            if (face == 0 || face == 1) texIndex = TEX_PUMPKIN_TOP;
            else if (face == 2) texIndex = TEX_PUMPKIN_FACE;
            else texIndex = TEX_PUMPKIN_SIDE;
            break;
        case MAT_CACTUS:
            if (face == 0) texIndex = TEX_CACTUS_TOP;
            else if (face == 1) texIndex = TEX_CACTUS_IN;
            else texIndex = TEX_CACTUS_SIDE;
            break;
        default:
            return half3(1.0h, 0.0h, 1.0h);
    }
    float lod = 0.5f * log2(max(distSq, 0.0001f)) - 6.0f;
    half4 t = (half4)textureAtlas.SampleLevel(atlasSampler, float3(uv.x, -uv.y, texIndex), lod);
    return t.rgb * tint;
}

[ForceInline]
inline uint hash3_to_1(int3 p) {
    uint3 u = uint3(p);
    u = ((u >> 8U) ^ u.yzx) * 0x45D9F3BU;
    u = ((u >> 8U) ^ u.yzx) * 0x45D9F3BU;
    u = ((u >> 8U) ^ u.yzx) * 0x45D9F3BU;
    return u.x ^ u.y ^ u.z;
}

[ForceInline]
inline uint pcg_hash(uint seed)
{
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

[ForceInline]
inline float rand_float(inout uint seed) {
    seed = pcg_hash(seed);
    return (float)seed / float(0xFFFFFFFFu);
}

[ForceInline]
inline float3 reconstructPos(float depth, float2 uv, CameraData cam) {
    float2 ndc = uv * 2.0f - 1.0f;
    float3 viewDir = normalize(cam.forward + ndc.x * cam.right + ndc.y * cam.up);
    return cam.position + viewDir * depth;
}

[ForceInline]
inline float2 reconstructUVFloat(float3 pos, float3 normal) {
    float3 fpos = floor(pos);
    float2 uv;
    if (abs(normal.x) > 0.5f)      uv = float2(pos.y - fpos.y, pos.z - fpos.z);
    else if (abs(normal.y) > 0.5f) uv = float2(pos.x - fpos.x, pos.z - fpos.z);
    else                           uv = float2(pos.x - fpos.x, pos.y - fpos.y);
    return uv;
}

[ForceInline]
inline half2 reconstructUV(float3 pos, half3 normal) {
    float3 fpos = floor(pos);
    half2 uv;
    if (abs(normal.x) > 0.5h)      uv = half2(half(pos.y - fpos.y), half(pos.z - fpos.z));
    else if (abs(normal.y) > 0.5h) uv = half2(half(pos.x - fpos.x), half(pos.z - fpos.z));
    else                           uv = half2(half(pos.x - fpos.x), half(pos.y - fpos.y));
    return uv;
}

[ForceInline]
inline int3 clamp3i(const int3 a, const int3 b, const int3 c)
{
    int3 d = int3(a.x < b.x ? b.x : a.x,
                   a.y < b.y ? b.y : a.y,
                   a.z < b.z ? b.z : a.z);
    return int3(d.x > c.x ? c.x : d.x,
                d.y > c.y ? c.y : d.y,
                d.z > c.z ? c.z : d.z);
}

[ForceInline]
inline half3 clamp3h(const half3 a, const half3 b, const half3 c)
{
    half3 d = half3(a.x < b.x ? b.x : a.x,
                    a.y < b.y ? b.y : a.y,
                    a.z < b.z ? b.z : a.z);
    return half3(d.x > c.x ? c.x : d.x,
                 d.y > c.y ? c.y : d.y,
                 d.z > c.z ? c.z : d.z);
}

[ForceInline]
inline bool checkBitLocal_Optimized(
    uint packed,
    float3 localEntry,
    float3 rayDir,
    float3 rayInvDir,
    int3 stepSign,
    float t_entry_global,
    float3 camPos,
    inout hitInfo HI)
{
    int3 mapPos = int3(
        clamp((int)floor(localEntry.x), 0, 3),
        clamp((int)floor(localEntry.y), 0, 3),
        clamp((int)floor(localEntry.z), 0, 1)
    );

    int currentBitIndex = (mapPos.z << 4) | (mapPos.y << 2) | mapPos.x;

    const int3 stepStride = int3(stepSign.x, stepSign.y * 4, stepSign.z * 16);

    float3 deltaDist = abs(rayInvDir);

    float3 originOffset = float3(mapPos) - localEntry;

    float3 sideDist;
    sideDist.x = (stepSign.x > 0 ? (originOffset.x + 1.0f) : -originOffset.x) * deltaDist.x;
    sideDist.y = (stepSign.y > 0 ? (originOffset.y + 1.0f) : -originOffset.y) * deltaDist.y;
    sideDist.z = (stepSign.z > 0 ? (originOffset.z + 1.0f) : -originOffset.z) * deltaDist.z;

    float tLocalHit = 0.0f;
    int lastAxis = -1;
    [unroll]
    for (int i = 0; i < 10; i++) {

        if ((packed & (1u << currentBitIndex)) != 0) {
            HI.hit = true;

            half3 nx = half3((half)(-stepSign.x), 0, 0);
            half3 ny = half3(0, (half)(-stepSign.y), 0);
            half3 nz = half3(0, 0, (half)(-stepSign.z));
            half3 nDefault = half3(0, 1, 0);
            HI.normal = (lastAxis == 0) ? nx : ((lastAxis == 1) ? ny : ((lastAxis == 2) ? nz : nDefault));

            HI.pos = camPos + rayDir * (t_entry_global + tLocalHit);
            HI.its += i;

            float3 fpos = floor(HI.pos);
            if (abs(HI.normal.x) > 0.5h)      HI.uv = half2(half(HI.pos.y - fpos.y), half(HI.pos.z - fpos.z));
            else if (abs(HI.normal.y) > 0.5h) HI.uv = half2(half(HI.pos.x - fpos.x), half(HI.pos.z - fpos.z));
            else                              HI.uv = half2(half(HI.pos.x - fpos.x), half(HI.pos.y - fpos.y));

            return true;
        }

        bool xMin = (sideDist.x <= sideDist.y) && (sideDist.x <= sideDist.z);
        bool yMin = (sideDist.y <= sideDist.z) && !xMin;

        tLocalHit = xMin ? sideDist.x : (yMin ? sideDist.y : sideDist.z);

        if (xMin) sideDist.x += deltaDist.x;
        else if (yMin) sideDist.y += deltaDist.y;
        else sideDist.z += deltaDist.z;

        if (xMin) {
            mapPos.x += stepSign.x;
            currentBitIndex += stepStride.x;
            lastAxis = 0;
        } else if (yMin) {
            mapPos.y += stepSign.y;
            currentBitIndex += stepStride.y;
            lastAxis = 1;
        } else {
            mapPos.z += stepSign.z;
            currentBitIndex += stepStride.z;
            lastAxis = 2;
        }
        if (((mapPos.x & ~3) | (mapPos.y & ~3) | (mapPos.z & ~1)) != 0) break;
    }

    return false;
}

[ForceInline]
inline half3 sampleSky(const float3 dir, const float3 sunDir)
{
    float sunDot = dot(dir, sunDir);

    if (sunDot > 0.999h) {
        return half3(4.5h, 3.6h, 3.0h) * 2.0h;
    }

    float y = clamp(dir.y, 0.0f, 1.0f);

    half3 zenith = half3(0.1h, 0.4h, 0.8h);
    half3 horizon = half3(0.4h, 0.6h, 0.9h);

    half3 skyColor = lerp(horizon, zenith, half(pow(y, 0.7f)));

    return skyColor;
}
