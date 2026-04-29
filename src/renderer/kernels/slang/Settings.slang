#pragma once

// Feature toggles and ray limits (mirroring shader_settings.h)

#define VOLUMETRIC_FOG 1
#define INDIRECT_LIGHTING 1
#define REFLECTIONS 1
#define SHADOWS 1

#define CHARACTER_MODELS 1
#define USE_METALFX 0

#define SHADOW_MAXDIST 256.0f
#define SHADOW_STEPS 64

#define WATER_SHADOW_MAXDIST 128.0f
#define WATER_SHADOW_STEPS 32

#define REFLECTION_SHADOW_MAXDIST 64.0f
#define REFLECTION_SHADOW_STEPS 16

#define INDIRECT_SHADOW_MAXDIST 100.0f
#define INDIRECT_SHADOW_STEPS 16

#define VOLUMETRIC_SHADOW_MAXDIST 200.0f
#define VOLUMETRIC_SHADOW_STEPS 32

#define VOLUMETRIC_MAXDIST 300.0f
#define VOLUMETRIC_STEPS 8
#define FOG_DENSITY 0.005f
#define FOG_COLOR float3(0.6f, 0.7f, 0.8f)
#define FOG_ANISOTROPY 0.6f

#define IMAGE_SATURATION 1.4f
#define SKY_IMAGE_SATURATION 1.05f

#define COMPOSITE_FOG_START 60.0f
#define COMPOSITE_FOG_DENSITY 0.0002f
#define COMPOSITE_FOG_COLOR float3(0.5f, 0.7f, 0.9f)

#define INDIRECT_BOUNCE_MAX_ITERS 256

#define DIST_APPROX_MAX_ITERS 256

#define CHARACTER_MAX_TRACE_DIST 500.0f
#define CHARACTER_MAX_TRACE_DIST_SQ (CHARACTER_MAX_TRACE_DIST * CHARACTER_MAX_TRACE_DIST)

// World dimension helpers (mirroring cumath.h / SystemConfig.h)
static const uint64_t SHIX = 12;
static const uint64_t SHIY = 9;
static const uint64_t SHIZ = 12;
static const uint64_t MODX = 4095;
static const uint64_t MODY = 511;
static const uint64_t MODZ = 4095;
static const uint64_t SIZEX = 4096;
static const uint64_t SIZEY = 512;
static const uint64_t SIZEZ = 4096;
static const uint64_t BYTESIZE = 1073741824;

// Texture atlas indices
#define TEX_GRASS_TOP 40
#define TEX_STONE 1
#define TEX_DIRT 2
#define TEX_GRASS_SIDE 3
#define TEX_PLANKS 4
#define TEX_SLAB_SIDE 5
#define TEX_SLAB_TOP 6
#define TEX_BRICK 7
#define TEX_TNT_SIDE 8
#define TEX_TNT_TOP 9
#define TEX_TNT_BOT 10
#define TEX_WEB 11
#define TEX_ROSE 12
#define TEX_FLOWER 13
#define TEX_WATER 14
#define TEX_SAPLING 15
#define TEX_COBBLE 16
#define TEX_BEDROCK 17
#define TEX_SAND 18
#define TEX_GRAVEL 19
#define TEX_LOG_SIDE 20
#define TEX_LOG_TOP 21
#define TEX_IRON_BLK 22
#define TEX_GOLD_BLK 23
#define TEX_DIAM_BLK 24
#define TEX_GOLD_ORE 32
#define TEX_IRON_ORE 33
#define TEX_COAL_ORE 34
#define TEX_BOOKSHELF 35
#define TEX_MOSSY 36
#define TEX_OBSIDIAN 37
#define TEX_TALLGRASS 39
#define TEX_GLASS 49
#define TEX_DIAM_ORE 50
#define TEX_LEAVES 52
#define TEX_LEAVES_OPAQUE 53
#define TEX_STONE_BRICK 54
#define TEX_WOOL_WHITE 64
#define TEX_SNOW 66
#define TEX_ICE 67
#define TEX_SANDSTONE_TOP 192
#define TEX_SANDSTONE_SID 193
#define TEX_SANDSTONE_BOT 194
#define TEX_NETHERRACK 224
#define TEX_SOULSAND 225
#define TEX_GLOWSTONE 226
#define TEX_CACTUS_TOP 69
#define TEX_CACTUS_SIDE 70
#define TEX_CACTUS_IN 71
#define TEX_CLAY 72
#define TEX_PUMPKIN_TOP 102
#define TEX_PUMPKIN_SIDE 103
#define TEX_PUMPKIN_FACE 104

// Material IDs
#define MAT_AIR 0
#define MAT_STONE 1
#define MAT_GRASS 2
#define MAT_DIRT 3
#define MAT_COBBLE 4
#define MAT_PLANKS 5
#define MAT_BEDROCK 7
#define MAT_SAND 12
#define MAT_GRAVEL 13
#define MAT_GOLD_ORE 14
#define MAT_IRON_ORE 15
#define MAT_COAL_ORE 16
#define MAT_LOG 17
#define MAT_LEAVES 18
#define MAT_GLASS 20
#define MAT_SANDSTONE 24
#define MAT_TNT 46
#define MAT_ICE 79
#define MAT_CLAY 82
#define MAT_WOOL 35
#define MAT_GOLD_BLK 41
#define MAT_IRON_BLK 42
#define MAT_BRICK 45
#define MAT_MOSSY 48
#define MAT_OBSIDIAN 49
#define MAT_DIAM_ORE 56
#define MAT_DIAM_BLK 57
#define MAT_SNOW 66
#define MAT_CACTUS 81
#define MAT_PUMPKIN 86
#define MAT_NETHERRACK 87
#define MAT_SOULSAND 88
#define MAT_GLOWSTONE 89

// Indirection grid dimensions
static const int IND_X = 512;
static const int IND_Y = 64;
static const int IND_Z = 512;
static const int IND_OFFSET = 2;

// Sector sentinels
#define SECTOR_HANDLE_EMPTY 0u
#define SECTOR_HANDLE_LOD 0xFFFFFFFEu
#define SECTOR_FLAG_DETAIL 0u
#define SECTOR_FLAG_LOD 1u

// Text rendering settings
#define TEXT_TILE_SIZE 64
#define TEXT_MAX_GLYPHS_PER_TILE 64
#define GLYPH_FLAG_SOLID_RECT 2u
