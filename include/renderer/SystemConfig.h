#pragma once

// ============================================================================
// RVGRT System Configuration
// Single source of truth for all system-wide constants
// ============================================================================

// ============================================================================
// WORLD DIMENSIONS
// ============================================================================
// Shift amounts for world coordinates
#define WORLD_SHIFT_X 12
#define WORLD_SHIFT_Y 9
#define WORLD_SHIFT_Z 12

// World dimensions (must be powers of 2)
#define WORLD_SIZE_X (1u << WORLD_SHIFT_X)  // 4096
#define WORLD_SIZE_Y (1u << WORLD_SHIFT_Y)  // 512
#define WORLD_SIZE_Z (1u << WORLD_SHIFT_Z)  // 4096

// Bit masks for coordinate wrapping
#define WORLD_MASK_X ((1u << WORLD_SHIFT_X) - 1u)
#define WORLD_MASK_Y ((1u << WORLD_SHIFT_Y) - 1u)
#define WORLD_MASK_Z ((1u << WORLD_SHIFT_Z) - 1u)

// Byte size of world occupancy bitmap
#define WORLD_BYTE_SIZE (WORLD_SIZE_X * WORLD_SIZE_Y * WORLD_SIZE_Z / 8u)

// ============================================================================
// BRICK CONFIGURATION
// ============================================================================
#define BRICK_SIZE 8
#define BRICK_SIZE_SHIFT 3
#define BRICK_MASK 7

// Indirection grid dimensions (world / brick_size)
#define IND_X (WORLD_SIZE_X >> BRICK_SIZE_SHIFT)
#define IND_Y (WORLD_SIZE_Y >> BRICK_SIZE_SHIFT)
#define IND_Z (WORLD_SIZE_Z >> BRICK_SIZE_SHIFT)
#define IND_SIZE (IND_X * IND_Y * IND_Z)

// Geometry packing (1 pixel holds 4x4x2 voxels)
#define GEO_PACK_X 4
#define GEO_PACK_Y 4
#define GEO_PACK_Z 2

// ============================================================================
// SECTOR CONFIGURATION
// ============================================================================
#define SECTOR_SIZE 32  // 4 bricks * 8 voxels

// Brick pool capacity (how many 8x8x8 bricks can exist simultaneously)
// Each brick uses 576 bytes (64 occupancy + 512 data)
// At 6M bricks: ~3.3GB total
// WebGPU is limited to 2GB buffers, so we reduce capacity.
#ifdef __EMSCRIPTEN__
#define BRICK_POOL_CAPACITY (2 * 1024 * 1024)  // 2M bricks ~ 1.3 GB
#else
#define BRICK_POOL_CAPACITY (6 * 1024 * 1024)  // 6M bricks ~ 3.3 GB
#endif

// Maximum active sectors (must be >= indirection cells)
// 256 * 16 * 256 = 1,048,576 sectors
#define MAX_ACTIVE_SECTORS (256 * 16 * 256)

// Detail radius (in sectors) for full-detail brick generation
// 125 sectors * 32 voxels = 4000 blocks
#define DETAIL_RADIUS_SECTORS 125

// ============================================================================
// SECTOR HANDLE SENTINELS
// ============================================================================
#define SECTOR_HANDLE_EMPTY 0u
#define SECTOR_HANDLE_LOD 0xFFFFFFFEu

// Sector flags
#define SECTOR_FLAG_DETAIL 0u
#define SECTOR_FLAG_LOD 1u

// ============================================================================
// CHARACTER LIMITS
// ============================================================================
#define MAX_CHARACTERS 16
#define BODY_PARTS_PER_CHARACTER 6

// ============================================================================
// MATERIAL IDs
// Based on Beta 1.7.3 / Release 1.0 IDs
// ============================================================================
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
#define MAT_WOOL 35
#define MAT_GOLD_BLK 41
#define MAT_IRON_BLK 42
#define MAT_BRICK 45
#define MAT_TNT 46
#define MAT_MOSSY 48
#define MAT_OBSIDIAN 49
#define MAT_DIAM_ORE 56
#define MAT_DIAM_BLK 57
#define MAT_SNOW 66
#define MAT_ICE 79
#define MAT_CACTUS 81
#define MAT_CLAY 82
#define MAT_PUMPKIN 86
#define MAT_NETHERRACK 87
#define MAT_SOULSAND 88
#define MAT_GLOWSTONE 89

// ============================================================================
// TEXTURE ATLAS INDICES
// ============================================================================
// Row 0
#define TEX_GRASS_TOP 40
#define TEX_STONE 1
#define TEX_DIRT 2
#define TEX_GRASS_SIDE 3
#define TEX_PLANKS 4
#define TEX_PLANKS_2 5

// ============================================================================
// RENDER SETTINGS
// ============================================================================
#define SHADOW_MAXDIST 256.0f
#define SHADOW_STEPS 64
#define WATER_SHADOW_MAXDIST 128.0f
#define WATER_SHADOW_STEPS 32
#define REFLECTION_SHADOW_MAXDIST 64.0f
#define REFLECTION_SHADOW_STEPS 16

// ============================================================================
// RAY TRACING SETTINGS
// ============================================================================
#define RT_MAX_DIST 1000.0f
#ifdef __EMSCRIPTEN__
#define RT_MAX_STEPS 128
#else
#define RT_MAX_STEPS 256
#endif
#define RT_EPSILON 0.001f

// ============================================================================
// TEXT RENDERING SETTINGS
// ============================================================================
#define TEXT_ATLAS_WIDTH 512
#define TEXT_ATLAS_HEIGHT 256
#define TEXT_FONT_SIZE 32.0f
#define TEXT_SDF_SPREAD 8
#define TEXT_TILE_SIZE 64
#define TEXT_MAX_GLYPHS 2048
#define TEXT_MAX_GLYPHS_PER_TILE 64

// GlyphInstance flags (bit field)
#define GLYPH_FLAG_DEPTH_TEST  1u   // Bit 0: enable depth test for 3D text occlusion
#define GLYPH_FLAG_SOLID_RECT  2u   // Bit 1: solid rectangle, skip SDF sampling
#define TEXT_FIRST_CHAR 32
#define TEXT_NUM_CHARS 96

// ============================================================================
// CONSOLE SETTINGS
// ============================================================================
#define CONSOLE_MAX_LINES 200
#define CONSOLE_VISIBLE_LINES 20
#define CONSOLE_INPUT_MAX_LENGTH 256
#define CONSOLE_HISTORY_SIZE 50
#define CONSOLE_FADE_TIME 5.0f
#define CONSOLE_LINE_HEIGHT 22.0f
#define CONSOLE_FONT_SCALE 0.9f
#define CONSOLE_MARGIN_X 12.0f
#define CONSOLE_MARGIN_BOTTOM 30.0f
#define CONSOLE_BG_ALPHA 0.5f
#define CONSOLE_TEXT_ALPHA 0.9f
#define CONSOLE_TEXT_ALPHA_FADED 0.4f
#define CONSOLE_CURSOR_BLINK_INTERVAL 0.5f

// ============================================================================
// POST-PROCESSING
// ============================================================================
#define EXPOSURE_HISTOGRAM_SIZE 256
#define EXPOSURE_ADAPT_SPEED 0.05f
#define DENOISE_ITERATIONS 3
