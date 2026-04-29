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
#define BRICK_POOL_CAPACITY (6 * 1024 * 1024)

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

// Row 1
#define TEX_COBBLE 16
#define TEX_BEDROCK 17
#define TEX_SAND 18
#define TEX_GRAVEL 19
#define TEX_LOG_SIDE 20
#define TEX_LOG_TOP 21
#define TEX_IRON_BLK 22
#define TEX_GOLD_BLK 23
#define TEX_DIAM_BLK 24
#define TEX_CHEST_TOP 25
#define TEX_CHEST_SIDE 26
#define TEX_CHEST_FRONT 27
#define TEX_MUSHROOM_RED 28
#define TEX_MUSHROOM_BRN 29

// Row 2
#define TEX_GOLD_ORE 32
#define TEX_IRON_ORE 33
#define TEX_COAL_ORE 34
#define TEX_BOOKSHELF 35
#define TEX_MOSSY 36
#define TEX_OBSIDIAN 37
#define TEX_GRID 38
#define TEX_TALLGRASS 39
#define TEX_CRAFT_TOP 43
#define TEX_CRAFT_FRONT 44
#define TEX_CRAFT_SIDE 45

// Row 3
#define TEX_SPONGE 48
#define TEX_GLASS 49
#define TEX_DIAM_ORE 50
#define TEX_REDSTONE_ORE 51
#define TEX_LEAVES 52
#define TEX_LEAVES_OPAQUE 53
#define TEX_STONE_BRICK 54
#define TEX_DEAD_BUSH 55
#define TEX_FERN 56

// Row 4
#define TEX_WOOL_WHITE 64
#define TEX_SPAWNER 65
#define TEX_SNOW 66
#define TEX_ICE 67
#define TEX_GRASS_SNOW 68
#define TEX_CACTUS_TOP 69
#define TEX_CACTUS_SIDE 70
#define TEX_CACTUS_IN 71
#define TEX_CLAY 72
#define TEX_REEDS 73
#define TEX_NOTEBLOCK 74
#define TEX_JUKEBOX 75

// Row 5
#define TEX_TORCH 80
#define TEX_DOOR_W_UP 81
#define TEX_DOOR_I_UP 82
#define TEX_LADDER 83
#define TEX_TRAPDOOR 84
#define TEX_IRON_BARS 85
#define TEX_FARMLAND_WET 86
#define TEX_FARMLAND_DRY 87
#define TEX_WHEAT_0 88
#define TEX_WHEAT_7 95

// Row 6
#define TEX_LEVER 96
#define TEX_DOOR_W_DN 97
#define TEX_DOOR_I_DN 98
#define TEX_REDTORCH_ON 99
#define TEX_PUMPKIN_TOP 102
#define TEX_PUMPKIN_SIDE 103
#define TEX_PUMPKIN_FACE 104
#define TEX_PUMPKIN_OFF 119

// Row 7
#define TEX_RAIL_CORNER 112
#define TEX_WOOL_BLACK 113
#define TEX_WOOL_GRAY 114
#define TEX_RAIL_STR 128

// Row 8
#define TEX_LAPIS_BLK 144

// Row 9
#define TEX_LAPIS_ORE 160

// Row 12
#define TEX_SANDSTONE_TOP 192
#define TEX_SANDSTONE_SID 193
#define TEX_SANDSTONE_BOT 194

// Row 14
#define TEX_NETHERRACK 224
#define TEX_SOULSAND 225
#define TEX_GLOWSTONE 226
#define TEX_PISTON_TOP 227
#define TEX_PISTON_SIDE 228
#define TEX_PISTON_BOT 229
#define TEX_PISTON_IN 230

// ============================================================================
// RAY TRACING SETTINGS
// ============================================================================
#define RT_MAX_DIST 1000.0f
#define RT_MAX_STEPS 256
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
