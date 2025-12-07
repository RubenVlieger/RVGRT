#include <metal_stdlib>
#include "cumath.h"
#include "TerrainGeneration.h" // Needed for simplex noise

using namespace metal;


// Bit flag to indicate a value in the Indirection Grid is a Constant Material ID
// and not a pointer to the Brick Pool.
constant uint FLAG_CONSTANT_MAT = 0x80000000;

// Helper to determine material ID based on world position (Replaces logic previously in sampleTexture)
uint8_t get_procedural_material_id(float3 pos) {
    int y = (int)pos.y;
    
    // 1. BEDROCK FLOOR
    // The bottom of the world is unbreakable.
    if (y < 4) {
        // Add some noise to the bedrock layer so it's not perfectly flat
        if (y == 0) return MAT_BEDROCK;
        if (simplex2D(pos.x * 0.1f, pos.z * 0.1f) > 0.0f) return MAT_BEDROCK;
    }

    // 2. BIOME CALCULATION (Large Scale 2D Noise)
    // -1.0 to -0.2 : Desert (Sand)
    // -0.2 to  0.4 : Plains/Forest (Grass)
    //  0.4 to  1.0 : Mountains (Snow/Stone)
    float biomeNoise = simplex2D(pos.x * 0.003f, pos.z * 0.003f); 
    
    // 3. APPROXIMATE TERRAIN HEIGHT
    // We re-calculate a cheap version of the terrain height here to know 
    // if we are "near the surface" or "deep underground".
    // Note: This must roughly match your Evaluate() logic in TerrainGeneration.h
    // but can be lower quality for speed.
    float baseHeight = 140.0f; 
    float mountain = (biomeNoise + 1.0f) * 0.5f * 200.0f; // 0 to 200
    float approxSurfaceY = baseHeight + (simplex2D(pos.x * 0.005f, pos.z * 0.005f) * 20.0f);
    
    if (biomeNoise > 0.4f) approxSurfaceY += mountain; // Mountains are higher

    // Distance from the "top" of the terrain
    int depthFromSurface = (int)(approxSurfaceY - pos.y);

    // 4. SURFACE LAYERS (The "Pretty" Part)
    // If we are within the top 4 blocks of the terrain...
    if (depthFromSurface >= 0 && depthFromSurface < 4) {
        
        // Desert Biome
        if (biomeNoise < -0.2f) {
            if (depthFromSurface == 0) return MAT_SAND;
            return MAT_SANDSTONE;
        }
        // Mountain Biome (Peaks)
        else if (biomeNoise > 0.5f && y > 160) {
            return MAT_BRICK; // Snow on peaks
        }
        // Forest/Plains Biome
        else {
            if (depthFromSurface == 0) return MAT_GRASS;
            return MAT_GRASS;
        }
    }

    // 5. UNDERGROUND (The "Stone" Part)
    // Now we are deeper than 4 blocks. Default is Stone.
    
    // --- CAVES / DECORATION ---
    // High frequency noise to create pockets of different rock types
    float pocketNoise = simplex3D(pos.x * 0.05f, pos.y * 0.05f, pos.z * 0.05f);
    
    // Patches of Dirt/Gravel underground
    if (pocketNoise > 0.7f) return MAT_GRASS;
    if (pocketNoise < -0.7f) return MAT_GRAVEL;

    // --- ORES (The "Reward" Part) ---
    // Ores are rare, so we use high thresholds on high-frequency noise.
    float oreNoise = simplex3D(pos.x * 0.12f, pos.y * 0.12f, pos.z * 0.12f);
    
    // Only generate ores if the rock isn't a "pocket" material
    
    // COAL: Common, found at any height
    if (oreNoise > 0.65f) return MAT_COAL_ORE;

    // IRON: Found below sea level roughly (Y < 120)
    if (y < 120 && oreNoise > 0.72f) return MAT_IRON_ORE;

    // GOLD: Rare, found deep (Y < 40)
    // Use a different noise offset so gold doesn't always spawn inside Iron
    float goldNoise = simplex3D(pos.x * 0.12f + 123.0f, pos.y * 0.12f, pos.z * 0.12f);
    if (y < 40 && goldNoise > 0.78f) return MAT_GOLD_ORE;

    // DIAMOND: Very Rare, very deep (Y < 16)
    // No sand here! Sand is only at Surface Y (>60).
    float diamNoise = simplex3D(pos.x * 0.15f - 50.0f, pos.y * 0.15f, pos.z * 0.15f);
    if (y < 16 && diamNoise > 0.82f) return MAT_DIAM_ORE;
    
    // LAVA / OBSIDIAN POCKETS (Deep)
    if (y < 20) {
         float lavaNoise = simplex3D(pos.x * 0.03f, pos.y * 0.03f, pos.z * 0.03f);
         if (lavaNoise > 0.8f) return MAT_OBSIDIAN; 
    }

    // Default Filler
    return MAT_STONE;
}
// uint8_t get_procedural_material_id(float3 pos) {
//     const float freq = 0.05f;
//     float3 p = floor(pos);
    
//     float eval = simplex3D(p.x * freq, p.y * freq, p.z * freq);
//     float eval2 = simplex3D((p.x + 121.3f) * freq * 0.3f, 
//                             (p.y + 1321.3f) * freq * 0.3f, 
//                             (p.z + 721.5f) * freq * 0.3f);
//     eval = eval * 0.4f + eval2 * 0.6f;

//     if(eval < -1.3f) return MAT_STONE;
//     else if(eval < -1.2f) return MAT_WOOL;
//     else if(eval < -0.7f) return MAT_IRON_ORE;
//     else if(eval < 0.0f) return MAT_GOLD_ORE;
//     else if(eval < 0.1f) return MAT_COAL_ORE;
//     else if(eval < 0.4f) return MAT_BRICK;
//     else if(eval < 0.8f) return MAT_GRASS; 
//     else if(eval < 1.2f) return MAT_STONE; // "Stone2"
    
//     return MAT_STONE;
// }

// =============================================================================
// PASS 1: CLASSIFY
// Scans the packed geometry texture.
// - All 0s -> Write 0 (Air)
// - All 1s -> Write Constant Stone ID
// - Mixed  -> Allocate Index, Write Index
// =============================================================================
kernel void MaterialMap_Classify(
    texture3d<uint, access::read> packedGeometry [[texture(0)]],
    texture3d<uint, access::write> indirectionGrid [[texture(1)]],
    device atomic_uint* brickCounter [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= indirectionGrid.get_width() || 
        gid.y >= indirectionGrid.get_height() || 
        gid.z >= indirectionGrid.get_depth()) return;

    // Each thread handles one 8x8x8 Brick.
    // In the packed texture (R32Uint), 1 unit = 4x4x2 voxels.
    // So an 8x8x8 Brick corresponds to a 2x2x4 block of uints in the packed texture.
    
    uint3 baseTexCoord = uint3(gid.x * 2, gid.y * 2, gid.z * 4);
    
    bool hasAir = false;
    bool hasSolid = false;
    
    // Scan the 16 packed uints that make up this 8x8x8 brick
    for(uint z = 0; z < 4; z++) {
        for(uint y = 0; y < 2; y++) {
            for(uint x = 0; x < 2; x++) {
                uint blockBits = packedGeometry.read(baseTexCoord + uint3(x, y, z)).r;
                
                if (blockBits != 0xFFFFFFFF) hasAir = true;
                if (blockBits != 0x00000000) hasSolid = true;
            }
        }
    }

    uint resultValue = 0;

    if (!hasSolid) {
        // Case 1: Fully Empty
        resultValue = 0; // 0 means Air
    } 
    else if (!hasAir) {
        // Case 2: Fully Solid (Optimization requested)
        // Even if it might have been Iron procedurally, if the whole chunk is solid, 
        // we force it to Stone to save memory and lookup time.
        resultValue = FLAG_CONSTANT_MAT | MAT_STONE;
    } 
    else {
        // Case 3: Mixed (Surface, Caves, Ore boundaries next to air)
        // Allocate a slot in the Brick Pool
        uint index = atomic_fetch_add_explicit(brickCounter, 1, memory_order_relaxed);
        resultValue = index; 
        
        // Safety check for MSB (ensure we don't overflow into the flag bit)
        if (resultValue >= FLAG_CONSTANT_MAT) {
            // Fallback to error/stone if pool overflows (unlikely with 31 bits)
            resultValue = FLAG_CONSTANT_MAT | MAT_STONE; 
        }
    }

    indirectionGrid.write(uint4(resultValue, 0,0,0), gid);
}

// =============================================================================
// PASS 2: FILL BRICK POOL
// Scans Indirection Grid. If mixed, generates specific material bytes.
// =============================================================================
kernel void MaterialMap_Fill(
    texture3d<uint, access::read> indirectionGrid [[texture(0)]],
    device uchar* brickPool [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= indirectionGrid.get_width() || 
        gid.y >= indirectionGrid.get_height() || 
        gid.z >= indirectionGrid.get_depth()) return;

    uint lookup = indirectionGrid.read(gid).r;

    // If Air (0) or Constant (Flag set), we don't store data in the pool.
    if (lookup == 0 || (lookup & FLAG_CONSTANT_MAT)) return;

    // Use 'lookup' as the index into the pool
    uint brickIndex = lookup;
    uint baseOffset = brickIndex * 512; // 8*8*8 bytes

    // Base world coordinate of this brick
    float3 brickWorldPos = float3(gid.x * 8, gid.y * 8, gid.z * 8);

    // Iterate over the 512 voxels in this brick
    for(int z = 0; z < 8; z++) {
        for(int y = 0; y < 8; y++) {
            for(int x = 0; x < 8; x++) {
                
                float3 voxelPos = brickWorldPos + float3(x, y, z);
                
                // Z-Curve (Morton) or Linear? 
                // Linear is simpler for cache line fetching in the ray tracer.
                // Index = z * 64 + y * 8 + x
                uint localIndex = (z << 6) | (y << 3) | x;
                
                uint8_t matID = get_procedural_material_id(voxelPos);
                
                brickPool[baseOffset + localIndex] = matID;
            }
        }
    }
}