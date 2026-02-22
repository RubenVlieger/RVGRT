#pragma once
#include <iostream>

#include "cumath.h"


GPU_FUNC GPU_INLINE float fractf_dev(float x) { return x - floorf(x); }
GPU_FUNC GPU_INLINE float dot3(float ax, float ay, float az, float bx, float by, float bz) {
    return ax*bx + ay*by + az*bz;
}
GPU_FUNC GPU_INLINE float dot2(float ax, float ay, float bx, float by) {
    return ax*bx + ay*by;
}
GPU_FUNC GPU_INLINE float length3(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z);
}
GPU_FUNC GPU_INLINE void normalize3(float &x, float &y, float &z) {
    float L = length3(x,y,z);
    if (L > 0.0f) { x /= L; y /= L; z /= L; }
}


GPU_FUNC GPU_INLINE unsigned int hash3(int xi, int yi, int zi) {
    // fold coordinates into a single 32-bit key (use unsigned arithmetic)
    unsigned int x = (unsigned int)xi;
    unsigned int y = (unsigned int)yi;
    unsigned int z = (unsigned int)zi;

    // cheap spatial hashing using different large primes then XOR
    unsigned int key = x * 73856093u;
    key ^= y * 19349663u;
    key ^= z * 83492791u;

    // Thomas Wang 32-bit integer mix (finalizer)
    key = (key ^ 61u) ^ (key >> 16);
    key *= 9u;
    key = key ^ (key >> 4);
    key *= 0x27d4eb2du;
    key = key ^ (key >> 15);

    return key;
}
GPU_FUNC GPU_INLINE unsigned int hash2(int xi, int yi) {
    // fold coordinates into a single 32-bit key (use unsigned arithmetic)
    unsigned int x = (unsigned int)xi;
    unsigned int y = (unsigned int)yi;

    // cheap spatial hashing using different large primes then XOR
    unsigned int key = x * 73856093u;
    key ^= y * 19349663u;

    // Thomas Wang 32-bit integer mix (finalizer)
    key = (key ^ 61u) ^ (key >> 16);
    key *= 9u;
    key = key ^ (key >> 4);
    key *= 0x27d4eb2du;
    key = key ^ (key >> 15);

    return key;
}

// inline "gradient generator" for 2D.
GPU_FUNC GPU_INLINE float2 grad_from_hash2D(unsigned int hash) {
    hash &= 7u; // Restrict hash to a value between 0 and 7.

    float x = (hash & 1u) ? 1.0f : -1.0f;
    float y = (hash & 2u) ? 1.0f : -1.0f;

    // A single branch to zero out one component for the first four gradients.
    if (hash < 4u) {
        y = 0.0f;
    } else {
        x = 0.0f;
    }
    
    return make_float2(x, y);
}
// A performant simplex2D implementation.
GPU_FUNC GPU_INLINE float simplex2D(float px, float py) {
    // Standard skewing and un-skewing constants for 2D Simplex noise.
    const float F2 = (sqrtf(3.0f) - 1.0f) * 0.5f;
    const float G2 = (3.0f - sqrtf(3.0f)) * 0.5f;

    // Skew the input coordinates to a regular equilateral triangular grid.
    float s = (px + py) * F2;
    int i = floorf(px + s);
    int j = floorf(py + s);

    // Un-skew the coordinates to get the vector from the origin simplex vertex
    // back to the original point.
    float t = (float)(i + j) * G2;
    float x0 = px - (float)i + t;
    float y0 = py - (float)j + t;

    // Determine the second vertex of the simplex.
    // For 2D, the two remaining vertices are found by a simple coordinate comparison.
    int i1, j1;
    if (x0 > y0) {
        i1 = 1;
        j1 = 0;
    } else {
        i1 = 0;
        j1 = 1;
    }

    // Un-skewed vectors to the other two vertices.
    float x1 = x0 - (float)i1 + G2;
    float y1 = y0 - (float)j1 + G2;
    float x2 = x0 - 1.0f + 2.0f * G2;
    float y2 = y0 - 1.0f + 2.0f * G2;

    // Get the gradients at each of the three simplex vertices.
    float2 g0 = grad_from_hash2D(hash2(i, j));
    float2 g1 = grad_from_hash2D(hash2(i + i1, j + j1));
    float2 g2 = grad_from_hash2D(hash2(i + 1, j + 1));

    // Calculate contributions from each vertex using a distance-squared falloff.
    float n0, n1, n2;

    // Contribution from the first vertex (0,0).
    float t0 = 0.5f - x0*x0 - y0*y0;
    t0 = fmaxf(0.0f, t0);
    t0 *= t0;
    n0 = t0 * t0 * (g0.x * x0 + g0.y * y0);

    // Contribution from the second vertex (i1, j1).
    float t1 = 0.5f - x1*x1 - y1*y1;
    t1 = fmaxf(0.0f, t1);
    t1 *= t1;
    n1 = t1 * t1 * (g1.x * x1 + g1.y * y1);

    // Contribution from the third vertex (1,1).
    float t2 = 0.5f - x2*x2 - y2*y2;
    t2 = fmaxf(0.0f, t2);
    t2 *= t2;
    n2 = t2 * t2 * (g2.x * x2 + g2.y * y2);

    // Sum the contributions and scale to a usable range.
    return 70.0f * (n0 + n1 + n2);
}
GPU_FUNC GPU_INLINE float dot3(const float3 &g, float x, float y, float z) {
    return g.x * x + g.y * y + g.z * z;
}

// inline "gradient generator" instead of table lookup, benchmarking proofs this is incredibly faster on my system with a 2.5x speedup.
GPU_FUNC GPU_INLINE float3 grad_from_hash(unsigned int h) 
{
    h &= 15u;

    float3 g;
    g.x = (h & 1u) ? 1.0f : -1.0f;
    g.y = (h & 2u) ? 1.0f : -1.0f;
    g.z = (h & 4u) ? 1.0f : -1.0f;

    if (h < 8u) g.z = 0.0f;
    else if (h < 12u) g.x = 0.0f;
    else g.y = 0.0f;

    return g;
}

// A very optimized simplex3D implementation. Benchmarking along side optimization features 4.0x speedup on my system versus naive simplex3D algorithm.
GPU_FUNC float simplex3D(float px, float py, float pz) 
{
    const float F3 = 1.0f / 3.0f;
    float s = (px + py + pz) * F3;
    int i = int(floorf(px + s));
    int j = int(floorf(py + s));
    int k = int(floorf(pz + s));

    const float G3 = 1.0f / 6.0f;
    float t = float(i + j + k) * G3;
    float x0 = px - (float(i) - t);
    float y0 = py - (float(j) - t);
    float z0 = pz - (float(k) - t);

    int i1, j1, k1;
    int i2, j2, k2;

    int c_xy = (x0 >= y0);
    int c_xz = (x0 >= z0);
    int c_yz = (y0 >= z0);

    i1 = c_xy & c_xz;
    j1 = (1 - c_xy) & c_yz;
    k1 = (1 - c_xz) & (1 - c_yz);

    int x0_is_smallest = (1 - c_xy) & (1 - c_xz);
    int y0_is_smallest = c_xy & (1 - c_yz);
    int z0_is_smallest = c_xz & c_yz;
    i2 = 1 - x0_is_smallest;
    j2 = 1 - y0_is_smallest;
    k2 = 1 - z0_is_smallest;

    float x1 = x0 - float(i1) + G3;
    float y1 = y0 - float(j1) + G3;
    float z1 = z0 - float(k1) + G3;

    float x2 = x0 - float(i2) + 2.0f * G3;
    float y2 = y0 - float(j2) + 2.0f * G3;
    float z2 = z0 - float(k2) + 2.0f * G3;

    float x3 = x0 - 1.0f + 3.0f * G3;
    float y3 = y0 - 1.0f + 3.0f * G3;
    float z3 = z0 - 1.0f + 3.0f * G3;

    int i_1 = i + i1, j_1 = j + j1, k_1 = k + k1;
    int i_2 = i + i2, j_2 = j + j2, k_2 = k + k2;
    int i_3 = i + 1,  j_3 = j + 1,  k_3 = k + 1;

    float3 g0 = grad_from_hash(hash3(i,   j,   k));
    float3 g1 = grad_from_hash(hash3(i_1, j_1, k_1));
    float3 g2 = grad_from_hash(hash3(i_2, j_2, k_2));
    float3 g3 = grad_from_hash(hash3(i_3, j_3, k_3));

    float n0, n1, n2, n3;

    float t0 = 0.5f - x0*x0 - y0*y0 - z0*z0;
    t0 = fmaxf(0.0f, t0);
    t0 *= t0;
    n0 = t0 * t0 * dot3(g0, x0, y0, z0);

    float t1 = 0.5f - x1*x1 - y1*y1 - z1*z1;
    t1 = fmaxf(0.0f, t1);
    t1 *= t1;
    n1 = t1 * t1 * dot3(g1, x1, y1, z1);

    float t2 = 0.5f - x2*x2 - y2*y2 - z2*z2;
    t2 = fmaxf(0.0f, t2);
    t2 *= t2;
    n2 = t2 * t2 * dot3(g2, x2, y2, z2);

    float t3 = 0.5f - x3*x3 - y3*y3 - z3*z3;
    t3 = fmaxf(0.0f, t3);
    t3 *= t3;
    n3 = t3 * t3 * dot3(g3, x3, y3, z3);

    return 96.0f * (n0 + n1 + n2 + n3);
}


// Calculates 3D Fractional Brownian Motion (fBm) by summing multiple layers (octaves) of Simplex noise.
// This is the core of creating natural-looking, detailed procedural shapes.
GPU_FUNC GPU_INLINE float fbm3D(float x, float y, float z, int octaves, float frequency, float lacunarity, float persistence) {
    float total = 0.0f;
    float amplitude = 1.0f;
    for (int i = 0; i < octaves; i++) {
        total += simplex3D(x * frequency, y * frequency, z * frequency) * amplitude;
        frequency *= lacunarity;
        amplitude *= persistence;
    }
    return total;
}

// 2D version of fBm, used for the biome map.
GPU_FUNC GPU_INLINE float fbm2D(float x, float z, int octaves, float frequency, float lacunarity, float persistence) {
    float total = 0.0f;
    float amplitude = 1.0f;
    for (int i = 0; i < octaves; i++) {
        total += simplex2D(x * frequency, z * frequency) * amplitude;
        frequency *= lacunarity;
        amplitude *= persistence;
    }
    return total;
}


GPU_FUNC GPU_INLINE float Evaluate( float x, float y, float z) {

    const float GROUND_LEVEL = 140.0f;              // Base height of the terrain surface before noise is added. (World height is 512).
    const float PLAINS_AMPLITUDE = 25.0f;           // Max height variation in 'plains' biomes.
    const float MOUNTAIN_AMPLITUDE = 360.0f;        // Max height variation in 'mountain' biomes.
    
    // ** Biome Generation **
    const float BIOME_FREQUENCY = 0.01f;          // Controls the size of biomes. Smaller value = larger biomes.

    // ** Surface Noise (fBm settings for mountains/hills) **
    const int   SURFACE_OCTAVES = 7;                // More octaves = more surface detail.
    const float SURFACE_FREQUENCY = 0.002f;         // Initial scale of terrain features. Smaller = larger hills/mountains.
    const float SURFACE_LACUNARITY = 2.1f;          // How much detail is added each octave (frequency multiplier).
    const float SURFACE_PERSISTENCE = 0.45f;        // How much each successive octave contributes to the shape.

    // ** Cave Generation (fBm settings for caves) **
    const int   CAVE_OCTAVES = 3;                   // Fewer octaves for smoother caves.
    const float CAVE_FREQUENCY = 0.009f;            // Overall scale of the cave systems.
    const float CAVE_CARVE_VALUE = 2.0f;            // A large value to subtract from density to ensure caves are empty.

    // ** Cave Style **
    // For "Spaghetti" tunnels: Creates tunnels when 3D noise is very close to a specific value.
    const float SPAGHETTI_THRESHOLD = 0.025f;       // How thick the spaghetti tunnels are. Smaller = thinner.
    
    // For "Caverns": Uses a second, larger noise to create big open areas.
    const float CAVERN_REGION_FREQ = 0.006f;        // Frequency of the noise that defines where large caverns can form.
    const float CAVERN_THRESHOLD = 0.3f;            // If cave noise is below this value (inside a cavern region), it gets carved.


    // --- 1. Biome Calculation --- 🏞️
    // Use 2D noise to create a "mountainous-ness" map.
    // The result is remapped from [-1, 1] to a clean [0, 1] factor.
    float biome_factor = (simplex2D(x * BIOME_FREQUENCY, z * BIOME_FREQUENCY) + 1.0f) * 0.5f;

    // Smoothly transition between plains and mountains based on the biome factor.
    float terrain_amplitude = PLAINS_AMPLITUDE + biome_factor * (MOUNTAIN_AMPLITUDE - PLAINS_AMPLITUDE);
    
    // --- 2. Surface Density Calculation --- ⛰️
    // Start with a base density gradient: positive below ground level, negative above.
    float density = GROUND_LEVEL - y;

    // Add 3D noise to create the terrain shape.
    float surface_noise = fbm3D(x, y, z, SURFACE_OCTAVES, SURFACE_FREQUENCY, SURFACE_LACUNARITY, SURFACE_PERSISTENCE);
    density += surface_noise * terrain_amplitude;

    // --- 3. Cave Carving --- 🦇
    // We only try to carve caves if the point is already determined to be solid ground.
    if (density > 0.0f) {
        // Calculate a separate 3D noise value specifically for the caves.
        // We offset the coordinates slightly (e.g., adding 123.456) to get a different noise pattern from the surface.
        float cave_noise_raw = fbm3D(x + 123.456f, y, z, CAVE_OCTAVES, CAVE_FREQUENCY, SURFACE_LACUNARITY, SURFACE_PERSISTENCE);
        float cave_noise_normalized = (cave_noise_raw + 1.0f) * 0.5f; // Remap to [0, 1] for easy thresholding.

        // ** A) Spaghetti Tunnels **
        // Check if the noise value is very close to the middle of its range (0.0 in raw, 0.5 in normalized).
        bool is_spaghetti = abs(cave_noise_raw) < SPAGHETTI_THRESHOLD;

        // ** B) Large Caverns **
        // Use another, very low-frequency noise to define large regions where caverns can appear.
        float cavern_region_noise = (simplex3D(x * CAVERN_REGION_FREQ, y * CAVERN_REGION_FREQ, z * CAVERN_REGION_FREQ) + 1.0f) * 0.5f;
        
        // If we are in a "cavern region" AND the main cave noise is below a certain threshold, we carve.
        // This creates large, irregular open spaces.
        bool is_cavern = (cavern_region_noise > 0.65f) && (cave_noise_normalized < CAVERN_THRESHOLD);

        // If either condition is met, carve out the rock!
        if (is_spaghetti || is_cavern) {
            density -= CAVE_CARVE_VALUE;
        }
    }

    return density;
}
