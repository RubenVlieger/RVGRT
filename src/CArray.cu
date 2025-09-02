#include "CArray.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <math_constants.h> // for CUDA constants like CUDART_PI_F

#include "cumath.cuh"


// helpers
__device__ inline float fractf_dev(float x) { return x - floorf(x); }
__device__ inline float dot3(float ax, float ay, float az, float bx, float by, float bz) {
    return ax*bx + ay*by + az*bz;
}
__device__ inline float length3(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z);
}
__device__ inline void normalize3(float &x, float &y, float &z) {
    float L = length3(x,y,z);
    if (L > 0.0f) { x /= L; y /= L; z /= L; }
}

// device noise3D matching your CPU version (returns -1..1)
__device__ float noise3D(float x, float y, float z) {
    float d = sinf(x*12.9898f + y*78.233f + z*128.852f) * 43758.5453f;
    return fractf_dev(d) * 2.0f - 1.0f;
}

// device simplex3D port (returns same scale as CPU)
__device__ float simplex3D(float px, float py, float pz) {
    const float f3 = 1.0f / 3.0f;
    float s = (px + py + pz) * f3;
    int i = int(floorf(px + s));
    int j = int(floorf(py + s));
    int k = int(floorf(pz + s));

    const float g3 = 1.0f / 6.0f;
    float t = float(i + j + k) * g3;
    float x0 = float(i) - t; x0 = px - x0;
    float y0 = float(j) - t; y0 = py - y0;
    float z0 = float(k) - t; z0 = pz - z0;

    int i1, j1, k1;
    int i2, j2, k2;

    if (x0 >= y0) {
        if (y0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
        else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
        else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
    } else {
        if (y0 < z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
        else if (x0 < z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
        else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
    }

    float x1 = x0 - float(i1) + g3;
    float y1 = y0 - float(j1) + g3;
    float z1 = z0 - float(k1) + g3;
    float x2 = x0 - float(i2) + 2.0f*g3;
    float y2 = y0 - float(j2) + 2.0f*g3;
    float z2 = z0 - float(k2) + 2.0f*g3;
    float x3 = x0 - 1.0f + 3.0f*g3;
    float y3 = y0 - 1.0f + 3.0f*g3;
    float z3 = z0 - 1.0f + 3.0f*g3;

    // integer lattice points
    int i3 = i+0, j3 = j+0, k3 = k+0;
    int i_1 = i + i1, j_1 = j + j1, k_1 = k + k1;
    int i_2 = i + i2, j_2 = j + j2, k_2 = k + k2;
    int i_3 = i + 1, j_3 = j + 1, k_3 = k + 1;

    // gradient vectors using device noise3D_dev to build pseudo-random gradients
    float gx0 = noise3D(float(i3), float(j3), float(k3));
    float gy0 = noise3D(float(i3)*2.01f, float(j3)*2.01f, float(k3)*2.01f);
    float gz0 = noise3D(float(i3)*2.02f, float(j3)*2.02f, float(k3)*2.02f);
    normalize3(gx0, gy0, gz0);

    float gx1 = noise3D(float(i_1), float(j_1), float(k_1));
    float gy1 = noise3D(float(i_1)*2.01f, float(j_1)*2.01f, float(k_1)*2.01f);
    float gz1 = noise3D(float(i_1)*2.02f, float(j_1)*2.02f, float(k_1)*2.02f);
    normalize3(gx1, gy1, gz1);

    float gx2 = noise3D(float(i_2), float(j_2), float(k_2));
    float gy2 = noise3D(float(i_2)*2.01f, float(j_2)*2.01f, float(k_2)*2.01f);
    float gz2 = noise3D(float(i_2)*2.02f, float(j_2)*2.02f, float(k_2)*2.02f);
    normalize3(gx2, gy2, gz2);

    float gx3 = noise3D(float(i_3), float(j_3), float(k_3));
    float gy3 = noise3D(float(i_3)*2.01f, float(j_3)*2.01f, float(k_3)*2.01f);
    float gz3 = noise3D(float(i_3)*2.02f, float(j_3)*2.02f, float(k_3)*2.02f);
    normalize3(gx3, gy3, gz3);

    float n0 = 0.0f, n1 = 0.0f, n2 = 0.0f, n3 = 0.0f;

    float t0 = 0.5f - x0*x0 - y0*y0 - z0*z0;
    if (t0 >= 0.0f) {
        t0 *= t0;
        n0 = t0 * t0 * dot3(gx0, gy0, gz0, x0, y0, z0);
    }
    float t1 = 0.5f - x1*x1 - y1*y1 - z1*z1;
    if (t1 >= 0.0f) {
        t1 *= t1;
        n1 = t1 * t1 * dot3(gx1, gy1, gz1, x1, y1, z1);
    }
    float t2 = 0.5f - x2*x2 - y2*y2 - z2*z2;
    if (t2 >= 0.0f) {
        t2 *= t2;
        n2 = t2 * t2 * dot3(gx2, gy2, gz2, x2, y2, z2);
    }
    float t3 = 0.5f - x3*x3 - y3*y3 - z3*z3;
    if (t3 >= 0.0f) {
        t3 *= t3;
        n3 = t3 * t3 * dot3(gx3, gy3, gz3, x3, y3, z3);
    }

    return 96.0f * (n0 + n1 + n2 + n3);
}

// // Evaluate: fractal 3D noise + vertical bias + ground boost + optional hard/soft floor.
// // Returns density: higher -> more likely solid (use e.g. density > 0.0f -> block).
// __device__ float Evaluate(
//     float x, float _y, float z,
//     // noise/fractal params
//     float baseFrequency = 0.003f,
//     float baseAmplitude = 1.0f,
//     int octaves = 5,
//     float lacunarity = 2.0f,
//     float persistence = 0.5f,
//     // vertical falloff (make top sparser)
//     float verticalBiasScale = 1.5f,
//     float verticalBiasPower = 1.7f,
//     // ground control (soft blended floor)
//     float groundLayerStrength = 0.65f,   // low-freq chunky detail near floor
//     float groundNoiseFreq = 0.01f,       // frequency for ground boost
//     // floor params (soft/hard)
//     float floorHeight = 12.0f,           // y <= floorHeight -> strongest floor
//     float floorThickness = 6.0f,         // blend distance above floorHeight
//     float floorStrength = 6.0f,          // strength of floor (raise density)
//     bool useHardFloor = false            // if true, enforce hard solid floor
// ) {
//     const float Y_MAX = 512.0f;

//     // Hard floor quick path (guaranteed solid under threshold).
//     if (useHardFloor && _y <= floorHeight) {
//         return 999.0f; // sufficiently large density so density>0 always true
//     }

//     // --- fractal noise ---
//     float nx = x * baseFrequency;
//     float ny = _y * baseFrequency;
//     float nz = z * baseFrequency;

//     float sum = 0.0f;
//     float amp = baseAmplitude;
//     float freq = 1.0f;
//     for (int i = 0; i < octaves; ++i) {
//         sum += simplex3D(nx * freq, ny * freq, nz * freq) * amp;
//         freq *= lacunarity;
//         amp *= persistence;
//     }

//     // --- vertical falloff (makes ceiling sparser) ---
//     float yn = clampf(_y / Y_MAX, 0.0f, 1.0f);
//     float s = smoothstepf(0.0f, 1.0f, yn);
//     float verticalFalloff = powf(s, verticalBiasPower) * verticalBiasScale;

//     // --- ground boost: low-frequency chunk noise that fades with height ---
//     float groundNoise = simplex3D(x * groundNoiseFreq, 0.0f, z * groundNoiseFreq) * 0.6f;
//     float groundBoost = groundNoise * groundLayerStrength * (1.0f - s);

//     // --- soft floor contribution (blends to a strong positive density near y=0) ---
//     // floorBlend = 0.0 at y <= floorHeight, 1.0 at y >= floorHeight+floorThickness
//     float floorBlend = smoothstepf(floorHeight, floorHeight + floorThickness, _y);
//     // floorContribution goes from floorStrength (at base) -> 0 (above thickness)
//     float floorContribution = (1.0f - floorBlend) * floorStrength;

//     // --- final density ---
//     float density = sum - verticalFalloff + groundBoost + floorContribution;

//     return density;
// }






// Basic FBM: returns approx in [-1, 1]
__device__ float fbm3(float x, float y, float z, int octaves, float lacunarity, float persistence, float baseFreq) {
    float sum = 0.0f;
    float amp = 1.0f;
    float ampSum = 0.0f;
    float freq = baseFreq;
    for (int i = 0; i < octaves; ++i) {
        sum += simplex3D(x * freq, y * freq, z * freq) * amp;
        ampSum += amp;
        freq *= lacunarity;
        amp *= persistence;
    }
    if (ampSum == 0.0f) return 0.0f;
    return sum / ampSum; // approx in [-1,1]
}

// Ridged multifractal-ish 2D on (x,z) using simplex3D(x,0,z).
// Returns in [0,1] (higher = sharper ridge)
__device__ float ridged2D(float x, float z, int octaves, float lacunarity, float persistence, float baseFreq) {
    float sum = 0.0f;
    float amp = 1.0f;
    float ampSum = 0.0f;
    float freq = baseFreq;
    for (int i = 0; i < octaves; ++i) {
        float n = simplex3D(x * freq, 0.0f, z * freq); // [-1,1]
        float signal = 1.0f - fabsf(n);                // [0,1]
        signal = signal * signal;                      // sharpen ridge
        sum += signal * amp;
        ampSum += amp;
        freq *= lacunarity;
        amp *= persistence;
    }
    if (ampSum == 0.0f) return 0.0f;
    return clampf(sum / ampSum, 0.0f, 1.0f);
}

// Domain warp helper: returns warped (x,z) pair via cheap FBM
__device__ void domainWarp2D(float x, float z, float warpFreq, int warpOctaves, float warpAmp, float lacunarity, float persistence, float &outX, float &outZ) {
    float wx = fbm3(x, 0.0f, z, warpOctaves, lacunarity, persistence, warpFreq);
    float wz = fbm3(x + 37.0f, 0.0f, z - 17.0f, warpOctaves, lacunarity, persistence, warpFreq);
    outX = x + wx * warpAmp;
    outZ = z + wz * warpAmp;
}

// Main Evaluate: returns density (higher -> solid). Use `density > 0.0f` for block.
__device__ float Evaluate(
    float x, float _y, float z,

    // global scale (map coord -> noise)
    float baseFrequency = 0.003f,    // base coordinate scale for most FBM

    // mountain (ridged) parameters (2D)
    int mountainOctaves = 6,
    float mountainLacunarity = 2.0f,
    float mountainPersistence = 0.5f,
    float mountainFreq = 0.0008f,    // lower -> bigger mountains
    float mountainHeight = 160.0f,   // maximum mountain height above base
    float mountainBase = 32.0f,      // base height offset (sea level / ground baseline)

    // domain warp for interesting mountain shapes
    float warpFreq = 0.0012f,
    int warpOctaves = 3,
    float warpAmp = 200.0f,

    // vertical terracing (Minecraft-like steps)
    float terraceHeight = 1.0f,      // quantize mountain height into steps (1.0 = blocky)
    float terraceBlend = 0.25f,      // how soft the terracing edges are (0..1)

    // 3D detail / overhangs (adds/subtracts from mountain height)
    int detailOctaves = 4,
    float detailLacunarity = 2.0f,
    float detailPersistence = 0.5f,
    float detailFreq = 0.008f,
    float detailAmp = 18.0f,         // amplitude of 3D detail (positive -> overhangs)

    // caves (subtractive): carve away where caveNoise > caveThreshold
    int caveOctaves = 3,
    float caveFreq = 0.04f,
    float caveThreshold = 0.45f,     // higher => fewer caves
    float caveSmooth = 0.12f,        // width of transition
    float caveCarveStrength = 1.1f,  // how strongly caves carve density

    // floor / bedrock
    float floorHeight = 8.0f,
    float floorThickness = 6.0f,
    float floorStrength = 6.0f,
    bool useHardFloor = true,

    // top falloff (less blocks up top)
    float topBiasScale = 1.6f,
    float topBiasPower = 1.8f,

    // small ground-layer low frequency to make plains look nicer
    float groundLayerStrength = 0.35f,
    float groundNoiseFreq = 0.01f,

    // expected Y range for normalization (set to your map ceiling)
    float Y_MAX = 512.0f
) {
    // Hard floor
    if (useHardFloor && _y <= floorHeight) return 999.0f;
    //if(_y <=340) return 999.0f;

    // --- Domain warp mountains on (x,z) ---
    float wx, wz;
    domainWarp2D(x, z, warpFreq, warpOctaves, warpAmp, mountainLacunarity, mountainPersistence, wx, wz);

    // compute ridged mountain factor [0..1]
    float ridged = ridged2D(wx, wz, mountainOctaves, mountainLacunarity, mountainPersistence, mountainFreq);

    // mountain height (quantize for terraces)
    float rawMountainH = mountainBase + ridged * mountainHeight; // base + ridge * scaled
    // terracing: quantize to terraceHeight but blend edges by terraceBlend
    if (terraceHeight > 0.0f) {
        float q = floorf(rawMountainH / terraceHeight) * terraceHeight;
        // soft blend between q and rawMountainH
        float t = clampf((rawMountainH - q) / (terraceBlend * terraceHeight + 1e-6f), 0.0f, 1.0f);
        rawMountainH = q * (1.0f - t) + rawMountainH * t;
    }

    // --- 3D detail to create overhangs + local bumps (can push some voxels above H to solid) ---
    float detail = fbm3(x * detailFreq, _y * detailFreq, z * detailFreq, detailOctaves, detailLacunarity, detailPersistence, 1.0f);
    // detail in [-1,1], scale by detailAmp
    float mountainH_withDetail = rawMountainH + detail * detailAmp;

    // --- low-frequency ground boost for lower plains ---
    float yn = clampf(_y / Y_MAX, 0.0f, 1.0f);
    float s = smoothstepf(0.0f, 1.0f, yn);
    float groundNoise = fbm3(x * groundNoiseFreq, 0.0f, z * groundNoiseFreq, 4, 2.0f, 0.5f, 1.0f);
    float groundBoost = groundNoise * groundLayerStrength * (1.0f - s);

    // --- density from mountain surface: positive when solid
    // if voxel y is below the (mountain height + detail), it becomes solid.
    // We compute density = mountainH_withDetail - y + small smoothing noise
    float density = mountainH_withDetail - _y;
    density += groundBoost * 8.0f; // strengthen ground boost contribution

    // --- caves: subtract density where cave field is strong -->
    // caveNoise in [-1,1]. Map to [0,1], then smoothstep around threshold to create hollow regions.
    float caveNoise = fbm3(x * caveFreq, _y * caveFreq, z * caveFreq, caveOctaves, 2.0f, 0.5f, 1.0f); // [-1,1]
    float caveN01 = caveNoise * 0.5f + 0.5f;
    float caveMask = smoothstepf(caveThreshold - caveSmooth, caveThreshold + caveSmooth, caveN01); // 0..1 where 1 = cave
    density -= caveMask * caveCarveStrength * (10.0f); // carve; multiply to make caves deep

    // --- top bias: make ceiling sparse (optional) ---
    float topFalloff = powf(s, topBiasPower) * topBiasScale;
    density -= topFalloff;

    // --- floor contribution (soft) ---
    float floorBlend = smoothstepf(floorHeight, floorHeight + floorThickness, _y);
    float floorContribution = (1.0f - floorBlend) * floorStrength;
    density += floorContribution;

    
    return density;
}



extern "C" __global__
void fillKernelWords(uint32_t* __restrict__ data, uint64_t numWords, uint64_t totalBits)
{
    // 64-bit idx to support very large worlds
    uint64_t wordIdx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (wordIdx >= numWords) return;

    // compute base bit index for this 32-bit word
    uint64_t baseBit = wordIdx * 32ull;

    // build the 32-bit word locally (no atomics)
    uint32_t w = 0u;

    for (uint64_t bit = 0; bit < 32; ++bit) {
        uint64_t bitIndex = baseBit + (uint64_t)bit;
        if (bitIndex >= totalBits) break; // in case totalBits isn't divisible by 32

        // convert bitIndex -> (x,y,z) using your shifts/masks or SIZEX,SIZEY,SIZEZ
        // Example if SIZEX, SIZEY, SIZEZ are powers of two and you use shifts:
        // Assuming you have SHIX, SHIY, SHIZ and MODX/MODY/MODZ defined appropriately
        uint64_t idx = (uint64_t)bitIndex; // safe if totalBits < 2^32; else compute using 64bit math below

        // If your world might exceed 2^32 bits, compute x,y,z using 64-bit ops:
        // uint64_t z = bitIndex >> (SHIX + SHIY);
        // uint64_t y = (bitIndex >> SHIX) & MODY;
        // uint64_t x = bitIndex & MODX;

        // I'll show the 64-bit variant:
        uint64_t z = bitIndex >> (SHIX + SHIY);
        uint64_t y = (bitIndex >> SHIX) & (uint64_t)MODY;
        uint64_t x = bitIndex & (uint64_t)MODX;

        // call Evaluate (adapted to accept float/double as you have)
      
        float v = Evaluate((float)x, (float)y, (float)z);
        bool solid = v > 0.7f;
        if (solid) {
            w |= (1u << bit);
        }
    }

    // store with a single write
    data[wordIdx] = w;
}

// // Kernel to fill the 3D bits array based on Evaluate
// __global__ void fillKernel(uint32_t* data) 
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if(idx >= (1 << (SHIX + SHIY + SHIX))) return;

//     int z = idx >> (SHIX + SHIY); // idx / (256*256*SHIX >> SHIY;          // idx / (256*256)
//     int y = (idx >> SHIX) & MODY;   // (idx / 256) % 256
//     int x = idx & MODX;          // idx % 256

//     float value = Evaluate((float)x, (float)y, (float)z);
//     bool solid = value > 0.7f;

//     if(solid) {
//         int uintIdx = idx >> 5;      // divide by 32
//         int bitIdx  = idx & 31;      // modulo 32
//         atomicOr(&data[uintIdx], 1u << bitIdx);
//     }
// }

uint32_t* CArray::getPtr() {
    return dev_data;
}

uint64_t CArray::getSize() {
    return SIZE;
}

void CArray::fill() 
{
    if(SIZE == -1)  {
        std::cout << "ERROR CARRAY NOT ALLOCATED" << std::endl;
        exit(1);
    }
    CUDA_CHECK(cudaMemset(dev_data, 0, SIZE));
    CUDA_CHECK(cudaGetLastError());

    uint64_t totalBits = (uint64_t)SIZEX * (uint64_t)SIZEY * (uint64_t)SIZEZ;
    uint64_t numWords  = (totalBits + 31ull) / 32ull;

    uint64_t threads = 256;
    uint64_t blocks64 = (numWords + threads - 1) / threads;

    std::cout << "blocks64: " << blocks64 << std::endl;

    // If blocks64 fits in a single grid dimension:
    fillKernelWords<<<blocks64, threads>>>(dev_data, numWords, totalBits);
    CUDA_CHECK(cudaGetLastError());


    // int threads = 256;
    // int blocks = (SIZE*8 + threads - 1) / threads;
    // fillKernel<<<blocks, threads>>>(dev_data);
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}


void CArray::readback(uint32_t* buffer) 
{
    if(!buffer) {
        std::cout << "NO BUFFER IS INITIALIZED" << std::endl;
        exit(1);
    }
    cudaMemcpy(buffer, dev_data, SIZE, cudaMemcpyDeviceToHost);
}

void CArray::Allocate(uint64_t _size)
{
    SIZE = _size;
    if(!dev_data)
    {
        std::cout << "CREATING ARRAY" << std::endl;
        CUDA_CHECK(cudaMalloc(&dev_data, SIZE));
    }
}

void CArray::Free()
{
    if(dev_data)
    {
        cudaFree(dev_data);
        dev_data = nullptr;
    }
}
CArray::CArray() 
{

}

CArray::~CArray() 
{
  
}