#import "renderer/MaterialMap.hpp"
#import "State.hpp"
#import "cumath.h"
#import "renderer/MetalDevice.hpp"
#include <cmath>
#include <vector>

namespace {
id<MTLDevice> get_device() {
  return static_cast<MetalDevice *>(State::state.graphicsDevice.get())
      ->GetMetalDevice();
}

id<MTLTexture> create3DTex(id<MTLDevice> dev, int w, int h, int d,
                           MTLPixelFormat fmt, NSString *label) {
  MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
  desc.textureType = MTLTextureType3D;
  desc.pixelFormat = fmt;
  desc.width = w;
  desc.height = h;
  desc.depth = d;
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
               MTLTextureUsageShaderAtomic;
  desc.storageMode = MTLStorageModePrivate;
  id<MTLTexture> tex = [dev newTextureWithDescriptor:desc];
  tex.label = label;
  return tex;
}
} // namespace

void MaterialMap::GenerateDynamic() {
  id<MTLDevice> dev = (id<MTLDevice>)_device;
  id<MTLCommandQueue> queue = [dev newCommandQueue];

  NSLog(@"[MaterialMap] Starting GPU-Accelerated Generation...");

  // --- 1. SETUP DIMENSIONS ---
  int sectorsX = (SIZEX + 31) / 32;
  int sectorsY = (SIZEY + 31) / 32;
  int sectorsZ = (SIZEZ + 31) / 32;
  int totalSectors = sectorsX * sectorsY * sectorsZ;

  // Buffer to store coarse analysis (1 bitmask per sector)
  id<MTLBuffer> sectorAnalysisBuffer =
      [dev newBufferWithLength:totalSectors * sizeof(uint64_t)
                       options:MTLResourceStorageModeShared];

  // --- 2. PASS 1: ANALYZE (Determine Empty Sectors) ---
  {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:_psoAnalyze];
    [enc setBuffer:sectorAnalysisBuffer offset:0 atIndex:0];

    MTLSize gridSize = MTLSizeMake(sectorsX, sectorsY, sectorsZ);
    MTLSize threadGroupSize = MTLSizeMake(8, 8, 4);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }

  // --- 3. PASS 2: COMPACTION (CPU) ---
  // This is fast enough on CPU for now (iterating 200k integers is instant).

  uint64_t *analysisResults = (uint64_t *)[sectorAnalysisBuffer contents];

  std::vector<uint32_t> indirectionGrid(totalSectors, 0);
  std::vector<SectorInfo> sectors;
  sectors.reserve(totalSectors / 8);
  // Push dummy null sector (index 0)
  sectors.push_back({});

  // The work list for the GPU
  std::vector<BrickWorkItem> workList;
  workList.reserve(totalSectors * 4); // Heuristic

  // Global counters for the buffers we will allocate
  uint32_t totalActiveBricks = 0;

  for (int i = 0; i < totalSectors; i++) {
    uint64_t brickMask = analysisResults[i];

    if (brickMask != 0) {
      // 1. Create Sector Entry
      SectorInfo sInfo;
      sInfo.baseBrickIndex = totalActiveBricks;
      sInfo.brickMask = brickMask;

      sectors.push_back(sInfo);
      uint32_t sectorHandle = (uint32_t)sectors.size() - 1;
      indirectionGrid[i] = sectorHandle;

      // 2. Generate Work Items for active bricks
      // We iterate bits to find which bricks are active
      for (int b = 0; b < 64; b++) {
        if ((brickMask >> b) & 1) {
          BrickWorkItem item;
          item.sectorIndex = (uint32_t)i;
          item.localBrickIndex = b;

          // The offsets in the large arrays
          item.occupancyOffset =
              (uint64_t)totalActiveBricks * 8; // 8 sub-masks per brick
          item.dataOffset =
              (uint64_t)totalActiveBricks * 512; // 512 voxels per brick

          workList.push_back(item);
          totalActiveBricks++;
        }
      }
    }
  }

  NSLog(@"[MaterialMap] Compaction: %lu Active Sectors, %u Active Bricks",
        sectors.size(), totalActiveBricks);

  // --- SECTOR MASKS (128³ skipping) ---
  // Group sectors into 4x4x4 tiles. Each bit in a uint64_t indicates
  // whether the corresponding sector in the group has any bricks.
  int superX = (sectorsX + 3) / 4;
  int superY = (sectorsY + 3) / 4;
  int superZ = (sectorsZ + 3) / 4;
  int totalSuper = superX * superY * superZ;
  std::vector<uint64_t> sectorMasks(totalSuper, 0);

  for (int sy = 0; sy < sectorsY; sy++) {
    for (int sz = 0; sz < sectorsZ; sz++) {
      for (int sx = 0; sx < sectorsX; sx++) {
        int sectorIdx = sx + sz * sectorsX + sy * sectorsX * sectorsZ;
        if (indirectionGrid[sectorIdx] != 0) {
          // This sector is non-empty
          int superIdx =
              (sx / 4) + (sz / 4) * superX + (sy / 4) * superX * superZ;
          // Local position within the 4x4x4 super-sector (XZY order)
          int lx = sx & 3, ly = sy & 3, lz = sz & 3;
          int bitIdx = lx + (lz << 2) + (ly << 4);
          sectorMasks[superIdx] |= (1ULL << bitIdx);
        }
      }
    }
  }

  NSLog(@"[MaterialMap] Sector masks: %d super-sectors", totalSuper);

  if (totalActiveBricks == 0)
    return; // World is empty

  // --- 4. ALLOCATE GPU MEMORY ---

  // Level 1
  _indirectionTexture = create3DTex(dev, sectorsX, sectorsY, sectorsZ,
                                    MTLPixelFormatR32Uint, @"Indirection");

  // Use a staging buffer to upload to the Private texture
  NSUInteger bytesPerRow = sectorsX * sizeof(uint32_t);
  NSUInteger bytesPerImage = bytesPerRow * sectorsY;
  NSUInteger totalBytes = bytesPerImage * sectorsZ;

  id<MTLBuffer> stagingBuffer =
      [dev newBufferWithBytes:indirectionGrid.data()
                       length:totalBytes
                      options:MTLResourceStorageModeShared];

  id<MTLCommandBuffer> blitCmd = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blitEnc = [blitCmd blitCommandEncoder];

  [blitEnc copyFromBuffer:stagingBuffer
             sourceOffset:0
        sourceBytesPerRow:bytesPerRow
      sourceBytesPerImage:bytesPerImage
               sourceSize:MTLSizeMake(sectorsX, sectorsY, sectorsZ)
                toTexture:_indirectionTexture
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];

  [blitEnc endEncoding];
  [blitCmd commit];
  [blitCmd waitUntilCompleted];
  // Level 2
  NSUInteger sectorBufferSize = sectors.size() * sizeof(SectorInfo);

  // 1. Create the empty Private destination buffer
  _sectorBuffer = [dev newBufferWithLength:sectorBufferSize
                                   options:MTLResourceStorageModePrivate];

  // 2. Create a Shared staging buffer with your CPU data
  id<MTLBuffer> stagingSectorBuffer =
      [dev newBufferWithBytes:sectors.data()
                       length:sectorBufferSize
                      options:MTLResourceStorageModeShared];

  // 3. Blit the data from the staging buffer to the private buffer
  blitCmd = [queue commandBuffer];
  blitEnc = [blitCmd blitCommandEncoder];

  [blitEnc copyFromBuffer:stagingSectorBuffer
             sourceOffset:0
                 toBuffer:_sectorBuffer
        destinationOffset:0
                     size:sectorBufferSize];

  [blitEnc endEncoding];
  [blitCmd commit];
  [blitCmd waitUntilCompleted];
  // Level 3 & 4 (Huge buffers)
  _occupancyBuffer = [dev
      newBufferWithLength:(uint64_t)totalActiveBricks * 8 * sizeof(uint64_t)
                  options:MTLResourceStorageModePrivate];
  _dataBuffer = [dev
      newBufferWithLength:(uint64_t)totalActiveBricks * 512 * sizeof(uint8_t)
                  options:MTLResourceStorageModePrivate];

  // Work List Buffer
  id<MTLBuffer> workListBuffer =
      [dev newBufferWithBytes:workList.data()
                       length:workList.size() * sizeof(BrickWorkItem)
                      options:MTLResourceStorageModeShared];

  // --- 5. PASS 3: PARALLEL FILL (GPU) ---
  {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:_psoFill]; // This will be the new kernel

    [enc setBuffer:workListBuffer offset:0 atIndex:0];
    [enc setBuffer:_sectorBuffer offset:0 atIndex:1];
    [enc setBuffer:_occupancyBuffer offset:0 atIndex:2];
    [enc setBuffer:_dataBuffer offset:0 atIndex:3];

    NSUInteger totalSubBricks = workList.size() * 8;
    MTLSize gridSize = MTLSizeMake(totalSubBricks, 1, 1);
    MTLSize groupSize = MTLSizeMake(64, 1, 1);

    [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:groupSize];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }

  // Sector Mask Buffer (for 128³ skipping)
  NSUInteger sectorMaskSize = sectorMasks.size() * sizeof(uint64_t);
  id<MTLBuffer> stagingSectorMask =
      [dev newBufferWithBytes:sectorMasks.data()
                       length:sectorMaskSize
                      options:MTLResourceStorageModeShared];
  _sectorMaskBuffer = [dev newBufferWithLength:sectorMaskSize
                                       options:MTLResourceStorageModePrivate];

  blitCmd = [queue commandBuffer];
  blitEnc = [blitCmd blitCommandEncoder];
  [blitEnc copyFromBuffer:stagingSectorMask
             sourceOffset:0
                 toBuffer:_sectorMaskBuffer
        destinationOffset:0
                     size:sectorMaskSize];
  [blitEnc endEncoding];
  [blitCmd commit];
  [blitCmd waitUntilCompleted];

  NSLog(@"[MaterialMap] World Generation Complete.");

  // Cleanup temporary buffers
  sectorAnalysisBuffer = nil;
}

MaterialMap::MaterialMap() {
  // Grab the global Metal device
  _device = get_device();

  id<MTLDevice> dev = (id<MTLDevice>)_device;
  id<MTLLibrary> lib = [dev newDefaultLibrary];

  if (lib) {
    NSError *err = nil;

    id<MTLFunction> fnAnalyze =
        [lib newFunctionWithName:@"XMap_AnalyzeSectors"];
    if (fnAnalyze) {
      _psoAnalyze = [dev newComputePipelineStateWithFunction:fnAnalyze
                                                       error:&err];
      if (err)
        NSLog(@"[MaterialMap] Error creating Analyze PSO: %@", err);
    }
    id<MTLFunction> fnFill = [lib newFunctionWithName:@"XMap_FillBricks"];
    if (fnFill) {
      _psoFill = [dev newComputePipelineStateWithFunction:fnFill error:&err];
      if (err)
        NSLog(@"[MaterialMap] Error creating Fill PSO: %@", err);
    }
  }
}

MaterialMap::~MaterialMap() {
  _indirectionTexture = nil;
  _sectorBuffer = nil;
  _occupancyBuffer = nil;
  _dataBuffer = nil;
  _sectorMaskBuffer = nil;
  _psoAnalyze = nil;
  _psoFill = nil;
  _device = nil;
}

id MaterialMap::GetIndirectionTexture() { return _indirectionTexture; }

id MaterialMap::GetSectorBuffer() { return _sectorBuffer; }

id MaterialMap::GetOccupancyBuffer() { return _occupancyBuffer; }

id MaterialMap::GetDataBuffer() { return _dataBuffer; }

id MaterialMap::GetSectorMaskBuffer() { return _sectorMaskBuffer; }