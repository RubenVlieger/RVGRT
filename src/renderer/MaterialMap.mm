#import "renderer/MaterialMap.hpp"
#import "State.hpp"
#import "TerrainGeneration.h"
#import "VoxelQuery.hpp"
#import "cumath.h"
#import "renderer/Metal/MetalDevice.hpp"
#include <algorithm>
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
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  desc.storageMode = MTLStorageModeManaged;
  id<MTLTexture> tex = [dev newTextureWithDescriptor:desc];
  tex.label = label;
  return tex;
}

// Positive modulo (C++ % can be negative)
int posmod(int a, int m) { return ((a % m) + m) % m; }

} // namespace

// ============================================================================
// Constructor / Destructor
// ============================================================================

MaterialMap::MaterialMap()
    : _brickPool(), _firstUpdate(true), _nextSectorHandle(1) {
  _device = get_device();

  id<MTLDevice> dev = (id<MTLDevice>)_device;
  id<MTLLibrary> lib = [dev newDefaultLibrary];

  _commandQueue = [dev newCommandQueue];

  // Compute indirection dimensions from world size
  _indW = (int)(SIZEX / 32);
  _indH = (int)(SIZEY / 32);
  _indD = (int)(SIZEZ / 32);

  _worldOrigin = simd_make_int3(0, 0, 0);
  _lastCameraSector =
      simd_make_int3(INT_MAX, INT_MAX, INT_MAX); // Force first update

  // Initialize sector states
  int totalCells = _indW * _indH * _indD;
  _sectorStates.resize(totalCells);
  for (auto &s : _sectorStates) {
    s.isLoaded = false;
    s.isLOD = false;
    s.isAnalyzing = false;
    s.sectorHandle = 0;
    s.brickPoolBase = UINT32_MAX;
    s.brickCount = 0;
    s.worldX = s.worldY = s.worldZ = INT_MAX;
  }

  // Pre-allocate sector info CPU array
  // Handle 0 is the null sector (unused), so we start at 1
  _sectorInfoCPU.resize(MAX_ACTIVE_SECTORS + 1);
  memset(_sectorInfoCPU.data(), 0, _sectorInfoCPU.size() * sizeof(SectorInfo));

  _indirectionCPU.resize(totalCells, 0);

  // --- Create GPU Resources ---

  // L1: Indirection 3D Texture
  _indirectionTexture = create3DTex(dev, _indW, _indH, _indD,
                                    MTLPixelFormatR32Uint, @"Indirection");

  // Zero out the indirection texture
  {
    [(id<MTLTexture>)_indirectionTexture
        replaceRegion:MTLRegionMake3D(0, 0, 0, _indW, _indH, _indD)
          mipmapLevel:0
                slice:0
            withBytes:_indirectionCPU.data()
          bytesPerRow:_indW * sizeof(uint32_t)
        bytesPerImage:_indW * _indH * sizeof(uint32_t)];

#if !TARGET_OS_IPHONE
    id<MTLCommandBuffer> cmd = [_commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit synchronizeResource:(id<MTLTexture>)_indirectionTexture];
    [blit endEncoding];
    [cmd commit];
#endif
  }

  // L2: Sector buffer (pre-allocated for max sectors)
  NSUInteger sectorBufferSize = (MAX_ACTIVE_SECTORS + 1) * sizeof(SectorInfo);
  _sectorBuffer = [dev newBufferWithLength:sectorBufferSize
                                   options:MTLResourceStorageModeShared];
  ((id<MTLBuffer>)_sectorBuffer).label = @"SectorBuffer";

  // Super-sector masks
  int superX = (_indW + 3) / 4;
  int superY = (_indH + 3) / 4;
  int superZ = (_indD + 3) / 4;
  int totalSuper = superX * superY * superZ;
  _sectorMaskBuffer = [dev newBufferWithLength:totalSuper * sizeof(uint64_t)
                                       options:MTLResourceStorageModeShared];
  ((id<MTLBuffer>)_sectorMaskBuffer).label = @"SectorMasks";

  // Zero the sector mask buffer
  memset([(id<MTLBuffer>)_sectorMaskBuffer contents], 0,
         totalSuper * sizeof(uint64_t));

  // --- Load Compute Pipelines ---
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

    id<MTLFunction> fnLOD = [lib newFunctionWithName:@"XMap_AnalyzeLOD"];
    if (fnLOD) {
      _psoAnalyzeLOD = [dev newComputePipelineStateWithFunction:fnLOD
                                                          error:&err];
      if (err)
        NSLog(@"[MaterialMap] Error creating LOD PSO: %@", err);
    } else {
      NSLog(@"[MaterialMap] Warning: XMap_AnalyzeLOD kernel not found. LOD "
            @"will fall back to detail.");
    }

    id<MTLFunction> fnStream =
        [lib newFunctionWithName:@"XMap_AnalyzeStreaming"];
    if (fnStream) {
      _psoAnalyzeStreaming = [dev newComputePipelineStateWithFunction:fnStream
                                                                error:&err];
      if (err)
        NSLog(@"[MaterialMap] Error creating Streaming Analyze PSO: %@", err);
    }
  }
}

MaterialMap::~MaterialMap() {
  _indirectionTexture = nil;
  _sectorBuffer = nil;
  _sectorMaskBuffer = nil;
  _psoAnalyze = nil;
  _psoFill = nil;
  _psoAnalyzeLOD = nil;
  _commandQueue = nil;
  _device = nil;
}

// ============================================================================
// Coordinate Helpers
// ============================================================================

void MaterialMap::WorldToWrapped(int wx, int wy, int wz, int &ix, int &iy,
                                 int &iz) const {
  ix = posmod(wx, _indW);
  iy = posmod(wy, _indH);
  iz = posmod(wz, _indD);
}

int MaterialMap::WrappedToLinear(int ix, int iy, int iz) const {
  return ix + iy * _indW + iz * _indW * _indH;
}

// ============================================================================
// Sector Handle Management
// ============================================================================

uint32_t MaterialMap::AllocSectorHandle() {
  if (!_freeSectorHandles.empty()) {
    uint32_t h = _freeSectorHandles.back();
    _freeSectorHandles.pop_back();
    return h;
  }
  if (_nextSectorHandle >= MAX_ACTIVE_SECTORS) {
    NSLog(@"[MaterialMap] ERROR: ran out of sector handles!");
    return 0;
  }
  return _nextSectorHandle++;
}

void MaterialMap::FreeSectorHandle(uint32_t handle) {
  if (handle > 0) {
    _freeSectorHandles.push_back(handle);
  }
}

// ============================================================================
// GPU Upload Helpers
// ============================================================================

void MaterialMap::UploadIndirectionCell(int ix, int iy, int iz,
                                        uint32_t value) {
  int idx = WrappedToLinear(ix, iy, iz);
  _indirectionCPU[idx] = value;
}

void MaterialMap::UploadSectorInfo(uint32_t handle, const SectorInfo &info) {
  _sectorInfoCPU[handle] = info;

  id<MTLBuffer> buf = (id<MTLBuffer>)_sectorBuffer;
  SectorInfo *ptr = (SectorInfo *)[buf contents];
  ptr[handle] = info;
}

// ============================================================================
// Sector Load / Unload
// ============================================================================

void MaterialMap::UnloadSector(int ix, int iy, int iz) {
  int idx = WrappedToLinear(ix, iy, iz);
  SectorState &state = _sectorStates[idx];

  if (!state.isLoaded)
    return;

  // Free bricks
  if (state.brickCount > 0 && state.brickPoolBase != UINT32_MAX) {
    _brickPool.Free(state.brickPoolBase, state.brickCount);
  }

  // Free sector handle
  FreeSectorHandle(state.sectorHandle);

  // Clear indirection
  UploadIndirectionCell(ix, iy, iz, SECTOR_HANDLE_EMPTY);

  // Reset state
  state.isLoaded = false;
  state.isLOD = false;
  state.sectorHandle = 0;
  state.brickPoolBase = UINT32_MAX;
  state.brickCount = 0;
  state.worldX = state.worldY = state.worldZ = INT_MAX;
}

void MaterialMap::LoadSector(int wx, int wy, int wz, bool isLOD,
                             std::vector<BrickWorkItem> &workList) {
  int ix, iy, iz;
  WorldToWrapped(wx, wy, wz, ix, iy, iz);
  int idx = WrappedToLinear(ix, iy, iz);

  // Unload existing content if different
  SectorState &state = _sectorStates[idx];
  if (state.isLoaded) {
    if (state.worldX == wx && state.worldY == wy && state.worldZ == wz &&
        state.isLOD == isLOD) {
      return; // Already loaded correctly
    }
    UnloadSector(ix, iy, iz);
  }

  // Analyze this sector: determine which bricks are active
  float3 sectorWorldPos = {(float)(wx * 32), (float)(wy * 32),
                           (float)(wz * 32)};

  // Run analysis on CPU for single sectors
  uint64_t brickMask = 0;

  for (int b = 0; b < 64; b++) {
    int bx = b & 3;
    int bz = (b >> 2) & 3;
    int by = (b >> 4) & 3;
    float3 brickPos = sectorWorldPos;
    brickPos.x += bx * 8.0f;
    brickPos.y += by * 8.0f;
    brickPos.z += bz * 8.0f;

    if (isLOD) {
      // Fast LOD heuristic: 1 sample in the center of the 8x8x8 brick
      if (Evaluate(brickPos.x + 4.0f, brickPos.y + 4.0f, brickPos.z + 4.0f) >
          0.0f) {
        brickMask |= (1ULL << b);
      }
    } else {
      // Detail: Robust heuristic check
      bool active = false;
      for (int dz = 0; dz < 8 && !active; dz += 3) {
        for (int dy = 0; dy < 8 && !active; dy += 3) {
          for (int dx = 0; dx < 8 && !active; dx += 3) {
            if (Evaluate(brickPos.x + dx, brickPos.y + dy, brickPos.z + dz) >
                0.0f) {
              active = true;
            }
          }
        }
      }
      if (active)
        brickMask |= (1ULL << b);
    }
  }

  if (brickMask == 0) {
    // Fully empty sector — mark as empty (no handle needed)
    state.isLoaded = true;
    state.isLOD = isLOD;
    state.worldX = wx;
    state.worldY = wy;
    state.worldZ = wz;
    state.sectorHandle = 0;
    state.brickPoolBase = UINT32_MAX;
    state.brickCount = 0;
    UploadIndirectionCell(ix, iy, iz, SECTOR_HANDLE_EMPTY);
    return;
  }

  // Allocate sector handle
  uint32_t handle = AllocSectorHandle();
  if (handle == 0)
    return; // Failed

  // Count active bricks
  uint32_t activeBricks = __builtin_popcountll(brickMask);

  SectorInfo sInfo;
  sInfo.brickMask = brickMask;

  if (isLOD) {
    // LOD: no brick data allocated, just brickMask
    sInfo.baseBrickIndex = 0; // Not used for LOD
    sInfo.flags = SECTOR_FLAG_LOD;
    state.brickPoolBase = UINT32_MAX;
    state.brickCount = 0;
  } else {
    // Detail: allocate bricks
    uint32_t base = _brickPool.Allocate(activeBricks);
    if (base == UINT32_MAX) {
      NSLog(@"[MaterialMap] Brick pool full! Falling back to LOD for sector "
            @"(%d,%d,%d)",
            wx, wy, wz);
      sInfo.baseBrickIndex = 0;
      sInfo.flags = SECTOR_FLAG_LOD;
      state.brickPoolBase = UINT32_MAX;
      state.brickCount = 0;
      isLOD = true;
    } else {
      sInfo.baseBrickIndex = base;
      sInfo.flags = SECTOR_FLAG_DETAIL;
      state.brickPoolBase = base;
      state.brickCount = activeBricks;
    }
  }

  // Upload sector info
  UploadSectorInfo(handle, sInfo);

  // Upload indirection cell
  UploadIndirectionCell(ix, iy, iz, handle);

  // Update state
  state.isLoaded = true;
  state.isLOD = isLOD;
  state.worldX = wx;
  state.worldY = wy;
  state.worldZ = wz;
  state.sectorHandle = handle;

  // For detail sectors, queue brick data on GPU
  if (!isLOD && state.brickCount > 0) {
    uint32_t brickIdx = 0;
    for (int b = 0; b < 64; b++) {
      if ((brickMask >> b) & 1) {
        BrickWorkItem item;
        item.localBrickIndex = b;
        item.occupancyOffset = (uint64_t)(state.brickPoolBase + brickIdx) * 8;
        item.dataOffset = (uint64_t)(state.brickPoolBase + brickIdx) * 512;

        int dx = wx - _worldOrigin.x;
        int dy = wy - _worldOrigin.y;
        int dz = wz - _worldOrigin.z;
        item.sectorIndex = dx + dy * _indW + dz * _indW * _indH;

        workList.push_back(item);
        brickIdx++;
      }
    }
  }
}

// ============================================================================
// Streaming Update
// ============================================================================

bool MaterialMap::UpdateStreaming(simd_float3 cameraPos) 
{
  // 1. Process Completed Async Analyses
  std::vector<AsyncResult> completedResults;
  {
    std::lock_guard<std::mutex> lock(_asyncResultsMutex);
    completedResults = std::move(_asyncResults);
    _asyncResults.clear();
  }

  std::vector<BrickWorkItem> workList;
  bool anyChanged = false;

  for (const auto &res : completedResults) {
    int expectedWX = res.item.worldX;
    int expectedWY = res.item.worldY;
    int expectedWZ = res.item.worldZ;
    int idx = res.item.wrappedIdx;

    SectorState &state = _sectorStates[idx];

    // Safety check: is this result stale?
    if (state.worldX != expectedWX || state.worldY != expectedWY ||
        state.worldZ != expectedWZ) {
      continue;
    }

    state.isAnalyzing = false;
    uint64_t brickMask = res.brickMask;
    int ix, iy, iz;
    WorldToWrapped(expectedWX, expectedWY, expectedWZ, ix, iy, iz);

    if (brickMask == 0) {
      state.isLoaded = true;
      state.sectorHandle = 0;
      state.brickPoolBase = UINT32_MAX;
      state.brickCount = 0;
      UploadIndirectionCell(ix, iy, iz, SECTOR_HANDLE_EMPTY);
      anyChanged = true;
      continue;
    }

    uint32_t handle = AllocSectorHandle();
    if (handle == 0)
      continue;

    uint32_t activeBricks = __builtin_popcountll(brickMask);
    SectorInfo sInfo;
    sInfo.brickMask = brickMask;

    if (state.isLOD) {
      sInfo.baseBrickIndex = 0;
      sInfo.flags = SECTOR_FLAG_LOD;
      state.brickPoolBase = UINT32_MAX;
      state.brickCount = 0;
    } else {
      uint32_t base = _brickPool.Allocate(activeBricks);
      if (base == UINT32_MAX) {
        NSLog(@"[MaterialMap] Brick pool full! LOD fallback (%d,%d,%d)",
              expectedWX, expectedWY, expectedWZ);
        sInfo.baseBrickIndex = 0;
        sInfo.flags = SECTOR_FLAG_LOD;
        state.brickPoolBase = UINT32_MAX;
        state.brickCount = 0;
        state.isLOD = true;
      } else {
        sInfo.baseBrickIndex = base;
        sInfo.flags = SECTOR_FLAG_DETAIL;
        state.brickPoolBase = base;
        state.brickCount = activeBricks;
      }
    }

    UploadSectorInfo(handle, sInfo);
    UploadIndirectionCell(ix, iy, iz, handle);

    state.isLoaded = true;
    state.sectorHandle = handle;

    if (!state.isLOD && state.brickCount > 0) {
      uint32_t brickIdx = 0;
      for (int b = 0; b < 64; b++) {
        if ((brickMask >> b) & 1) {
          BrickWorkItem item;
          item.localBrickIndex = b;
          item.occupancyOffset = (uint64_t)(state.brickPoolBase + brickIdx) * 8;
          item.dataOffset = (uint64_t)(state.brickPoolBase + brickIdx) * 512;

          int dx = expectedWX - _worldOrigin.x;
          int dy = expectedWY - _worldOrigin.y;
          int dz = expectedWZ - _worldOrigin.z;
          item.sectorIndex = dx + dy * _indW + dz * _indW * _indH;

          workList.push_back(item);
          brickIdx++;
        }
      }
    }
    anyChanged = true;
  }

  // 2. Queue New Work
  int camSX = (int)std::floor(cameraPos.x / 32.0f);
  int camSY = (int)std::floor(cameraPos.y / 32.0f);
  int camSZ = (int)std::floor(cameraPos.z / 32.0f);
  simd_int3 camSector = simd_make_int3(camSX, camSY, camSZ);

  _worldOrigin = simd_make_int3(camSX - _indW / 2, 0, camSZ - _indD / 2);

  std::vector<SectorWorkItem> pendingRequests;
  int unloaded = 0;

  for (int dy = 0; dy < _indH; dy++) {
    for (int dz = 0; dz < _indD; dz++) {
      for (int dx = 0; dx < _indW; dx++) {
        int expectedWX = _worldOrigin.x + dx;
        int expectedWY = dy;
        int expectedWZ = _worldOrigin.z + dz;

        if (expectedWY < 0 || expectedWY >= _indH)
          continue;

        int distX = std::abs(expectedWX - camSX);
        int distZ = std::abs(expectedWZ - camSZ);
        int maxDist = std::max(distX, distZ);
        bool shouldBeLOD = (maxDist > DETAIL_RADIUS_SECTORS);

        int ix, iy, iz;
        WorldToWrapped(expectedWX, expectedWY, expectedWZ, ix, iy, iz);
        int idx = WrappedToLinear(ix, iy, iz);
        SectorState &state = _sectorStates[idx];

        if (state.isLoaded && state.worldX == expectedWX &&
            state.worldY == expectedWY && state.worldZ == expectedWZ &&
            state.isLOD == shouldBeLOD) {
          continue; // Already correct
        }

        if (state.isAnalyzing && state.worldX == expectedWX &&
            state.worldY == expectedWY && state.worldZ == expectedWZ &&
            state.isLOD == shouldBeLOD) {
          continue; // Already queued for analysis
        }

        if (state.isLoaded) {
          UnloadSector(ix, iy, iz);
          unloaded++;
        }

        state.isAnalyzing = false;

        SectorWorkItem item;
        item.worldX = expectedWX;
        item.worldY = expectedWY;
        item.worldZ = expectedWZ;
        item.wrappedIdx = idx;

        pendingRequests.push_back(item);

        state.worldX = expectedWX;
        state.worldY = expectedWY;
        state.worldZ = expectedWZ;
        state.isLOD = shouldBeLOD;
      }
    }
  }

// 3. Dispatch Async Analysis
  if (!pendingRequests.empty() && _psoAnalyzeStreaming) {
    // Sort pendingRequests by 3D distance to camera so closer chunks load first!
    std::sort(pendingRequests.begin(), pendingRequests.end(),
      [camSX, camSY, camSZ](const SectorWorkItem& a, const SectorWorkItem& b) {
        int dxA = a.worldX - camSX;
        int dyA = a.worldY - camSY;
        int dzA = a.worldZ - camSZ;
        int distSqA = dxA*dxA + dyA*dyA + dzA*dzA;

        int dxB = b.worldX - camSX;
        int dyB = b.worldY - camSY;
        int dzB = b.worldZ - camSZ;
        int distSqB = dxB*dxB + dyB*dyB + dzB*dzB;

        return distSqA < distSqB;
      });

    size_t numReq = std::min(pendingRequests.size(),
                             (size_t)16384); // Cap chunk requests per frame
                             
    // Only update the state for the items that will ACTUALLY be dispatched
    for (size_t i = 0; i < numReq; ++i) {
        const auto& item = pendingRequests[i];
        SectorState& state = _sectorStates[item.wrappedIdx];
        
        state.worldX = item.worldX;
        state.worldY = item.worldY;
        state.worldZ = item.worldZ;
        
        int distX = std::abs(item.worldX - camSX);
        int distZ = std::abs(item.worldZ - camSZ);
        int maxDist = std::max(distX, distZ);
        state.isLOD = (maxDist > DETAIL_RADIUS_SECTORS);
        
        // Now safely mark it as analyzing
        state.isAnalyzing = true; 
    }

    id<MTLDevice> dev = (id<MTLDevice>)_device;
    id<MTLCommandQueue> queue = (id<MTLCommandQueue>)_commandQueue;

    id<MTLBuffer> inBuf =[dev newBufferWithBytes:pendingRequests.data()
                         length:numReq * sizeof(SectorWorkItem)
                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuf =[dev newBufferWithLength:numReq * sizeof(uint64_t)
                         options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:_psoAnalyzeStreaming];[enc setBuffer:inBuf offset:0 atIndex:0];[enc setBuffer:outBuf offset:0 atIndex:1];

    uint32_t totalItems = (uint32_t)numReq;[enc setBytes:&totalItems length:sizeof(uint32_t) atIndex:2];

    MTLSize gridSize = MTLSizeMake(numReq, 1, 1);
    MTLSize groupSize =
        MTLSizeMake(std::min((NSUInteger)64, (NSUInteger)numReq), 1, 1);[enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
    [enc endEncoding];

    std::vector<SectorWorkItem> dispatchedItems(
        pendingRequests.begin(), pendingRequests.begin() + numReq);[cmd addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      uint64_t *masks = (uint64_t *)[outBuf contents];
      std::vector<AsyncResult> results;
      results.reserve(dispatchedItems.size());
      for (size_t i = 0; i < dispatchedItems.size(); ++i) {
        AsyncResult res;
        res.item = dispatchedItems[i];
        res.brickMask = masks[i];
        results.push_back(res);
      }

      std::lock_guard<std::mutex> lock(this->_asyncResultsMutex);
      this->_asyncResults.insert(this->_asyncResults.end(), results.begin(),
                                 results.end());
    }];

    [cmd commit];
  }

  // 4. Update Render State
  if (anyChanged || unloaded > 0) {
    RebuildSectorMasks();

    [(id<MTLTexture>)_indirectionTexture
        replaceRegion:MTLRegionMake3D(0, 0, 0, _indW, _indH, _indD)
          mipmapLevel:0
                slice:0
            withBytes:_indirectionCPU.data()
          bytesPerRow:_indW * sizeof(uint32_t)
        bytesPerImage:_indW * _indH * sizeof(uint32_t)];

#if !TARGET_OS_IPHONE
    id<MTLCommandQueue> queue = (id<MTLCommandQueue>)_commandQueue;
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit synchronizeResource:(id<MTLTexture>)_indirectionTexture];
    [blit endEncoding];
    [cmd commit];
#endif

    if (!workList.empty()) {
      id<MTLDevice> dev = (id<MTLDevice>)_device;
      id<MTLCommandQueue> fqueue = (id<MTLCommandQueue>)_commandQueue;

      id<MTLBuffer> workListBuffer =
          [dev newBufferWithBytes:workList.data()
                           length:workList.size() * sizeof(BrickWorkItem)
                          options:MTLResourceStorageModeShared];

      id<MTLCommandBuffer> fcmd = [fqueue commandBuffer];
      id<MTLComputeCommandEncoder> fenc = [fcmd computeCommandEncoder];

      [fenc setComputePipelineState:_psoFill];

      [fenc setBuffer:workListBuffer offset:0 atIndex:0];
      [fenc setBuffer:(id<MTLBuffer>)_sectorBuffer offset:0 atIndex:1];
      [fenc setBuffer:(id<MTLBuffer>)_brickPool.GetOccupancyBuffer()
               offset:0
              atIndex:2];
      [fenc setBuffer:(id<MTLBuffer>)_brickPool.GetDataBuffer()
               offset:0
              atIndex:3];

      simd_int3 wo = _worldOrigin;
      [fenc setBytes:&wo length:sizeof(simd_int3) atIndex:4];

      NSUInteger totalSubBricks = workList.size() * 8;
      MTLSize fgridSize = MTLSizeMake(totalSubBricks, 1, 1);
      MTLSize fgroupSize = MTLSizeMake(64, 1, 1);

      [fenc dispatchThreadgroups:fgridSize threadsPerThreadgroup:fgroupSize];
      [fenc endEncoding];
      [fcmd commit];
    }
  }

  if (_firstUpdate) {
    _firstUpdate = false;
  }

  _lastCameraSector = camSector;

  return anyChanged;
}

// ============================================================================
// Initial Generation (called at startup)
// ============================================================================

void MaterialMap::GenerateDynamic() {
  NSLog(@"[MaterialMap] Starting initial streaming world generation...");

  // Use character spawn position as initial camera position
  simd_float3 spawnPos = simd_make_float3(State::state.character.position.x,
                                          State::state.character.position.y,
                                          State::state.character.position.z);

  UpdateStreaming(spawnPos);

  NSLog(@"[MaterialMap] Initial world generation complete.");
}

// ============================================================================
// Super-Sector Masks
// ============================================================================

void MaterialMap::RebuildSectorMasks() {
  int superX = (_indW + 3) / 4;
  int superY = (_indH + 3) / 4;
  int superZ = (_indD + 3) / 4;
  int totalSuper = superX * superY * superZ;

  uint64_t *masks = (uint64_t *)[(id<MTLBuffer>)_sectorMaskBuffer contents];
  memset(masks, 0, totalSuper * sizeof(uint64_t));

  for (int iy = 0; iy < _indH; iy++) {
    for (int iz = 0; iz < _indD; iz++) {
      for (int ix = 0; ix < _indW; ix++) {
        int idx = WrappedToLinear(ix, iy, iz);
        if (_sectorStates[idx].isLoaded &&
            _sectorStates[idx].sectorHandle != 0) {
          int superIdx =
              (ix / 4) + (iz / 4) * superX + (iy / 4) * superX * superZ;
          int lx = ix & 3, ly = iy & 3, lz = iz & 3;
          int bitIdx = lx + (lz << 2) + (ly << 4);
          masks[superIdx] |= (1ULL << bitIdx);
        }
      }
    }
  }
}

// ============================================================================
// Batch Generation (placeholders for future GPU-batch path)
// ============================================================================

void MaterialMap::GenerateDetailBatch(
    const std::vector<SectorWorkItem> &sectors) {
  std::vector<BrickWorkItem> workList;
  for (const auto &s : sectors) {
    LoadSector(s.worldX, s.worldY, s.worldZ, false, workList);
  }
}

void MaterialMap::GenerateLODBatch(const std::vector<SectorWorkItem> &sectors) {
  std::vector<BrickWorkItem> workList;
  for (const auto &s : sectors) {
    LoadSector(s.worldX, s.worldY, s.worldZ, true, workList);
  }
}

// ============================================================================
// Getters
// ============================================================================

id MaterialMap::GetIndirectionTexture() { return _indirectionTexture; }
id MaterialMap::GetSectorBuffer() { return _sectorBuffer; }
id MaterialMap::GetOccupancyBuffer() { return _brickPool.GetOccupancyBuffer(); }
id MaterialMap::GetDataBuffer() { return _brickPool.GetDataBuffer(); }
id MaterialMap::GetSectorMaskBuffer() { return _sectorMaskBuffer; }

// ============================================================================
// Block Modification (Phase 2)
// ============================================================================
//
// These methods modify voxel data directly in the GPU shared-memory brick pool
// buffers, making edits immediately visible to the path tracer. They also
// update the block-edit overlay map (g_blockEdits) for collision/raycasting.
//
// SVO Structure:
//   L1: Indirection texture (3D) → sector handle (1-based, 0 = empty)
//   L2: Sector buffer[handle] → SectorInfo { baseBrickIndex, flags, brickMask }
//   L3: Brick pool occupancy → 8 x uint64_t per brick (8x8x8 voxels, 1 bit each)
//   L4: Brick pool data    → 512 x uint8_t per brick (material IDs)
//
// Brick Mask Layout (64 bits = 4x4x4 bricks per sector):
//   Bit index b maps to brick at (b & 3, (b >> 4) & 3, (b >> 2) & 3)
//   This matches GetLinearIndex4 in intersections.h:
//     GetLinearIndex4(p) = (p.x & 3) + ((p.z & 3) << 2) + ((p.y & 3) << 4)
//
// Sub-brick Occupancy Layout (512 bits = 8 x uint64_t per brick):
//   Each 8x8x8 brick is subdivided into 2x2x2 = 8 sub-bricks of 4x4x4 voxels.
//   Sub-brick index subIdx = localSubPos.x + localSubPos.z * 2 + localSubPos.y * 4
//   where localSubPos = (localPos >> 2) & 1 for each axis.
//   Each sub-brick has 64 voxel occupancy bits using GetLinearIndex4:
//     vIdx = (vRel.x & 3) + ((vRel.z & 3) << 2) + ((vRel.y & 3) << 4)
//   where vRel = localPos & 3 for each axis.
//
// Data Layout:
//   Data byte index = (brickPoolIndex * 512) + (subIdx * 64) + vIdx
// ============================================================================

// Helper: Compute LinearIndex4 matching the GPU/cross-platform implementation
static inline uint32_t GetLinearIndex4Local(uint32_t px, uint32_t py, uint32_t pz) {
    return (px & 3) + ((pz & 3) << 2) + ((py & 3) << 4);
}

// Helper: Compute prefix population count of a 64-bit mask up to (not including) bit position
static inline uint32_t PrefixPopcount64Local(uint64_t mask, uint32_t width) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < width; ++i) {
        count += (mask >> i) & 1;
    }
    return count;
}

bool MaterialMap::RemoveVoxel(int32_t wx, int32_t wy, int32_t wz) {
    // Step 1: Convert world voxel coord to sector coord
    int32_t sx = wx >> 5; // wx / 32
    int32_t sy = wy >> 5;
    int32_t sz = wz >> 5;

    // Step 2: Find the sector state via toroidal wrapping
    int32_t relX = sx - _worldOrigin.x;
    int32_t relY = sy - _worldOrigin.y;
    int32_t relZ = sz - _worldOrigin.z;

    // Out of loaded region bounds
    if (relX < 0 || relX >= _indW || relY < 0 || relY >= _indH ||
        relZ < 0 || relZ >= _indD) {
        return false;
    }

    uint32_t wx_wrapped = ((uint32_t)(sx % _indW) + _indW) % _indW;
    uint32_t wy_wrapped = ((uint32_t)(sy % _indH) + _indH) % _indH;
    uint32_t wz_wrapped = ((uint32_t)(sz % _indD) + _indD) % _indD;
    int idx = WrappedToLinear(wx_wrapped, wy_wrapped, wz_wrapped);
    const SectorState &state = _sectorStates[idx];

    if (!state.isLoaded || state.sectorHandle == 0 || state.isLOD) {
        return false;
    }

    // Step 3: Get sector info
    uint32_t handle = state.sectorHandle;
    SectorInfo sInfo = _sectorInfoCPU[handle];

    // Step 4: Compute local position within the 32x32x32 sector
    uint32_t lx = wx & 31;
    uint32_t ly = wy & 31;
    uint32_t lz = wz & 31;

    // Step 5: Compute brick index within sector (4x4x4 bricks)
    uint32_t bx = (lx >> 3) & 3;
    uint32_t by = (ly >> 3) & 3;
    uint32_t bz = (lz >> 3) & 3;
    uint32_t brickLinearIdx = GetLinearIndex4Local(bx, by, bz);

    // Step 6: Check if this brick exists in the brick mask
    uint64_t brickMaskBit = 1ULL << brickLinearIdx;
    if ((sInfo.brickMask & brickMaskBit) == 0) {
        // This brick is already empty, voxel doesn't exist
        return false;
    }

    // Step 7: Compute packed brick offset
    uint32_t packedBrickOffset = PrefixPopcount64Local(sInfo.brickMask, brickLinearIdx);
    uint32_t brickPoolIndex = sInfo.baseBrickIndex + packedBrickOffset;

    // Step 8: Compute sub-brick index (2x2x2 within brick)
    uint32_t subPx = (lx >> 2) & 1;
    uint32_t subPy = (ly >> 2) & 1;
    uint32_t subPz = (lz >> 2) & 1;
    uint32_t subIdx = subPx + (subPz << 1) + (subPy << 2); // matches (subPos.x + subPos.z*2 + subPos.y*4)

    // Step 9: Compute voxel index within 4x4x4 sub-brick
    uint32_t vx = lx & 3;
    uint32_t vy = ly & 3;
    uint32_t vz = lz & 3;
    uint32_t vIdx = GetLinearIndex4Local(vx, vy, vz);

    // Step 10: Access occupancy data in the brick pool (shared-memory Metal buffer)
    uint64_t occIndexBase = (uint64_t)brickPoolIndex * 8;
    id<MTLBuffer> occBuffer = (id<MTLBuffer>)_brickPool.GetOccupancyBuffer();
    uint64_t *occPtr = (uint64_t *)[occBuffer contents];

    uint64_t &voxMask = occPtr[occIndexBase + subIdx];

    // Check if this voxel is already empty
    uint64_t voxelBit = 1ULL << vIdx;
    if ((voxMask & voxelBit) == 0) {
        // Voxel is already air
        return false;
    }

    // Clear the voxel occupancy bit
    voxMask &= ~voxelBit;

    // Step 11: Clear the data byte (set to air)
    id<MTLBuffer> dataBuffer = (id<MTLBuffer>)_brickPool.GetDataBuffer();
    uint8_t *dataPtr = (uint8_t *)[dataBuffer contents];
    uint64_t dataOffset = (uint64_t)brickPoolIndex * 512 + (subIdx * 64) + vIdx;
    dataPtr[dataOffset] = 0; // MAT_AIR

    // Step 11b: Synchronize so GPU sees the CPU-written changes
    id<MTLCommandQueue> syncQueue = (id<MTLCommandQueue>)_commandQueue;
    id<MTLCommandBuffer> syncCmd = [syncQueue commandBuffer];
    id<MTLBlitCommandEncoder> syncEnc = [syncCmd blitCommandEncoder];
    [syncEnc synchronizeResource:(id<MTLBuffer>)occBuffer];
    [syncEnc synchronizeResource:(id<MTLBuffer>)dataBuffer];
    [syncEnc endEncoding];
    [syncCmd commit];
    [syncCmd waitUntilCompleted];

    // Step 12: Update the block-edit overlay map
    SetBlockEdit(wx, wy, wz, 0);

    return true;
}

bool MaterialMap::PlaceVoxel(int32_t wx, int32_t wy, int32_t wz, uint8_t matID) {
    int32_t sx = wx >> 5;
    int32_t sy = wy >> 5;
    int32_t sz = wz >> 5;

    int32_t relX = sx - _worldOrigin.x;
    int32_t relY = sy - _worldOrigin.y;
    int32_t relZ = sz - _worldOrigin.z;

    if (relX < 0 || relX >= _indW || relY < 0 || relY >= _indH ||
        relZ < 0 || relZ >= _indD) {
        return false;
    }

    uint32_t wx_wrapped = ((uint32_t)(sx % _indW) + _indW) % _indW;
    uint32_t wy_wrapped = ((uint32_t)(sy % _indH) + _indH) % _indH;
    uint32_t wz_wrapped = ((uint32_t)(sz % _indD) + _indD) % _indD;
    int idx = WrappedToLinear(wx_wrapped, wy_wrapped, wz_wrapped);
    SectorState &state = _sectorStates[idx];

    if (!state.isLoaded || state.sectorHandle == 0) {
        // Sector not loaded — cannot place voxel
        return false;
    }

    if (state.isLOD) {
        // LOD sector — cannot place voxel (no brick data allocated)
        return false;
    }

    uint32_t handle = state.sectorHandle;
    SectorInfo sInfo = _sectorInfoCPU[handle];

    uint32_t lx = wx & 31;
    uint32_t ly = wy & 31;
    uint32_t lz = wz & 31;

    uint32_t bx = (lx >> 3) & 3;
    uint32_t by = (ly >> 3) & 3;
    uint32_t bz = (lz >> 3) & 3;
    uint32_t brickLinearIdx = GetLinearIndex4Local(bx, by, bz);

    uint64_t brickMaskBit = 1ULL << brickLinearIdx;

    uint32_t subPx = (lx >> 2) & 1;
    uint32_t subPy = (ly >> 2) & 1;
    uint32_t subPz = (lz >> 2) & 1;
    uint32_t subIdx = subPx + (subPz << 1) + (subPy << 2);

    uint32_t vx = lx & 3;
    uint32_t vy = ly & 3;
    uint32_t vz = lz & 3;
    uint32_t vIdx = GetLinearIndex4Local(vx, vy, vz);
    uint64_t voxelBit = 1ULL << vIdx;

    // ── Case A: Brick already exists in the mask ──
    if (sInfo.brickMask & brickMaskBit) {
        uint32_t packedOffset = PrefixPopcount64Local(sInfo.brickMask, brickLinearIdx);
        uint32_t brickPoolIndex = sInfo.baseBrickIndex + packedOffset;

        uint64_t occIndexBase = (uint64_t)brickPoolIndex * 8;
        id<MTLBuffer> occBuffer = (id<MTLBuffer>)_brickPool.GetOccupancyBuffer();
        uint64_t *occPtr = (uint64_t *)[occBuffer contents];

        // Set the occupancy bit
        occPtr[occIndexBase + subIdx] |= voxelBit;

        // Write the material ID
        id<MTLBuffer> dataBuffer = (id<MTLBuffer>)_brickPool.GetDataBuffer();
        uint8_t *dataPtr = (uint8_t *)[dataBuffer contents];
        uint64_t dataOffset = (uint64_t)brickPoolIndex * 512 + (subIdx * 64) + vIdx;
        dataPtr[dataOffset] = matID;

        // Synchronize so GPU sees the CPU-written changes
        id<MTLCommandQueue> syncQueue = (id<MTLCommandQueue>)_commandQueue;
        id<MTLCommandBuffer> syncCmd = [syncQueue commandBuffer];
        id<MTLBlitCommandEncoder> syncEnc = [syncCmd blitCommandEncoder];
        [syncEnc synchronizeResource:(id<MTLBuffer>)occBuffer];
        [syncEnc synchronizeResource:(id<MTLBuffer>)dataBuffer];
        [syncEnc endEncoding];
        [syncCmd commit];
        [syncCmd waitUntilCompleted];

        SetBlockEdit(wx, wy, wz, matID);
        return true;
    }

    // ── Case B: Brick doesn't exist — need to allocate and rebuild ──
    //
    // This requires:
    // 1. Computing a new brickMask with the new brick bit set
    // 2. Allocating new brick pool slots (popcount of new mask)
    // 3. Copying existing brick data from old allocation to new, shifting
    //    for the inserted brick
    // 4. Initializing the new brick's occupancy and data for just this voxel
    // 5. Freeing the old brick pool allocation
    // 6. Updating SectorInfo and sector state

    uint64_t newBrickMask = sInfo.brickMask | brickMaskBit;
    uint32_t newBrickCount = __builtin_popcountll(newBrickMask);

    // Allocate new brick pool slots
    uint32_t newBase = _brickPool.Allocate(newBrickCount);
    if (newBase == UINT32_MAX) {
        NSLog(@"[MaterialMap] Brick pool full! Cannot place voxel at (%d,%d,%d)", wx, wy, wz);
        return false;
    }

    uint32_t oldBase = sInfo.baseBrickIndex;
    uint32_t oldBrickCount = __builtin_popcountll(sInfo.brickMask);

    // Get buffer pointers
    id<MTLBuffer> occBuffer = (id<MTLBuffer>)_brickPool.GetOccupancyBuffer();
    uint64_t *occPtr = (uint64_t *)[occBuffer contents];
    id<MTLBuffer> dataBuffer = (id<MTLBuffer>)_brickPool.GetDataBuffer();
    uint8_t *dataPtr = (uint8_t *)[dataBuffer contents];

    // Copy existing bricks to their new positions, inserting a blank for the new brick
    uint32_t oldBrickIdx = 0;
    uint32_t newBrickIdx = 0;
    for (uint32_t b = 0; b < 64; ++b) {
        if (newBrickMask & (1ULL << b)) {
            if (sInfo.brickMask & (1ULL << b)) {
                // Existing brick — copy from old allocation
                uint32_t oldPoolIdx = oldBase + oldBrickIdx;
                uint32_t newPoolIdx = newBase + newBrickIdx;

                // Copy occupancy (8 x uint64_t per brick)
                for (uint32_t s = 0; s < 8; ++s) {
                    occPtr[(uint64_t)newPoolIdx * 8 + s] = occPtr[(uint64_t)oldPoolIdx * 8 + s];
                }

                // Copy data (512 bytes per brick)
                memcpy(dataPtr + (uint64_t)newPoolIdx * 512,
                       dataPtr + (uint64_t)oldPoolIdx * 512,
                       512);

                oldBrickIdx++;
            } else {
                // This is the newly inserted brick — initialize with just our voxel
                uint32_t newPoolIdx = newBase + newBrickIdx;

                // Zero all occupancy and data for the new brick
                for (uint32_t s = 0; s < 8; ++s) {
                    occPtr[(uint64_t)newPoolIdx * 8 + s] = 0;
                }
                memset(dataPtr + (uint64_t)newPoolIdx * 512, 0, 512);

                // Set the single voxel occupancy bit
                occPtr[(uint64_t)newPoolIdx * 8 + subIdx] |= voxelBit;

                // Write the material ID
                dataPtr[(uint64_t)newPoolIdx * 512 + (subIdx * 64) + vIdx] = matID;
            }
            newBrickIdx++;
        }
    }

    // Synchronize so GPU sees the CPU-written changes (memcpy for existing bricks + init of new brick)
    id<MTLCommandQueue> syncQueue = (id<MTLCommandQueue>)_commandQueue;
    id<MTLCommandBuffer> syncCmd = [syncQueue commandBuffer];
    id<MTLBlitCommandEncoder> syncEnc = [syncCmd blitCommandEncoder];
    [syncEnc synchronizeResource:(id<MTLBuffer>)occBuffer];
    [syncEnc synchronizeResource:(id<MTLBuffer>)dataBuffer];
    [syncEnc endEncoding];
    [syncCmd commit];
    [syncCmd waitUntilCompleted];

    // Free old brick pool allocation
    if (oldBrickCount > 0) {
        _brickPool.Free(oldBase, oldBrickCount);
    }

    // Update sector info
    sInfo.brickMask = newBrickMask;
    sInfo.baseBrickIndex = newBase;
    sInfo.flags = SECTOR_FLAG_DETAIL;
    UploadSectorInfo(handle, sInfo);

    // Update sector state
    state.brickPoolBase = newBase;
    state.brickCount = newBrickCount;

    // Update the block-edit overlay map
    SetBlockEdit(wx, wy, wz, matID);

    return true;
}

void MaterialMap::ResetBlockEdits() {
    // Iterate over all block edits and restore each voxel to procedural state
    for (auto it = g_blockEdits.begin(); it != g_blockEdits.end(); ) {
        int32_t x = it->first.x;
        int32_t y = it->first.y;
        int32_t z = it->first.z;
        uint8_t overlayMatID = it->second;

        // Determine what the procedural terrain says at this position
        bool proceduralSolid = Evaluate(float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f) > 0.0f;

        if (overlayMatID == 0 && proceduralSolid) {
            // Overlay says air, but procedural says solid → restore solid
            PlaceVoxel(x, y, z, 1); // Use MAT_STONE as default
        } else if (overlayMatID != 0 && !proceduralSolid) {
            // Overlay says solid (placed), but procedural says air → remove
            RemoveVoxel(x, y, z);
        }
        // If both agree (overlay solid + procedural solid, or overlay air + procedural air),
        // no SVO change needed, but we still remove the overlay entry.

        it = g_blockEdits.erase(it);
    }
}