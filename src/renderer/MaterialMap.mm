#import "renderer/MaterialMap.hpp"
#import "State.hpp"
#import "TerrainGeneration.h"
#import "cumath.h"
#import "renderer/MetalDevice.hpp"
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