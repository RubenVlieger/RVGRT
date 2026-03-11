#include "renderer/CUDA/CudaMaterialMap.cuh"
#include "TerrainGeneration.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>

// Forward declarations for kernels from material_gen.shader
extern __global__ void XMap_AnalyzeStreaming(
    SectorWorkItem* workItems,
    uint64_t* resultBuffer,
    uint32_t totalItems,
    int _width, int _height
);

extern __global__ void XMap_FillBricks(
    BrickWorkItem* workList,
    SectorInfo* sectorBuffer,
    uint64_t* occupancyBuffer,
    uint8_t* dataBuffer,
    int3 worldOrigin,
    int _width, int _height
);

// Host-side popcount for 64-bit (works on MSVC)
static inline uint32_t host_popcount64(uint64_t x) {
#if defined(_M_X64) || defined(__x86_64__)
    return __popcnt64(x);
#else
    // Fallback for 32-bit
    return static_cast<uint32_t>(__popcnt(static_cast<uint32_t>(x)) + __popcnt(static_cast<uint32_t>(x >> 32)));
#endif
}

// Positive modulo (C++ % can be negative)
static inline int posmod(int a, int m) { 
  return ((a % m) + m) % m; 
}

CudaMaterialMap::CudaMaterialMap()
    : d_indirection(nullptr), d_sectors(nullptr), d_sectorMasks(nullptr),
      d_workItems(nullptr), d_analysisResults(nullptr), _workItemCapacity(16384),
      d_brickWorkList(nullptr), _brickWorkCapacity(65536),
      _brickPool(), _firstUpdate(true), _nextSectorHandle(1), _cudaStream(nullptr) {
  
  // Create CUDA stream for async operations
  cudaError_t err = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&_cudaStream));
  if (err != cudaSuccess) {
    fprintf(stderr, "[CudaMaterialMap] ERROR: Failed to create CUDA stream: %s\n", cudaGetErrorString(err));
    _cudaStream = nullptr;
  }

  // Compute indirection dimensions from world size
  _indW = (int)(SIZEX / 32);
  _indH = (int)(SIZEY / 32);
  _indD = (int)(SIZEZ / 32);

  _worldOrigin = make_int3(0, 0, 0);
  _lastCameraSector = make_int3(INT_MAX, INT_MAX, INT_MAX); // Force first update

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

  // L1: Indirection 3D Texture as flat buffer
  size_t indirectionSize = totalCells * sizeof(uint32_t);
  err = cudaMalloc(&d_indirection, indirectionSize);
  if (err != cudaSuccess) {
    fprintf(stderr, "[CudaMaterialMap] ERROR: Failed to allocate indirection buffer: %s\n", cudaGetErrorString(err));
    return;
  }
  cudaMemset(d_indirection, 0, indirectionSize);

  // L2: Sector buffer (pre-allocated for max sectors)
  size_t sectorBufferSize = (MAX_ACTIVE_SECTORS + 1) * sizeof(SectorInfo);
  err = cudaMalloc(&d_sectors, sectorBufferSize);
  if (err != cudaSuccess) {
    fprintf(stderr, "[CudaMaterialMap] ERROR: Failed to allocate sector buffer: %s\n", cudaGetErrorString(err));
    cudaFree(d_indirection);
    d_indirection = nullptr;
    return;
  }
  cudaMemset(d_sectors, 0, sectorBufferSize);

  // Super-sector masks
  int superX = (_indW + 3) / 4;
  int superY = (_indH + 3) / 4;
  int superZ = (_indD + 3) / 4;
  int totalSuper = superX * superY * superZ;
  err = cudaMalloc(&d_sectorMasks, totalSuper * sizeof(uint64_t));
  if (err != cudaSuccess) {
    fprintf(stderr, "[CudaMaterialMap] ERROR: Failed to allocate sector mask buffer: %s\n", cudaGetErrorString(err));
    cudaFree(d_indirection);
    cudaFree(d_sectors);
    d_indirection = nullptr;
    d_sectors = nullptr;
    return;
  }
  cudaMemset(d_sectorMasks, 0, totalSuper * sizeof(uint64_t));

  // Async streaming buffers
  err = cudaMalloc(&d_workItems, _workItemCapacity * sizeof(SectorWorkItem));
  err = cudaMalloc(&d_analysisResults, _workItemCapacity * sizeof(uint64_t));
  err = cudaMalloc(&d_brickWorkList, _brickWorkCapacity * sizeof(BrickWorkItem));
  if (err != cudaSuccess) {
        fprintf(stderr, "[CudaMaterialMap] ERROR: Failed to allocate streaming work buffers.\n");
  }

  printf("[CudaMaterialMap] Initialized: %dx%dx%d indirection (%d sectors), %d super-sectors\n",
         _indW, _indH, _indD, totalCells, totalSuper);
}

CudaMaterialMap::~CudaMaterialMap() {
  if (_cudaStream) {
    cudaStreamDestroy(reinterpret_cast<cudaStream_t>(_cudaStream));
    _cudaStream = nullptr;
  }
  if (d_indirection) {
    cudaFree(d_indirection);
    d_indirection = nullptr;
  }
  if (d_sectors) {
    cudaFree(d_sectors);
    d_sectors = nullptr;
  }
  if (d_sectorMasks) {
    cudaFree(d_sectorMasks);
    d_sectorMasks = nullptr;
  }
  if (d_workItems) {
      cudaFree(d_workItems);
      d_workItems = nullptr;
  }
  if (d_analysisResults) {
      cudaFree(d_analysisResults);
      d_analysisResults = nullptr;
  }
  if (d_brickWorkList) {
      cudaFree(d_brickWorkList);
      d_brickWorkList = nullptr;
  }
}

void CudaMaterialMap::WorldToWrapped(int wx, int wy, int wz, int &ix, int &iy, int &iz) const {
  ix = posmod(wx, _indW);
  iy = posmod(wy, _indH);
  iz = posmod(wz, _indD);
}

int CudaMaterialMap::WrappedToLinear(int ix, int iy, int iz) const {
  return ix + iy * _indW + iz * _indW * _indH;
}

uint32_t CudaMaterialMap::AllocSectorHandle() {
  if (!_freeSectorHandles.empty()) {
    uint32_t h = _freeSectorHandles.back();
    _freeSectorHandles.pop_back();
    return h;
  }
  if (_nextSectorHandle >= MAX_ACTIVE_SECTORS) {
    fprintf(stderr, "[CudaMaterialMap] ERROR: ran out of sector handles!\n");
    return 0;
  }
  return _nextSectorHandle++;
}

void CudaMaterialMap::FreeSectorHandle(uint32_t handle) {
  if (handle > 0) {
    _freeSectorHandles.push_back(handle);
  }
}

void CudaMaterialMap::UploadIndirectionCell(int ix, int iy, int iz, uint32_t value) {
  int idx = WrappedToLinear(ix, iy, iz);
  _indirectionCPU[idx] = value;
}

void CudaMaterialMap::UploadSectorInfo(uint32_t handle, const SectorInfo &info) {
  _sectorInfoCPU[handle] = info;
  
  // Upload to GPU
  cudaError_t err = cudaMemcpy(d_sectors + handle, &info, sizeof(SectorInfo), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "[CudaMaterialMap] ERROR: Failed to upload sector info: %s\n", cudaGetErrorString(err));
  }
}

void CudaMaterialMap::UnloadSector(int ix, int iy, int iz) {
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

void CudaMaterialMap::LoadSector(int wx, int wy, int wz, bool isLOD,
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
  float3 sectorWorldPos = make_float3((float)(wx * 32), (float)(wy * 32), (float)(wz * 32));

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
      if (Evaluate(brickPos.x + 4.0f, brickPos.y + 4.0f, brickPos.z + 4.0f) > 0.0f) {
        brickMask |= (1ULL << b);
      }
    } else {
      // Detail: Robust heuristic check
      bool active = false;
      for (int dz = 0; dz < 8 && !active; dz += 3) {
        for (int dy = 0; dy < 8 && !active; dy += 3) {
          for (int dx = 0; dx < 8 && !active; dx += 3) {
            if (Evaluate(brickPos.x + dx, brickPos.y + dy, brickPos.z + dz) > 0.0f) {
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
  uint32_t activeBricks = host_popcount64(brickMask);

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
      fprintf(stderr, "[CudaMaterialMap] Brick pool full! Falling back to LOD for sector (%d,%d,%d)\n",
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

void CudaMaterialMap::RebuildSectorMasks() {
  int superX = (_indW + 3) / 4;
  int superY = (_indH + 3) / 4;
  int superZ = (_indD + 3) / 4;
  int totalSuper = superX * superY * superZ;

  std::vector<uint64_t> masks(totalSuper, 0);

  for (int iy = 0; iy < _indH; iy++) {
    for (int iz = 0; iz < _indD; iz++) {
      for (int ix = 0; ix < _indW; ix++) {
        int idx = WrappedToLinear(ix, iy, iz);
        if (_sectorStates[idx].isLoaded && _sectorStates[idx].sectorHandle != 0) {
          int superIdx = (ix / 4) + (iz / 4) * superX + (iy / 4) * superX * superZ;
          int lx = ix & 3, ly = iy & 3, lz = iz & 3;
          int bitIdx = lx + (lz << 2) + (ly << 4);
          masks[superIdx] |= (1ULL << bitIdx);
        }
      }
    }
  }

  // Upload to GPU
  cudaError_t err = cudaMemcpy(d_sectorMasks, masks.data(), 
                               totalSuper * sizeof(uint64_t), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "[CudaMaterialMap] ERROR: Failed to upload sector masks: %s\n", cudaGetErrorString(err));
  }
}

bool CudaMaterialMap::UpdateStreaming(simd_float3 cameraPos) {
  // 1. Process Completed Async Analyses
  std::vector<AsyncResult> completedResults;
  {
    std::lock_guard<std::mutex> lock(_asyncResultsMutex);
    completedResults = std::move(_asyncResults);
    _asyncResults.clear();
  }

  std::vector<BrickWorkItem> workList;
  bool anyChanged = false;

  // Process completed async results (similar to Metal implementation)
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
    if (handle == 0) continue;

    uint32_t activeBricks = host_popcount64(brickMask);
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
        fprintf(stderr, "[CudaMaterialMap] Brick pool full! LOD fallback (%d,%d,%d)\n",
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
  int3 camSector = make_int3(camSX, camSY, camSZ);

  _worldOrigin = make_int3(camSX - _indW / 2, 0, camSZ - _indD / 2);

  std::vector<SectorWorkItem> pendingRequests;
  int unloaded = 0;

  for (int dy = 0; dy < _indH; dy++) {
    for (int dz = 0; dz < _indD; dz++) {
      for (int dx = 0; dx < _indW; dx++) {
        int expectedWX = _worldOrigin.x + dx;
        int expectedWY = dy;
        int expectedWZ = _worldOrigin.z + dz;

        if (expectedWY < 0 || expectedWY >= _indH) continue;

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

  // 3. Dispatch Async Analysis to GPU
  if (!pendingRequests.empty()) {
      // Sort pendingRequests by 3D distance to camera so closer chunks load first
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

      size_t numReq = std::min(pendingRequests.size(), _workItemCapacity);
      
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
          state.isAnalyzing = true;
      }
      
      std::vector<SectorWorkItem> dispatchedItems(pendingRequests.begin(), pendingRequests.begin() + numReq);
      DispatchAsyncAnalysis(dispatchedItems);
  }

  // 4. Update Render State
  if (anyChanged || unloaded > 0) {
    RebuildSectorMasks();

    // Upload indirection buffer to GPU
    size_t indirectionSize = _indW * _indH * _indD * sizeof(uint32_t);
    cudaError_t err = cudaMemcpy(d_indirection, _indirectionCPU.data(), 
                                 indirectionSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf(stderr, "[CudaMaterialMap] ERROR: Failed to upload indirection: %s\n", cudaGetErrorString(err));
    }

    // Generate brick data on GPU if needed
    if (!workList.empty()) {
      GenerateBrickData(workList);
    }
  }

  if (_firstUpdate) {
    _firstUpdate = false;
  }

  _lastCameraSector = camSector;

  return anyChanged;
}

void CudaMaterialMap::GenerateDynamic() {
  printf("[CudaMaterialMap] Starting initial streaming world generation...\n");

  // For initial generation, we need a spawn position
  // Use origin as default spawn
  simd_float3 spawnPos = make_float3(0.0f, 0.0f, 0.0f);

  UpdateStreaming(spawnPos);

  printf("[CudaMaterialMap] Initial world generation complete.\n");
}

void CudaMaterialMap::GenerateBrickData(const std::vector<BrickWorkItem> &workList) {
  if (workList.empty()) return;
  
  if (workList.size() > _brickWorkCapacity) {
      // Reallocate if needed
      if (d_brickWorkList) cudaFree(d_brickWorkList);
      _brickWorkCapacity = workList.size() + 1024;
      cudaMalloc(&d_brickWorkList, _brickWorkCapacity * sizeof(BrickWorkItem));
  }
  
  cudaError_t err = cudaMemcpyAsync(d_brickWorkList, workList.data(), 
                                    workList.size() * sizeof(BrickWorkItem), 
                                    cudaMemcpyHostToDevice, (cudaStream_t)_cudaStream);
  if (err != cudaSuccess) {
      fprintf(stderr, "[CudaMaterialMap] ERROR: Failed to upload worklist: %s\n", cudaGetErrorString(err));
      return;
  }
  
  uint32_t totalSubBricks = workList.size() * 8;
  dim3 gridSize(totalSubBricks, 1, 1);
  dim3 blockSize(64, 1, 1);
  
  XMap_FillBricks<<<gridSize, blockSize, 0, (cudaStream_t)_cudaStream>>>(
      d_brickWorkList,
      d_sectors,
      _brickPool.GetOccupancyPtr(),
      _brickPool.GetDataPtr(),
      _worldOrigin,
      0, 0
  );
  
  cudaStreamSynchronize((cudaStream_t)_cudaStream);
}

void CudaMaterialMap::DispatchAsyncAnalysis(const std::vector<SectorWorkItem> &items) {
  if (items.empty()) return;
  
  size_t numReq = items.size();
  
  cudaError_t err = cudaMemcpyAsync(d_workItems, items.data(), numReq * sizeof(SectorWorkItem), cudaMemcpyHostToDevice, (cudaStream_t)_cudaStream);
  if (err != cudaSuccess) {
      fprintf(stderr, "[CudaMaterialMap] ERROR: Failed to upload items: %s\n", cudaGetErrorString(err));
      return;
  }
  
  uint32_t totalItems = (uint32_t)numReq;
  dim3 gridSize(numReq, 1, 1);
  dim3 blockSize(std::min((uint32_t)64, totalItems), 1, 1);
  
  XMap_AnalyzeStreaming<<<gridSize, blockSize, 0, (cudaStream_t)_cudaStream>>>(
      d_workItems,
      d_analysisResults,
      totalItems,
      0, 0
  );
  
  std::vector<uint64_t> resultsMasks(numReq);
  cudaMemcpyAsync(resultsMasks.data(), d_analysisResults, numReq * sizeof(uint64_t), cudaMemcpyDeviceToHost, (cudaStream_t)_cudaStream);
  
  // Wait for results
  cudaStreamSynchronize((cudaStream_t)_cudaStream);
  
  std::vector<AsyncResult> results;
  results.reserve(numReq);
  for (size_t i = 0; i < numReq; ++i) {
      AsyncResult res;
      res.item = items[i];
      res.brickMask = resultsMasks[i];
      results.push_back(res);
  }
  
  std::lock_guard<std::mutex> lock(_asyncResultsMutex);
  _asyncResults.insert(_asyncResults.end(), results.begin(), results.end());
}

void CudaMaterialMap::ProcessAsyncResults(std::vector<BrickWorkItem> &workList) {
  (void)workList;
  // Results are processed in UpdateStreaming
}
