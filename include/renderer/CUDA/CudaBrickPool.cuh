#pragma once

#include "renderer/BrickPool.hpp"

using CudaBrickPool = BrickPool<BrickPoolImpl::CudaBrickPoolTraits>;
