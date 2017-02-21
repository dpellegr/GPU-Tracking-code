#pragma once

#define CLGLOBAL
#define _GPUCODE
#define __CUDA_HOST_DEVICE__ __host__ __device__
#include "track.h"

//need c++17 (in NVRTC) to avoid this macro without going to function pointers...
#define Element(X) struct Element {                       \
  X##_data data;                                                 \
  template <typename... Args>                             \
  __host__ __device__                                     \
  Element(Args... args): data( X##_init(args...) ) {}     \
  __host__ __device__                                     \
  void operator()(Particle & p) { X##_track(&p, &data); } \
}

// c++17 prototype with auto non-type template parameters:
// using cLinMap = Element<LinMap, decltype(LinMap_init), decltype(LinMap_track)>;

using LinMap = Element(LinMap);

#undef Element

