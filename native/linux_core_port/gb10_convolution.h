// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <hip/hip_runtime.h>
#include "gb10_convolution_math.h"
#include "aima/sha256.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace aima_port {
inline unsigned char* convolution_silu_device = nullptr;
inline constexpr char convolution_silu_sha[] =
    "673f8dd1280700578c1e8743afd2e3b4da134b1fbd463c890527e1c4d9f796b8";

class ConvolutionSiluOwner {
 public:
  ConvolutionSiluOwner() {
    const char* selected = std::getenv("AIMA_PORT_SILU_TABLE");
    if (!selected || !*selected || convolution_silu_device)
      throw std::runtime_error("Convolution SiLU table ownership is invalid");
    const auto path = std::filesystem::u8path(selected);
    if (std::filesystem::file_size(path) != qrt_sm121_silu::table_bytes)
      throw std::runtime_error("Convolution SiLU table size differs");
    std::vector<unsigned char> data(qrt_sm121_silu::table_bytes);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(data.data()), data.size());
    if (!input || input.peek() != std::char_traits<char>::eof() ||
        aima::sha256_bytes(data.data(), data.size()) != convolution_silu_sha ||
        !qrt_sm121_silu::valid_layout(data.data(), data.size()))
      throw std::runtime_error("Convolution SiLU table identity or layout differs");
    if (hipSetDevice(0) != hipSuccess ||
        hipMalloc(reinterpret_cast<void**>(&allocation_), data.size()) != hipSuccess)
      throw std::runtime_error("Convolution SiLU table allocation failed");
    if (hipMemcpy(allocation_, data.data(), data.size(), hipMemcpyHostToDevice) != hipSuccess) {
      hipFree(allocation_); allocation_ = nullptr;
      throw std::runtime_error("Convolution SiLU table upload failed");
    }
    convolution_silu_device = allocation_;
  }
  ConvolutionSiluOwner(const ConvolutionSiluOwner&) = delete;
  ConvolutionSiluOwner& operator=(const ConvolutionSiluOwner&) = delete;
  ~ConvolutionSiluOwner() {
    convolution_silu_device = nullptr;
    if (allocation_) hipFree(allocation_);
  }
 private:
  unsigned char* allocation_ = nullptr;
};

static __global__ void gb10_decode_convolution_kernel(
    std::uint16_t* qkv, const std::uint16_t* weights,
    std::uint16_t* history, const unsigned char* silu) {
  const unsigned feature = blockIdx.x * blockDim.x + threadIdx.x;
  if (feature >= 8192) return;
  const auto x0 = history[feature * 3], x1 = history[feature * 3 + 1];
  const auto x2 = history[feature * 3 + 2], x3 = qkv[feature];
  qkv[feature] = conv_value(x0, x1, x2, x3, weights + feature * 4, silu);
  history[feature * 3] = x1;
  history[feature * 3 + 1] = x2;
  history[feature * 3 + 2] = x3;
}

static __global__ void gb10_prefill_convolution_kernel(
    const std::uint16_t* qkv, const std::uint16_t* weights,
    const std::uint16_t* initial, std::uint16_t* output,
    unsigned tokens, bool has_initial, const unsigned char* silu) {
  const unsigned feature = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned begin = blockIdx.y * 32;
  if (feature >= 8192 || begin >= tokens) return;
  std::uint16_t previous[3];
  for (unsigned i = 0; i < 3; ++i) {
    previous[i] = begin + i >= 3 ? qkv[(begin + i - 3) * 8192 + feature]
        : has_initial ? initial[feature * 3 + begin + i] : 0;
  }
  for (unsigned token = begin; token < begin + 32 && token < tokens; ++token) {
    const auto current = qkv[token * 8192 + feature];
    output[token * 8192 + feature] = conv_value(
        previous[0], previous[1], previous[2], current, weights + feature * 4, silu);
    previous[0] = previous[1]; previous[1] = previous[2]; previous[2] = current;
  }
}

static __global__ void gb10_prefill_history_kernel(
    const std::uint16_t* qkv, std::uint16_t* history, unsigned tokens) {
  const unsigned feature = blockIdx.x * blockDim.x + threadIdx.x;
  if (feature >= 8192) return;
  for (unsigned i = 0; i < 3; ++i)
    history[feature * 3 + i] = qkv[(tokens - 3 + i) * 8192 + feature];
}

inline void gb10_decode_convolution(void* qkv, const void* weights, void* history,
                                    hipStream_t stream) {
  if (!qkv || !weights || !history || !convolution_silu_device)
    throw std::runtime_error("Decode convolution binding is incomplete");
  hipLaunchKernelGGL(gb10_decode_convolution_kernel, dim3(32), dim3(256), 0, stream,
      static_cast<std::uint16_t*>(qkv), static_cast<const std::uint16_t*>(weights),
      static_cast<std::uint16_t*>(history), convolution_silu_device);
  if (hipGetLastError() != hipSuccess) throw std::runtime_error("Decode convolution launch failed");
}

inline void gb10_prefill_convolution(const void* qkv, const void* weights,
    void* history, void* output, std::size_t tokens, bool has_initial) {
  // This experiment covers the canonical q8192 owner. Keep other schedules
  // outside its claim until their state and padding boundaries are tested.
  if (!qkv || !weights || !history || !output || qkv == output || tokens != 8192 ||
      !convolution_silu_device)
    throw std::runtime_error("Prefill convolution binding or shape is unsupported");
  hipLaunchKernelGGL(gb10_prefill_convolution_kernel, dim3(32, (tokens + 31) / 32),
      dim3(256), 0, nullptr, static_cast<const std::uint16_t*>(qkv),
      static_cast<const std::uint16_t*>(weights), static_cast<const std::uint16_t*>(history),
      static_cast<std::uint16_t*>(output), static_cast<unsigned>(tokens), has_initial,
      convolution_silu_device);
  if (hipGetLastError() != hipSuccess) throw std::runtime_error("Prefill convolution launch failed");
  // Commit only after all consumers finish reading the initial history.
  hipLaunchKernelGGL(gb10_prefill_history_kernel, dim3(32), dim3(256), 0, nullptr,
      static_cast<const std::uint16_t*>(qkv), static_cast<std::uint16_t*>(history),
      static_cast<unsigned>(tokens));
  if (hipGetLastError() != hipSuccess) throw std::runtime_error("Prefill history launch failed");
}
}  // namespace aima_port
