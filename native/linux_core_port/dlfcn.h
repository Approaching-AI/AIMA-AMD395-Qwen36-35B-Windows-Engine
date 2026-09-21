// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <string>
#include <utility>

inline constexpr int RTLD_NOW = 2;
inline constexpr int RTLD_LOCAL = 0;
namespace aima_port {
inline thread_local std::string dynamic_library_error;
inline thread_local std::string returned_dynamic_library_error;
inline void remember_library_error(const char* operation) {
  const DWORD error = GetLastError();
  dynamic_library_error = std::string(operation) + ": Windows error " +
                          std::to_string(error);
}
}

inline void* dlopen(const wchar_t* path, int) {
  const HMODULE module = LoadLibraryW(path);
  if (!module) aima_port::remember_library_error("LoadLibraryW");
  return reinterpret_cast<void*>(module);
}
inline void* dlsym(void* handle, const char* symbol) {
  const FARPROC value = GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol);
  if (!value) aima_port::remember_library_error("GetProcAddress");
  return reinterpret_cast<void*>(value);
}
inline int dlclose(void* handle) {
  if (FreeLibrary(reinterpret_cast<HMODULE>(handle))) return 0;
  aima_port::remember_library_error("FreeLibrary");
  return -1;
}
inline const char* dlerror() {
  if (aima_port::dynamic_library_error.empty()) return nullptr;
  aima_port::returned_dynamic_library_error =
      std::move(aima_port::dynamic_library_error);
  aima_port::dynamic_library_error.clear();
  return aima_port::returned_dynamic_library_error.c_str();
}
#else
#include_next <dlfcn.h>
#endif
