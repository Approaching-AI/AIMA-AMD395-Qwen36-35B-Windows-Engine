// SPDX-License-Identifier: Apache-2.0
#pragma once

// Each scatter worker owns its handle. Windows reads therefore need no shared
// seek lock; offsets, file sizes and return values stay 64-bit on both hosts.
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace aima_port {
#ifdef _WIN32
using file_handle = HANDLE;
inline const file_handle invalid_file = INVALID_HANDLE_VALUE;

inline void file_error(DWORD error) {
  switch (error) {
    case ERROR_INVALID_PARAMETER: errno = EINVAL; break;
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND: errno = ENOENT; break;
    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION: errno = EACCES; break;
    default: errno = EIO; break;
  }
}

inline file_handle open_file(const char* utf8_path, bool direct) {
  const auto path = std::filesystem::u8path(utf8_path);
  const DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN |
                      (direct ? FILE_FLAG_NO_BUFFERING : 0);
  const auto handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                  nullptr, OPEN_EXISTING, flags, nullptr);
  if (handle == invalid_file) file_error(GetLastError());
  return handle;
}

inline void close_file(file_handle handle) {
  if (handle != invalid_file) (void)CloseHandle(handle);
}

inline bool file_size_matches(const char* path, std::uint64_t expected) {
  const auto handle = open_file(path, false);
  if (handle == invalid_file) return false;
  LARGE_INTEGER size{};
  const bool ok = GetFileSizeEx(handle, &size) != 0;
  if (!ok) file_error(GetLastError());
  close_file(handle);
  return ok && size.QuadPart >= 0 &&
         static_cast<std::uint64_t>(size.QuadPart) == expected;
}

inline std::int64_t read_file_at(file_handle handle, void* buffer,
                                 std::size_t bytes, std::uint64_t offset) {
  // Loader chunks are 128 MiB. Refuse oversized requests instead of silently
  // narrowing the Win32 DWORD or signed file offset.
  if (bytes > std::numeric_limits<DWORD>::max() ||
      offset > static_cast<std::uint64_t>(INT64_MAX)) {
    errno = EINVAL;
    return -1;
  }
  LARGE_INTEGER position{};
  position.QuadPart = static_cast<LONGLONG>(offset);
  if (!SetFilePointerEx(handle, position, nullptr, FILE_BEGIN)) {
    file_error(GetLastError());
    return -1;
  }
  DWORD amount = 0;
  if (!ReadFile(handle, buffer, static_cast<DWORD>(bytes), &amount, nullptr)) {
    const DWORD error = GetLastError();
    if (error == ERROR_HANDLE_EOF) return 0;
    file_error(error);
    return -1;
  }
  return amount;
}

// FILE_FLAG_SEQUENTIAL_SCAN gives the OS the buffered fallback's access hint.
// There is no process-local Windows equivalent of POSIX_FADV_DONTNEED.
inline void drop_file_cache(file_handle, std::uint64_t, std::uint64_t) {}
#else
using file_handle = int;
inline constexpr file_handle invalid_file = -1;

inline file_handle open_file(const char* path, bool direct) {
  int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_DIRECT
  if (direct) flags |= O_DIRECT;
#else
  if (direct) { errno = EINVAL; return invalid_file; }
#endif
  return ::open(path, flags);
}

inline void close_file(file_handle handle) {
  if (handle != invalid_file) (void)::close(handle);
}

inline bool file_size_matches(const char* path, std::uint64_t expected) {
  struct stat status{};
  return ::stat(path, &status) == 0 && status.st_size >= 0 &&
         static_cast<std::uint64_t>(status.st_size) == expected;
}

inline std::int64_t read_file_at(file_handle handle, void* buffer,
                                 std::size_t bytes, std::uint64_t offset) {
  if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
      bytes > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
    errno = EINVAL;
    return -1;
  }
  ssize_t result;
  do {
    result = ::pread(handle, buffer, bytes, static_cast<off_t>(offset));
  } while (result < 0 && errno == EINTR);
  return result;
}

inline void drop_file_cache(file_handle handle, std::uint64_t offset,
                            std::uint64_t bytes) {
#ifdef POSIX_FADV_DONTNEED
  (void)::posix_fadvise(handle, static_cast<off_t>(offset),
                       static_cast<off_t>(bytes), POSIX_FADV_DONTNEED);
#else
  (void)handle; (void)offset; (void)bytes;
#endif
}
#endif
}  // namespace aima_port
