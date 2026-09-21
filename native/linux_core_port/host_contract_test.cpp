// SPDX-License-Identifier: Apache-2.0
#include "file_io.h"
#include "probe_input.h"
#include <array>
#include <iostream>
#include <limits>
#ifdef _WIN32
#include <winioctl.h>
#endif

namespace {
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template <class F> void rejects(F operation) {
  bool rejected = false;
  try { operation(); } catch (const std::exception&) { rejected = true; }
  require(rejected, "Malformed input accepted");
}
}

int main(int argc, char** argv) {
  try {
    require(argc == 2, "Requires an empty evidence directory");
    const std::filesystem::path directory(argv[1]);
    require(!std::filesystem::exists(directory), "Evidence already exists");
    std::filesystem::create_directories(directory);
    const auto path = directory / std::filesystem::u8path(u8"offset-中文.bin");
    constexpr std::uint64_t offset = (1ULL << 32) + 17;
    const std::string expected = "64-bit-shard-offset";
    {
#ifdef _WIN32
      const auto fixture = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr,
                                      CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
      require(fixture != INVALID_HANDLE_VALUE, "Sparse shard fixture create failed");
      DWORD amount = 0;
      LARGE_INTEGER position{};
      position.QuadPart = offset;
      const bool written = DeviceIoControl(fixture, FSCTL_SET_SPARSE, nullptr, 0,
                                           nullptr, 0, &amount, nullptr) != 0 &&
          SetFilePointerEx(fixture, position, nullptr, FILE_BEGIN) != 0 &&
          WriteFile(fixture, expected.data(), static_cast<DWORD>(expected.size()),
                    &amount, nullptr) != 0 && amount == expected.size();
      CloseHandle(fixture);
      require(written, "64-bit sparse fixture write failed");
#else
      std::ofstream output(path, std::ios::binary);
      output.seekp(offset);
      output.write(expected.data(), expected.size());
      require(bool(output), "Sparse shard fixture write failed");
#endif
    }
    require(aima_port::file_size_matches(path.u8string().c_str(), offset + expected.size()),
            "64-bit file size or UTF-8 path failed");
    require(!aima_port::file_size_matches(path.u8string().c_str(), expected.size()),
            "Truncated file size accepted");
    const auto handle = aima_port::open_file(path.u8string().c_str(), false);
    require(handle != aima_port::invalid_file, "Cannot open sparse fixture");
    std::array<char, 64> buffer{};
    require(aima_port::read_file_at(handle, buffer.data(), buffer.size(), offset) ==
                static_cast<std::int64_t>(expected.size()) &&
                std::string(buffer.data(), expected.size()) == expected,
            "64-bit short read changed payload");
    require(aima_port::read_file_at(handle, buffer.data(), 1, offset + expected.size()) == 0,
            "EOF did not return zero");
    errno = 0;
    require(aima_port::read_file_at(handle, buffer.data(), 1, UINT64_MAX) == -1 && errno == EINVAL,
            "Overflowing read offset accepted");
    aima_port::close_file(handle);
    require(aima_port::open_file((directory / "absent").u8string().c_str(), false) ==
                aima_port::invalid_file, "Missing shard accepted");
    const auto prompt = directory / "prompt.u32";
    std::vector<unsigned char> bytes(8192 * 4);
    for (unsigned i = 0; i < 8192; ++i) {
      const unsigned id = (i * 79) % 248320;
      for (unsigned b = 0; b < 4; ++b) bytes[4 * i + b] = (id >> (8 * b)) & 255;
    }
    auto write_prompt = [&] {
      std::ofstream output(prompt, std::ios::binary | std::ios::trunc);
      output.write(reinterpret_cast<char*>(bytes.data()), bytes.size());
      require(bool(output), "Prompt fixture write failed");
    };
    write_prompt();
    const auto tokens = aima_port::read_prompt(prompt);
    for (unsigned i = 0; i < 8192; ++i)
      require(tokens[i] == (i * 79) % 248320, "Little-endian token decoding failed");
    bytes[0] = bytes[1] = bytes[2] = bytes[3] = 255;
    write_prompt(); rejects([&] { aima_port::read_prompt(prompt); });
    bytes.pop_back(); write_prompt(); rejects([&] { aima_port::read_prompt(prompt); });
    const std::vector<std::string> arguments = {"--model", "m", "--input-u32", "i",
        "--ck-provider", "c", "--vision-image", "v", "--load-report", "r"};
    require(aima_port::probe_arguments(arguments).size() == 5, "Valid argument set rejected");
    auto malformed = arguments; malformed[0] = "--unknown";
    rejects([&] { aima_port::probe_arguments(malformed); });
    malformed = arguments; malformed[0] = "--input-u32";
    rejects([&] { aima_port::probe_arguments(malformed); });
    malformed = arguments; malformed[1].clear();
    rejects([&] { aima_port::probe_arguments(malformed); });
    malformed = arguments; malformed.pop_back();
    rejects([&] { aima_port::probe_arguments(malformed); });
    require(aima_port::json_string(std::string("a\"\\\n\0", 5)) == "\"a\\\"\\\\\\u000a\\u0000\"",
            "JSON escaping failed");
    // Only remove the huge logical sparse fixture created by this test.
    std::filesystem::remove(path);
    std::cout << "{\"passed\":true,\"large_offset\":" << offset
              << ",\"tokens_checked\":8192,\"model_loaded\":false}" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
