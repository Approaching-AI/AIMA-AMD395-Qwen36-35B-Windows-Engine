// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace aima_port {
inline std::map<std::string, std::string> probe_arguments(
    const std::vector<std::string>& arguments) {
  const std::vector<std::string> required = {
      "--model", "--input-u32", "--ck-provider", "--vision-image", "--load-report"};
  if (arguments.size() != required.size() * 2) {
    throw std::invalid_argument(
        "Expected --model DIR --input-u32 FILE --ck-provider DLL "
        "--vision-image HSACO --load-report NEW_FILE; fixed q8192/out512");
  }
  std::map<std::string, std::string> result;
  for (std::size_t i = 0; i < arguments.size(); i += 2) {
    const auto& key = arguments[i];
    bool known = false;
    for (const auto& option : required) known = known || option == key;
    if (!known || arguments[i + 1].empty() ||
        arguments[i + 1].find('\0') != std::string::npos ||
        !result.emplace(key, arguments[i + 1]).second) {
      throw std::invalid_argument("Unknown, empty or repeated argument: " + key);
    }
  }
  return result;
}

inline std::vector<std::uint32_t> read_prompt(const std::filesystem::path& path) {
  constexpr std::size_t count = 8192;
  if (std::filesystem::file_size(path) != count * 4) {
    throw std::invalid_argument("Prompt must contain exactly 8192 little-endian uint32 IDs");
  }
  std::ifstream stream(path, std::ios::binary);
  std::vector<unsigned char> bytes(count * 4);
  stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!stream || stream.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("Prompt read failed or input changed");
  }
  std::vector<std::uint32_t> tokens(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto* p = bytes.data() + i * 4;
    const std::uint32_t id = std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
        (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
    if (id >= 248320) throw std::invalid_argument("Prompt token exceeds model vocabulary");
    tokens[i] = id;
  }
  return tokens;
}

inline std::string json_string(const std::string& text) {
  static const char* hex = "0123456789abcdef";
  std::string output = "\"";
  for (unsigned char c : text) {
    if (c == '\\' || c == '"') { output += '\\'; output += c; }
    else if (c < 32) {
      output += "\\u00"; output += hex[c >> 4]; output += hex[c & 15];
    } else output += c;
  }
  return output + '"';
}
}  // namespace aima_port
