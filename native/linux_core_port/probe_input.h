// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace aima_port {
inline std::size_t observation_number(const std::string& value, std::size_t maximum) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos ||
      value.size() > 3 || (value.size() > 1 && value[0] == '0')) {
    throw std::invalid_argument("Invalid observation selector");
  }
  const auto number = static_cast<std::size_t>(std::stoul(value));
  if (number > maximum) throw std::invalid_argument("Observation selector exceeds bound");
  return number;
}

inline std::optional<std::size_t> full_attention_observation_layer(
    const char* setting, bool has_directory, bool first64) {
  if (setting == nullptr) return std::nullopt;
  const auto layer = observation_number(setting, 39);
  if (layer % 4 != 3 || !has_directory || first64)
    throw std::invalid_argument("Full-attention observation requires a full layer, output directory and no first64 capture");
  return layer;
}

inline std::map<std::string, std::string> probe_arguments(
    const std::vector<std::string>& arguments) {
  const std::vector<std::string> required = {
      "--model", "--input-u32", "--ck-provider", "--vision-image", "--load-report"};
  const std::vector<std::string> optional = {
      "--observe-directory", "--observe-output-index", "--observe-linear-layer"};
  if (arguments.size() != required.size() * 2 &&
      arguments.size() != (required.size() + optional.size()) * 2) {
    throw std::invalid_argument(
        "Expected --model DIR --input-u32 FILE --ck-provider DLL "
        "--vision-image HSACO --load-report NEW_FILE; fixed q8192/out512");
  }
  std::map<std::string, std::string> result;
  for (std::size_t i = 0; i < arguments.size(); i += 2) {
    const auto& key = arguments[i];
    bool known = false;
    for (const auto& option : required) known = known || option == key;
    for (const auto& option : optional) known = known || option == key;
    if (!known || arguments[i + 1].empty() ||
        arguments[i + 1].find('\0') != std::string::npos ||
        !result.emplace(key, arguments[i + 1]).second) {
      throw std::invalid_argument("Unknown, empty or repeated argument: " + key);
    }
  }
  for (const auto& option : required) {
    if (!result.count(option)) throw std::invalid_argument("Missing argument: " + option);
  }
  std::size_t optional_count = 0;
  for (const auto& option : optional) optional_count += result.count(option);
  if (optional_count != 0 && optional_count != optional.size()) {
    throw std::invalid_argument("Observation requires directory, output index and linear layer");
  }
  if (optional_count) {
    if (observation_number(result.at("--observe-output-index"), 511) == 0 ||
        observation_number(result.at("--observe-linear-layer"), 39) % 4 == 3) {
      throw std::invalid_argument("Observation requires decode output 1..511 and a linear layer");
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
