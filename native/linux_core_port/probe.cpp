// SPDX-License-Identifier: Apache-2.0
#include "aima/native_resident_engine.h"
#include "aima/sha256.h"
#include "aima_port_build_identity.h"
#include "probe_input.h"
#include <hip/hip_runtime.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#ifdef AIMA_PORT_GB10_CONVOLUTION
#include "gb10_convolution.h"
#endif
#ifdef AIMA_PORT_GB10_GDN
#include "gb10_gdn.h"
#endif
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::string host_name() {
#ifdef _WIN32
  wchar_t name[MAX_COMPUTERNAME_LENGTH + 1]{};
  DWORD size = MAX_COMPUTERNAME_LENGTH + 1;
  if (!GetComputerNameW(name, &size)) throw std::runtime_error("Cannot identify host");
  return std::filesystem::path(name).u8string();
#else
  char name[256]{};
  if (gethostname(name, sizeof(name) - 1) != 0) throw std::runtime_error("Cannot identify host");
  return name;
#endif
}

class Observation {
 public:
  Observation(const std::filesystem::path& directory, std::size_t index,
              std::size_t layer) : directory_(directory), output_index_(index), layer_(layer) {
    if (!std::filesystem::create_directory(directory_))
      throw std::runtime_error("Observation directory already exists");
    manifest_.open(directory_ / "manifest.jsonl", std::ios::binary);
    if (!manifest_) throw std::runtime_error("Cannot create observation manifest");
  }
  void capture(const std::string& name, const void* device, std::uint64_t bytes,
               const char* dtype) {
    if (name.empty() || name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-_") !=
                            std::string::npos ||
        device == nullptr || bytes == 0 || bytes > (8ULL << 20) ||
        bytes > (32ULL << 20) - total_ || count_ >= 128) {
      throw std::runtime_error("Observation name, pointer or byte extent is invalid");
    }
    const auto file = name + ".bin";
    const auto path = directory_ / file;
    if (std::filesystem::exists(path)) throw std::runtime_error("Duplicate observation file");
    std::vector<unsigned char> host(static_cast<std::size_t>(bytes));
    if (hipDeviceSynchronize() != hipSuccess ||
        hipMemcpy(host.data(), device, host.size(), hipMemcpyDeviceToHost) != hipSuccess)
      throw std::runtime_error("Observation device-to-host copy failed");
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(host.data()), host.size());
    stream.close();
    if (!stream) throw std::runtime_error("Observation write failed");
    const auto digest = aima::sha256_bytes(host.data(), host.size());
    manifest_ << "{\"file\":" << aima_port::json_string(file)
              << ",\"bytes\":" << bytes << ",\"dtype\":" << aima_port::json_string(dtype)
              << ",\"sha256\":" << aima_port::json_string(digest)
              << ",\"selected_decode_output_index\":" << output_index_ << ",\"linear_layer\":" << layer_
              << ",\"output_only\":true}" << std::endl;
    if (!manifest_) throw std::runtime_error("Observation manifest write failed");
    total_ += bytes; ++count_;
  }
  void bind(aima::NativeResidentRequestOptions& request) {
    request.decode_layer_observer_output_index = output_index_;
    request.decode_linear_observer_layer_index = layer_;
    request.prefill_linear_state_observer = [this](std::size_t layer, const void* conv,
        std::uint64_t conv_bytes, const void* state, std::uint64_t state_bytes) {
      if (layer != layer_) return;
      capture("prefill-conv", conv, conv_bytes, "bf16");
      capture("prefill-state", state, state_bytes, "f32");
    };
    request.decode_layer_observer = [this](std::size_t boundary, const void* row) {
      if (boundary > 40) throw std::runtime_error("Invalid decode layer boundary");
      capture("decode-boundary-" + std::to_string(boundary), row, 2048 * 2, "bf16");
    };
    auto stage = [this](const char* name, const void* device, std::uint64_t bytes,
                       aima::DecodeTensorDtype dtype) {
      const char* type = dtype == aima::DecodeTensorDtype::kBfloat16 ? "bf16" :
                         dtype == aima::DecodeTensorDtype::kFloat32 ? "f32" :
                         dtype == aima::DecodeTensorDtype::kInt32 ? "i32" : nullptr;
      if (type == nullptr || name == nullptr) throw std::runtime_error("Unsupported observation dtype");
      capture(std::string("decode-linear-") + name, device, bytes, type);
    };
    request.decode_linear_layer0_observer = stage;
    request.decode_layer0_tail_observer = stage;
  }
 private:
  std::filesystem::path directory_;
  std::ofstream manifest_;
  std::size_t output_index_, layer_, count_ = 0;
  std::uint64_t total_ = 0;
};

int run(const std::vector<std::string>& argv) {
  const auto entered = Clock::now();
  std::cout.imbue(std::locale::classic());
  std::cout << std::setprecision(17);
  const auto args = aima_port::probe_arguments(argv);
  const auto path = [&](const char* key) { return std::filesystem::u8path(args.at(key)); };
  const auto model = std::filesystem::absolute(path("--model"));
  const auto input = std::filesystem::absolute(path("--input-u32"));
  const auto ck = std::filesystem::absolute(path("--ck-provider"));
  const auto vision = std::filesystem::absolute(path("--vision-image"));
  const auto report = std::filesystem::absolute(path("--load-report"));
  if (std::filesystem::exists(report)) throw std::invalid_argument("Load evidence already exists");
  for (const char* label : {"language", "visual"}) {
    const auto extension = report.extension().empty() ? ".json" : report.extension().u8string();
    const auto split = report.parent_path() / std::filesystem::u8path(
        report.stem().u8string() + "." + label + extension);
    if (std::filesystem::exists(split)) throw std::invalid_argument("Split load evidence already exists");
  }
  aima::NativeResidentRequestOptions request;
  request.input_token_ids = aima_port::read_prompt(input);
  request.max_new_tokens = 512;
  request.temperature = 0.0;
  request.disable_prefix_cache = true;
  std::unique_ptr<Observation> observation;
  if (args.count("--observe-directory")) {
    observation = std::make_unique<Observation>(
        std::filesystem::absolute(path("--observe-directory")),
        aima_port::observation_number(args.at("--observe-output-index"), 511),
        aima_port::observation_number(args.at("--observe-linear-layer"), 39));
    observation->bind(request);
  }
  const auto quote = aima_port::json_string;
  const auto prompt_sha = aima::sha256_bytes(request.input_token_ids.data(),
                                             request.input_token_ids.size() * sizeof(std::uint32_t));
  if (prompt_sha != aima::sha256_file(input)) throw std::runtime_error("Prompt changed after read");
  const auto provider_sha = aima::sha256_file(ck);
  const auto vision_sha = aima::sha256_file(vision);
  const auto host = host_name();
  std::cout << "{\"event\":\"start\",\"host\":" << quote(host)
            << ",\"model\":" << quote(model.u8string())
            << ",\"source_commit\":" << quote(AIMA_PORT_SOURCE_COMMIT)
            << ",\"upstream_commit\":" << quote(AIMA_PORT_UPSTREAM_COMMIT)
            << ",\"input_u32_sha256\":" << quote(prompt_sha)
            << ",\"ck_provider_sha256\":" << quote(provider_sha)
            << ",\"vision_image_sha256\":" << quote(vision_sha)
            << ",\"prompt_tokens\":8192,\"requested_outputs\":512";
  if (observation) {
    std::cout << ",\"observation_directory\":" << quote(std::filesystem::absolute(path("--observe-directory")).u8string())
              << ",\"observation_output_index\":" << args.at("--observe-output-index")
              << ",\"observation_linear_layer\":" << args.at("--observe-linear-layer")
              << ",\"observation_output_only\":true,\"diagnostic_timings_only\":true";
  }
  std::cout << "}" << std::endl;
#ifdef AIMA_PORT_GB10_CONVOLUTION
  aima_port::ConvolutionSiluOwner convolution_silu;
#endif
#ifdef AIMA_PORT_GB10_GDN
  aima_port::Gb10GdnOwner gdn;
#endif
  aima::NativeResidentEngineOptions options;
  options.weights.model_dir = model;
  options.weights.native_report = report;
  options.ck_provider = ck;
  options.vision_attention_image = vision;
  options.prompt_tokens = 8192;
  options.cache_capacity = 9216;
  // Keep the default resident cache backing in command-to-ready accounting.
  aima::NativeResidentEngine engine;
  const auto loaded = engine.load(options);
  const double ready_ms = elapsed(entered);
  std::cout << "{\"event\":\"ready\",\"command_to_ready_ms\":" << ready_ms
            << ",\"engine_load_ms\":" << loaded.command_to_ready_wall_ms
            << ",\"model_payload_bytes\":" << loaded.model_payload_bytes
            << ",\"visual_payload_bytes\":" << loaded.visual_model_payload_bytes
            << ",\"gpu_arch\":" << quote(loaded.gpu_arch)
            << ",\"aot_modules\":" << loaded.aot_loaded_modules
            << ",\"prefix_cache_bytes\":" << loaded.exact_prefix_cache_bytes
            << "}" << std::endl;
  std::vector<std::uint32_t> callbacks;
  callbacks.reserve(512);
  double first_callback_ms = 0, last_callback_ms = 0;
  const auto started = Clock::now();
  request.token_callback = [&](std::uint32_t token, std::size_t index) {
    if (index != callbacks.size()) throw std::runtime_error("Callback index is not sequential");
    last_callback_ms = elapsed(started);
    if (index == 0) first_callback_ms = last_callback_ms;
    callbacks.push_back(token);
    std::cout << "{\"event\":\"token\",\"index\":" << index
              << ",\"token\":" << token << ",\"elapsed_ms\":" << last_callback_ms
              << "}" << std::endl;
    return true;
  };
  const auto result = engine.run(request);
  if (!std::isfinite(result.first_token_raw_logit)) throw std::runtime_error("First logit is not finite");
  const bool complete = result.output_token_ids.size() == 512 &&
      callbacks == result.output_token_ids && result.oracle_tensor_reads == 0 &&
      result.first_token_certified && result.all_decode_tokens_certified &&
      !result.stopped && !result.client_cancelled &&
      std::isfinite(result.first_token_raw_logit) && engine.request_count() == 1;
  std::cout << "{\"event\":\"result\",\"complete\":" << (complete ? "true" : "false")
            << ",\"host\":" << quote(host)
            << ",\"model\":" << quote(model.u8string())
            << ",\"source_commit\":" << quote(AIMA_PORT_SOURCE_COMMIT)
            << ",\"upstream_commit\":" << quote(AIMA_PORT_UPSTREAM_COMMIT)
            << ",\"input_u32_sha256\":" << quote(prompt_sha)
            << ",\"first_token_raw_logit\":" << result.first_token_raw_logit
            << ",\"ttft_ms\":" << first_callback_ms
            << ",\"engine_prefill_ms\":" << result.prefill_wall_ms
            << ",\"tpot_ms\":" << (last_callback_ms - first_callback_ms) / 511.0
            << ",\"request_wall_ms\":" << result.request_wall_ms
            << ",\"command_to_ready_ms\":" << ready_ms
            << ",\"oracle_tensor_reads\":" << result.oracle_tensor_reads
            << ",\"callback_count\":" << callbacks.size()
            << ",\"first_token_certified\":" << (result.first_token_certified ? "true" : "false")
            << ",\"all_decode_tokens_certified\":" << (result.all_decode_tokens_certified ? "true" : "false")
            << ",\"prompt_execution\":" << quote(result.prompt_execution)
            << ",\"prefix_cache_lookup\":" << quote(result.prefix_cache_lookup)
            << ",\"output_token_ids\":[";
  for (std::size_t i = 0; i < result.output_token_ids.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << result.output_token_ids[i];
  }
  std::cout << "],\"gb10_correctness_evaluated\":false}" << std::endl;
  return complete ? 0 : 6;
}
int entry(const std::vector<std::string>& args) {
  try { return run(args); }
  catch (const std::exception& error) {
    std::cerr << "{\"event\":\"error\",\"message\":"
              << aima_port::json_string(error.what()) << "}" << std::endl;
    return 2;
  }
}
}
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i) args.push_back(std::filesystem::path(argv[i]).u8string());
  return entry(args);
}
#else
int main(int argc, char** argv) {
  return entry(std::vector<std::string>(argv + 1, argv + argc));
}
#endif
