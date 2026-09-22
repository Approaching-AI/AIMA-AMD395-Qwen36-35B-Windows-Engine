// SPDX-License-Identifier: Apache-2.0
#include "aima/native_resident_engine.h"
#include "aima/sha256.h"
#include "aima_port_build_identity.h"
#include "probe_input.h"
#include <hip/hip_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <optional>
#include <sstream>
#ifdef AIMA_PORT_GB10_CONVOLUTION
#include "gb10_convolution.h"
#endif
#ifdef AIMA_PORT_GB10_GDN
#include "gb10_gdn.h"
#endif
#ifdef AIMA_PORT_GB10_PROJECTIONS
#include "gb10_projection.h"
#endif
#ifdef AIMA_PORT_GB10_PREFILL_PROJECTIONS
#include "gb10_prefill_projection.h"
#endif
#ifdef AIMA_PORT_GB10_NORMALIZATION
#include "gb10_normalization.h"
#include "gb10_decode_attention.h"
#endif
#ifdef AIMA_PORT_GB10_MOE
#include "gb10_moe.h"
#include "gb10_decode_moe.h"
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
              std::size_t layer, std::optional<std::size_t> full_layer = std::nullopt)
      : directory_(directory), output_index_(index), layer_(layer), full_layer_(full_layer) {
    if (full_layer_ && (*full_layer_ > 39 || *full_layer_ % 4 != 3 ||
                       index == 0 || index > 511))
      throw std::invalid_argument("Invalid full-attention observation selector");
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
              << ",\"selected_decode_output_index\":" << output_index_;
    if (full_layer_) manifest_ << ",\"full_attention_layer\":" << *full_layer_;
    else manifest_ << ",\"linear_layer\":" << layer_;
    manifest_ << ",\"output_only\":true}" << std::endl;
    if (!manifest_) throw std::runtime_error("Observation manifest write failed");
    total_ += bytes; ++count_;
  }
  void bind(aima::NativeResidentRequestOptions& request) {
    request.decode_layer_observer_output_index = output_index_;
    request.decode_linear_observer_layer_index = layer_;
    request.decode_layer_observer = [this](std::size_t boundary, const void* row) {
      if (boundary > 40) throw std::runtime_error("Invalid decode layer boundary");
      capture("decode-boundary-" + std::to_string(boundary), row, 2048 * 2, "bf16");
    };
    if (full_layer_) {
      request.decode_full_attention_observer = [this](
          const aima::NativeDecodeFullAttentionObservation& value) {
        if (value.layer_index != *full_layer_) return;
        if (value.cache_end != 8192 + output_index_ ||
            value.key_cache == nullptr || value.value_cache == nullptr)
          throw std::runtime_error("Full-attention observation cache extent differs");
        const auto bf16 = [this](const char* name, const void* pointer, std::size_t count) {
          capture(std::string("decode-full-") + name, pointer, count * 2, "bf16");
        };
        bf16("qkv", value.qkv_projection, 9216);
        bf16("q-rope", value.query, 4096);
        bf16("k-rope", value.current_key, 512);
        bf16("value", value.current_value, 512);
        // Each complete cache exceeds the existing per-file limit. Preserve
        // the exact token-major layout in an 8192-row prefix and decode tail.
        for (const auto& cache : {std::make_pair("cache-k", value.key_cache),
                                  std::make_pair("cache-v", value.value_cache)}) {
          bf16((std::string(cache.first) + "-prefill").c_str(), cache.second, 8192 * 512);
          bf16((std::string(cache.first) + "-decode").c_str(),
               static_cast<const unsigned char*>(cache.second) + 8192 * 512 * 2,
               output_index_ * 512);
        }
        bf16("context", value.attention_output, 4096);
        bf16("gated", value.gated_attention, 4096);
        bf16("output", value.projected_attention, 2048);
        bf16("attention-residual", value.attention_residual, 2048);
        bf16("post-attention-norm", value.post_attention_norm, 2048);
        bf16("shared-gate", value.shared_gate_logits, 1);
        bf16("shared-gate-up", value.shared_gate_up_projection, 1024);
        bf16("shared-activation", value.shared_activation, 512);
        bf16("shared-down", value.shared_down_projection, 2048);
        bf16("shared", value.shared_moe_output, 2048);
        bf16("router", value.router_logits, 256);
        capture("decode-full-router-weights", value.router_weights, 8 * 4, "f32");
        capture("decode-full-router-indices", value.router_indices, 8 * 4, "i32");
        bf16("routed-gate-up", value.routed_gate_up_projection, 8 * 1024);
        bf16("routed-activation", value.routed_activation, 8 * 512);
        bf16("routed-weighted", value.routed_weighted_expert_outputs, 8 * 2048);
        bf16("routed", value.routed_moe_output, 2048);
        bf16("combined", value.combined_moe_output, 2048);
      };
      return;
    }
    request.prefill_linear_state_observer = [this](std::size_t layer, const void* conv,
        std::uint64_t conv_bytes, const void* state, std::uint64_t state_bytes) {
      if (layer != layer_) return;
      capture("prefill-conv", conv, conv_bytes, "bf16");
      capture("prefill-state", state, state_bytes, "f32");
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
  std::optional<std::size_t> full_layer_;
  std::uint64_t total_ = 0;
};

int run(const std::vector<std::string>& argv) {
  const auto entered = Clock::now();
  std::cout.imbue(std::locale::classic());
  std::cout << std::setprecision(17);
  const auto args = aima_port::probe_arguments(argv);
  const char* first64_setting = std::getenv("AIMA_PORT_GDN_PREFILL_FIRST64");
  if (first64_setting && std::string(first64_setting) != "0" &&
      std::string(first64_setting) != "1")
    throw std::invalid_argument("First64 observation requires 0 or 1");
  const bool first64 = first64_setting && std::string(first64_setting) == "1";
  const auto full_layer = aima_port::full_attention_observation_layer(
      std::getenv("AIMA_PORT_OBSERVE_FULL_LAYER"), args.count("--observe-directory"), first64);
  if (first64 && !args.count("--observe-directory"))
    throw std::invalid_argument("First64 observation requires an output directory");
#ifndef AIMA_PORT_GB10_GDN
  if (first64) throw std::invalid_argument("First64 observation requires the GDN owner");
#endif
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
        aima_port::observation_number(args.at("--observe-linear-layer"), 39), full_layer);
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
              << ",\"observation_prefill_first64\":" << (first64 ? "true" : "false")
              << ",\"observation_output_only\":true,\"diagnostic_timings_only\":true";
    if (full_layer) std::cout << ",\"observation_full_attention_layer\":" << *full_layer;
  }
  std::cout << "}" << std::endl;
#ifdef AIMA_PORT_GB10_CONVOLUTION
  aima_port::ConvolutionSiluOwner convolution_silu;
#endif
#ifdef AIMA_PORT_GB10_GDN
  aima_port::Gb10GdnOwner gdn;
  if (observation && !full_layer) {
    aima_port::set_gdn_prefill_observer(
        aima_port::observation_number(args.at("--observe-linear-layer"), 39),
        [](const char* name, const void* device, std::size_t bytes, void* owner) {
          static_cast<Observation*>(owner)->capture(name, device, bytes, "bf16");
        }, observation.get(), first64);
  }
#endif
#ifdef AIMA_PORT_GB10_PROJECTIONS
  aima_port::Gb10ProjectionOwner projections(request.input_token_ids);
#endif
#ifdef AIMA_PORT_GB10_PREFILL_PROJECTIONS
  aima_port::Gb10PrefillProjectionOwner prefill_projections;
#endif
#ifdef AIMA_PORT_GB10_NORMALIZATION
  aima_port::Gb10NormalizationOwner normalization;
  aima_port::Gb10DecodeAttentionOwner decode_attention(9216);
  if (aima_port::gb10_prefill_terminal_only_enabled() && observation)
    throw std::invalid_argument("Terminal prefill trial requires observers to be disabled");
#endif
#ifdef AIMA_PORT_GB10_MOE
  aima_port::Gb10MoeOwner moe;
  aima_port::Gb10DecodeMoeOwner decode_moe;
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
#ifdef AIMA_PORT_GB10_PREFILL_PROJECTIONS
  aima_port::gb10_prefill_projection_profile_begin();
#endif
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
