// Replay the real cold Request from original GB10 target hidden chunks.
// This component probe does not run a target model or establish inference acceptance.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <climits>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "native/providers/gdn/sm121_mtp_request.h"

namespace {
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
std::string sha256(const void* data, size_t bytes) {
    if (bytes > ULONG_MAX) throw std::runtime_error("hash extent");
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD length = 0, copied = 0;
    unsigned char digest[32];
    std::vector<unsigned char> object;
    const auto ok = [](NTSTATUS status) {
        if (status < 0) throw std::runtime_error("BCrypt SHA256");
    };
    try {
        ok(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0));
        ok(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
            reinterpret_cast<PUCHAR>(&length), sizeof(length), &copied, 0));
        object.resize(length);
        ok(BCryptCreateHash(algorithm, &hash, object.data(), length, nullptr, 0, 0));
        ok(BCryptHashData(hash, const_cast<PUCHAR>(static_cast<const unsigned char*>(data)),
            static_cast<ULONG>(bytes), 0));
        ok(BCryptFinishHash(hash, digest, sizeof(digest), 0));
        BCryptDestroyHash(hash); hash = nullptr;
        BCryptCloseAlgorithmProvider(algorithm, 0); algorithm = nullptr;
    } catch (...) {
        if (hash) BCryptDestroyHash(hash);
        if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
        throw;
    }
    std::ostringstream text;
    for (unsigned char value : digest)
        text << std::hex << std::setfill('0') << std::setw(2) << unsigned(value);
    return text.str();
}
struct Entry { std::string path, digest; size_t offset = 0, bytes = 0; };
using Plan = std::map<std::string, Entry>;
template<class T> std::vector<T> read(const Plan& plan, const std::string& role, size_t count) {
    const auto& entry = plan.at(role);
    if (!count || count > (size_t(1) << 30u) / sizeof(T) || entry.bytes != count * sizeof(T))
        throw std::runtime_error("input extent: " + role);
    std::ifstream file(entry.path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || uint64_t(file.tellg()) > (uint64_t(8) << 30u) ||
        entry.offset > uint64_t(file.tellg()) || entry.bytes > uint64_t(file.tellg()) - entry.offset ||
        (!entry.offset && entry.bytes != uint64_t(file.tellg())))
        throw std::runtime_error("file extent: " + role);
    std::vector<T> result(count);
    file.seekg(static_cast<std::streamoff>(entry.offset));
    file.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(entry.bytes));
    if (!file || sha256(result.data(), entry.bytes) != entry.digest)
        throw std::runtime_error("input hash: " + role);
    return result;
}
struct Buffer {
    unsigned char* raw = nullptr;
    size_t bytes = 0;
    std::string digest;
    explicit Buffer(size_t count) : bytes(count) {
        check(hipMalloc(reinterpret_cast<void**>(&raw), bytes + 512u));
        check(hipMemset(raw, 0xa5, bytes + 512u));
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (raw && hipDeviceSynchronize() == hipSuccess) (void)hipFree(raw); }
    void* data() const { return raw + 256u; }
    template<class T> void upload(const std::vector<T>& values) {
        if (values.size() * sizeof(T) != bytes) throw std::runtime_error("upload extent");
        digest = sha256(values.data(), bytes);
        check(hipMemcpy(data(), values.data(), bytes, hipMemcpyHostToDevice));
    }
    size_t verify() const {
        std::vector<unsigned char> values(bytes + 512u);
        check(hipMemcpy(values.data(), raw, values.size(), hipMemcpyDeviceToHost));
        size_t errors = sha256(values.data() + 256u, bytes) != digest;
        for (size_t i = 0; i < 256u; ++i)
            errors += (values[i] != 0xa5u) + (values[bytes + 256u + i] != 0xa5u);
        return errors;
    }
};
struct Inputs {
    std::vector<std::unique_ptr<Buffer>> buffers;
    template<class T> const T* role(const Plan& plan, const std::string& name, size_t count) {
        auto values = read<T>(plan, name, count);
        auto buffer = std::make_unique<Buffer>(count * sizeof(T));
        buffer->upload(values);
        const auto* pointer = static_cast<const T*>(buffer->data());
        buffers.push_back(std::move(buffer));
        return pointer;
    }
    size_t verify() const {
        size_t errors = 0;
        for (const auto& buffer : buffers) errors += buffer->verify();
        return errors;
    }
};
struct OriginalWeights final : qrt_sm121_mtp::ModelWeightSource {
    Inputs inputs;
    std::array<qrt_sm121_mtp::ModelTensorView, qrt_sm121_mtp::model_weight_specs.size()> views{};
    explicit OriginalWeights(const Plan& plan) {
        size_t index = 0;
        for (const auto& spec : qrt_sm121_mtp::model_weight_specs) {
            const auto* pointer = inputs.role<uint16_t>(plan, "weight/" + std::string(spec.name), spec.bytes() / 2u);
            views[index++] = {spec.name, pointer, spec.rank, spec.shape, spec.bytes(), 1u, true, true};
        }
    }
    uint64_t epoch() const noexcept override { return 1u; }
    bool tensor(const char* name, qrt_sm121_mtp::ModelTensorView* output) const override {
        if (!name || !output) return false;
        for (const auto& view : views) if (!std::strcmp(view.name, name)) {
            *output = view; return true;
        }
        return false;
    }
};
struct TableOwner {
    Inputs inputs;
    qrt_sm121_mtp::DrafterTables tables;
    explicit TableOwner(const Plan& plan) {
        tables = {inputs.role<unsigned char>(plan, "table/rsqrt", 17301808u),
            inputs.role<uint16_t>(plan, "table/rope", 262144u * 64u), 262144u,
            inputs.role<unsigned char>(plan, "table/exp2", 183174448u),
            inputs.role<unsigned char>(plan, "table/reciprocal", 8388640u),
            {inputs.role<uint16_t>(plan, "table/silu", 65536u + 12u) + 12u,
             inputs.role<uint16_t>(plan, "table/sigmoid", 65536u),
             inputs.role<uint32_t>(plan, "table/router-exp", 1u << 23u)}};
    }
};
struct Comparison {
    size_t elements = 0, mismatches = 0, first = 0;
    uint32_t actual = 0, expected = 0;
    void add(uint32_t got, uint32_t wanted, size_t index) {
        ++elements;
        if (got == wanted) return;
        if (!mismatches) { first = index; actual = got; expected = wanted; }
        ++mismatches;
    }
};
void emit(const std::string& name, const Comparison& value) {
    std::cout << '"' << name << "\":{\"elements\":" << value.elements
        << ",\"mismatches\":" << value.mismatches << ",\"first_difference\":";
    if (value.mismatches)
        std::cout << "{\"index\":" << value.first << ",\"actual_bits\":" << value.actual
            << ",\"expected_bits\":" << value.expected << '}';
    else std::cout << "null";
    std::cout << '}';
}
std::vector<uint16_t> copy(qrt_sm121_mtp::Request& owner, const uint16_t* pointer, size_t count) {
    if (!pointer || !count) throw std::runtime_error("unpublished device observation");
    std::vector<uint16_t> values(count);
    const auto status = hipMemcpy(values.data(), pointer, count * 2u, hipMemcpyDeviceToHost);
    if (status != hipSuccess) {
        owner.quarantine_borrower(status);
        check(status);
    }
    return values;
}
void complete(const qrt_sm121_mtp::PromptStep& step) {
    if (step.status != hipSuccess || step.completion_unknown)
        throw std::runtime_error(std::string(step.stage) + ": " + hipGetErrorString(step.status));
}
uint32_t bits(float value) { uint32_t word = 0; std::memcpy(&word, &value, sizeof(word)); return word; }
} // namespace

int main(int argc, char** argv) try {
    using namespace qrt_sm121_mtp;
    if (argc != 5) throw std::runtime_error("input_plan prompt_tokens original_first_target_token split1024_pre_fc_norm");
    const auto number = [](const char* value, unsigned maximum) {
        const std::string text(value);
        if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("integer");
        size_t used = 0;
        const auto n = std::stoul(text, &used);
        if (used != text.size() || n > maximum) throw std::runtime_error("integer bound");
        return static_cast<unsigned>(n);
    };
    const unsigned prompt_rows = number(argv[2], 32768u), first_target = number(argv[3], head_vocabulary - 1u);
    const bool split1024 = number(argv[4], 1u) != 0u;
    if (prompt_rows != 7169u && prompt_rows != 8192u && prompt_rows != 16384u && prompt_rows != 17408u)
        throw std::runtime_error("original component prompt shape");
    const unsigned chunk_count = (prompt_rows + 8191u) / 8192u;
    Plan plan;
    std::ifstream file(argv[1]);
    std::string label;
    Entry entry;
    while (file >> std::quoted(label) >> std::quoted(entry.path) >> entry.offset >> entry.bytes >> entry.digest) {
        if (label.empty() || label.size() > 160u || entry.path.empty() || entry.path.size() > 2048u ||
            entry.digest.size() != 64u || entry.digest.find_first_not_of("0123456789abcdef") != std::string::npos ||
            !entry.bytes || entry.bytes > (uint64_t(1) << 30u) || plan.size() >= 64u ||
            !plan.emplace(label, entry).second) throw std::runtime_error("original input plan");
    }
    if (!file.eof() || plan.size() != model_weight_specs.size() + 10u + 4u * chunk_count)
        throw std::runtime_error("incomplete original plan");
    const auto prompt = read<uint32_t>(plan, "prompt-ids", prompt_rows);
    auto original = std::make_shared<OriginalWeights>(plan);
    TableOwner tables(plan);
    // Inputs and their guards are initialized on the default stream. Drain
    // those writes before exercising the independent nonblocking stream.
    check(hipDeviceSynchronize());
    ModelWeights weights;
    const auto prepared = weights.prepare(original, 1u);
    if (prepared.status != hipSuccess || prepared.completion_unknown)
        throw std::runtime_error(std::string(prepared.stage) + ": model pack");
    const auto expected_hidden = read<uint16_t>(plan, "final-hidden", 2048u);
    const auto expected_logits = read<uint16_t>(plan, "final-logits", head_vocabulary);
    HeadBest expected_best;
    for (unsigned token = 0; token < head_vocabulary; ++token)
        if (!head_candidate(expected_logits[token], token, &expected_best))
            throw std::runtime_error("nonfinite original draft logits");
    std::map<std::string, Comparison> comparisons;
    unsigned intermediate_boundaries = 0, checkpoint_restores = 0, rejected_postfinal_chunks = 0;
    size_t maximum_owner_bytes = 0, maximum_inputs_bytes = 0;
    for (unsigned mode = 0; mode < 2u; ++mode) {
        hipStream_t stream = nullptr;
        if (mode) check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        {
            Request request;
            TargetFrontier frontier{&request, 17u, 1u, prompt.data(), 0u, first_target};
            for (unsigned first = 0, ordinal = 0; first < prompt_rows; ++ordinal) {
                const unsigned rows = std::min(8192u, prompt_rows - first);
                const std::string role = "chunk-" + std::to_string(ordinal) + '/';
                const auto hidden = read<uint16_t>(plan, role + "target-hidden", size_t(rows) * 2048u);
                const auto ids = read<uint32_t>(plan, role + "shifted-ids", rows);
                std::vector<float> normalized(hidden.size());
                for (size_t i = 0; i < hidden.size(); ++i) {
                    const uint32_t word = uint32_t(hidden[i]) << 16u;
                    std::memcpy(&normalized[i], &word, sizeof(word));
                }
                qrt_mtp_target_rows::PrefillRows batch(prompt.data(), prompt.size(), first, rows);
                // Original vLLM discards intermediate target samples. Use its
                // observed backup token only as the unpublished frontier marker.
                // The actual native shift independently derives that backup.
                const uint32_t marker = batch.discarded_prefill() ? prompt.back() : first_target;
                if (!batch.stage(batch.local_rows(), normalized, marker) || !batch.publish(marker))
                    throw std::runtime_error("original target chunk publication");
                for (unsigned row = 0; row < rows; ++row)
                    comparisons["shifted-input-ids"].add(batch.shifted_tokens()[row], ids[row], first + row);
                for (size_t i = 0; i < hidden.size(); ++i)
                    comparisons["target-hidden-roundtrip"].add(batch.hidden()[i], hidden[i], size_t(first) * 2048u + i);
                frontier.processed_count = first + rows; frontier.current_token = marker;
                if (first) complete(request.append_prefill_chunk(batch, frontier, stream));
                else if (chunk_count > 1u)
                    complete(request.seed_prefill_chunks(batch, frontier, weights.binding(1u), tables.tables, prompt_rows + 32u, stream, 1024u, split1024));
                else complete(request.seed(batch, frontier, weights.binding(1u), tables.tables, prompt_rows + 32u, stream, 1024u, split1024));
                maximum_owner_bytes = std::max(maximum_owner_bytes, request.allocated_bytes());
                maximum_inputs_bytes = std::max(maximum_inputs_bytes, request.input_allocated_bytes());
                if (request.committed_tokens() != first + rows || request.retained_tokens() != first + rows || request.quarantined())
                    throw std::runtime_error("completed chunk extent");
                if (first + rows < prompt_rows) {
                    RequestCheckpoint saved;
                    TargetBatch next{123u, 1u, {55u, 66u}};
                    if (request.matches(frontier) || request.cache_data() || request.proposal(1u).rows ||
                        request.begin(frontier, 2u, &next) || next.first_position != 123u ||
                        request.save(&saved, frontier, stream).status != hipErrorInvalidValue || saved.tokens())
                        throw std::runtime_error("intermediate chunk escaped");
                    ++intermediate_boundaries;
                } else {
                    if (!request.matches(frontier)) throw std::runtime_error("final original frontier");
                    const auto result = request.append_prefill_chunk(batch, frontier, stream);
                    if (result.status != hipErrorInvalidValue || result.completion_unknown)
                        throw std::runtime_error("duplicate final chunk accepted");
                    ++rejected_postfinal_chunks;
                }
                first += rows;
            }
            const auto proposal = request.proposal(1u);
            const auto observation = request.observation(1u);
            if (proposal.status != hipSuccess || proposal.completion_unknown || proposal.rows != 1u ||
                proposal.first_position + 1u != prompt_rows || observation.rows != 1u ||
                observation.first_position != proposal.first_position)
                throw std::runtime_error("complete original proposal extent");
            const auto actual_hidden = copy(request, observation.final_hidden, 2048u);
            const auto actual_logits = copy(request, observation.vocabulary_logits, head_vocabulary);
            for (size_t i = 0; i < actual_hidden.size(); ++i)
                comparisons["final-hidden"].add(actual_hidden[i], expected_hidden[i], i);
            for (size_t i = 0; i < actual_logits.size(); ++i)
                comparisons["full-vocabulary-logits"].add(actual_logits[i], expected_logits[i], i);
            comparisons["sampled-token"].add(proposal.tokens[0], expected_best.token, mode);
            comparisons["sampled-logit"].add(bits(proposal.logits[0]), bits(expected_best.logit), mode);
            const auto compare_cache = [&](Request& owner, const char* check_name) {
                for (unsigned first = 0, ordinal = 0; first < prompt_rows; ++ordinal) {
                    const unsigned rows = std::min(8192u, prompt_rows - first);
                    const std::string role = "chunk-" + std::to_string(ordinal) + '/';
                    const auto keys = read<uint16_t>(plan, role + "k", size_t(rows) * 512u);
                    const auto values = read<uint16_t>(plan, role + "v", size_t(rows) * 512u);
                    const auto actual = copy(owner, owner.cache_data() + size_t(first) * 1024u, size_t(rows) * 1024u);
                    for (unsigned row = 0; row < rows; ++row) for (unsigned column = 0; column < 1024u; ++column) {
                        const uint16_t wanted = (column < 512u ? keys : values)[size_t(row) * 512u + column % 512u];
                        comparisons[check_name].add(actual[size_t(row) * 1024u + column], wanted,
                            size_t(first + row) * 1024u + column);
                    }
                    first += rows;
                }
            };
            compare_cache(request, "complete-cache");
            RequestCheckpoint saved;
            complete(request.save(&saved, frontier, stream));
            if (!saved.matches(frontier) || saved.tokens() != prompt_rows)
                throw std::runtime_error("paired checkpoint publication");
            Request restored;
            complete(restored.restore(saved, frontier, prompt_rows + 32u, stream));
            compare_cache(restored, "restored-complete-cache");
            TargetBatch next;
            if (!restored.begin(frontier, 32u, &next) || next.first_position != prompt_rows || next.rows != 2u ||
                next.inputs[0] != first_target || next.inputs[1] != proposal.tokens[0] || !restored.abort(1u))
                throw std::runtime_error("paired checkpoint next draft");
            ++checkpoint_restores;
            check(hipStreamSynchronize(stream));
        }
        if (mode) check(hipStreamDestroy(stream));
    }
    const size_t immutable_errors = original->inputs.verify() + tables.inputs.verify();
    size_t errors = immutable_errors, values = 0;
    std::cout << "{\"kind\":\"original_cold_mtp_request_component_replay\",\"prompt_tokens\":" << prompt_rows
        << ",\"split1024_pre_fc_norm\":" << (split1024 ? "true" : "false")
        << ",\"chunks\":" << chunk_count << ",\"configurations\":2,\"streams\":[\"default\",\"nonblocking\"]"
        << ",\"intermediate_publication_boundaries\":" << intermediate_boundaries
        << ",\"checkpoint_restores\":" << checkpoint_restores
        << ",\"rejected_postfinal_chunks\":" << rejected_postfinal_chunks
        << ",\"maximum_owner_bytes\":" << maximum_owner_bytes << ",\"maximum_input_bytes\":" << maximum_inputs_bytes
        << ",\"first_target_token\":" << first_target << ",\"original_final_draft_token\":" << expected_best.token
        << ",\"original_final_draft_logit\":" << std::setprecision(9) << expected_best.logit << ",\"checks\":{";
    bool comma = false;
    for (const auto& entry : comparisons) {
        if (comma) std::cout << ',';
        comma = true; emit(entry.first, entry.second);
        values += entry.second.elements; errors += entry.second.mismatches;
    }
    std::cout << "},\"compared_values\":" << values << ",\"immutable_input_and_guard_errors\":" << immutable_errors
        << ",\"passed\":" << (errors ? "false" : "true")
        << ",\"original_target_hidden_drives_component_only\":true,\"original_backup_is_discarded_frontier_marker\":true"
        << ",\"expected_kv_and_logits_are_compute_input\":false,\"model_inference\":false"
        << ",\"inference_acceptance\":false,\"performance_acceptance\":false}\n";
    return errors ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 2;
}
