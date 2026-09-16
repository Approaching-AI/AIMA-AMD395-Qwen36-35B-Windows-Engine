#include "../../native/providers/ck_fmha/blackwell_attention.h"
#include "../../native/providers/moe_accumulator/sm121_compact_byte_core.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
namespace {
constexpr unsigned guard=64u;
void check(hipError_t s) { if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
struct Device {
    void* pointer = nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&pointer, bytes)); }
    ~Device() { if (pointer && hipFree(pointer) != hipSuccess) std::abort(); }
    template<class T> T* as() { return static_cast<T*>(pointer); }
};
template<class T> void upload(Device& d, const std::vector<T>& values) {
    check(hipMemcpy(d.pointer, values.data(), values.size()*sizeof(T), hipMemcpyHostToDevice));
}
template<class T> std::vector<T> download(Device& d, size_t count) {
    std::vector<T> values(count);
    check(hipMemcpy(values.data(), d.pointer, count*sizeof(T), hipMemcpyDeviceToHost));
    return values;
}
template<class T> void unchanged(Device& d, const std::vector<T>& expected) {
    const auto actual = download<T>(d, expected.size());
    if (std::memcmp(actual.data(), expected.data(), expected.size()*sizeof(T)))
        throw std::runtime_error("byte residue input, encoding or redzone changed");
}
void finish() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("byte residue completion deadline");
        // Keep the explicit deadline without quantizing each short GPU slab
        // through a fixed host sleep. The benchmark reports completed host time.
        std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
double elapsed(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-begin).count();
}
namespace byte_probe {
namespace core = qrt_sm121_compact_byte_core;
constexpr unsigned tiles = 8192u;
// Match the original116-byte row stride and actual high/low field offsets.
// The canonical metadata is not consumed by this integer-core-only timing.
struct OriginalRow { uint16_t unused_original[18]{};int high[4]{},low[4]{};uint32_t unused_tail[12]{}; };
static_assert(sizeof(OriginalRow)==116u && offsetof(OriginalRow,high)==36u && offsetof(OriginalRow,low)==52u);
struct Cell { int64_t recovered,integer; float approximate; int32_t residue; uint32_t accepted,padding; };
template<unsigned Headroom>
__global__ void prepare(const uint16_t* input, core::Row* output, OriginalRow* control, unsigned count) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    const core::Row result=core::prepare(input+size_t(row)*16u);output[row]=result;
    OriginalRow old{};
    for(unsigned i=0u;i<16u;++i){const uint16_t x=qrt_sm121_integer_core::encode(input[size_t(row)*16u+i],core::unit(result));old.high[i/4u]=int(uint32_t(old.high[i/4u])|uint32_t(x>>8u)<<((i&3u)*8u));old.low[i/4u]=int(uint32_t(old.low[i/4u])|uint32_t(x&255u)<<((i&3u)*8u));}
    control[row]=old;
}
template<unsigned Headroom>
__global__ void compare(const core::Row* rows,const OriginalRow* control, Cell* output) {
    const unsigned lane = threadIdx.x, base = blockIdx.x * 32u;
    const auto integer = qrt_blackwell_attention::blackwell_integer_prepared_products(control[base+lane%16u],control[base+16u+lane%16u]);
    const auto products = core::products(rows[base + lane % 16u], rows[base + 16u + lane % 16u]);
#pragma unroll
    for (unsigned i = 0u; i < 8u; ++i) {
        const unsigned row = 2u * i + lane / 16u, column = lane % 16u;
        Cell result{};result.approximate=products.approximate[i];result.residue=products.residue[i];
        result.integer=int64_t(integer.value[0][i])*65536+(int64_t(integer.value[1][i])+integer.value[2][i])*256+integer.value[3][i];
        int32_t recovered=0;result.accepted=unsigned(core::small::recover(result.approximate,uint32_t(result.residue),&recovered));result.recovered=recovered;
        output[blockIdx.x * 256u + row * 16u + column] = result;
    }
}
double half(uint16_t value) {
    const unsigned exponent = (value >> 10u) & 31u;
    return exponent ? ((value & 32768u) ? -1.0 : 1.0) * std::ldexp(double(1024u + (value & 1023u)), int(exponent) - 25) : 0.0;
}

template<unsigned Headroom> void run() {
    uint32_t state = 0x8192395u;
    auto random = [&]() { state ^= state << 13u; state ^= state >> 17u; state ^= state << 5u; return state; };
    std::vector<uint16_t> values(tiles * 512u + 2u * guard, 0x5a5au);
    std::vector<core::Row> expected_rows(tiles * 32u);
    std::vector<int> integers(tiles * 512u);
    std::vector<OriginalRow> original_rows(tiles*32u);
    size_t eligible_rows = 0u;
    for (unsigned row = 0u; row < expected_rows.size(); ++row) {
        const unsigned tile = row / 32u, mode = tile % 9u, exponent = 1u + random() % 246u;
        bool valid = true, nonzero = false; int maximum = 0;
        for (unsigned i = 0u; i < 16u; ++i) {
            uint16_t value = uint16_t((random() & 0x807fu) | ((exponent + random() % 8u) << 7u));
            if (tile < 128u) value = uint16_t(row * 16u + i); // All original BF16 bit patterns.
            else if (mode == 0u) value = uint16_t(0x3fffu | (random() & 0x8000u));
            else if (mode == 1u) value = uint16_t(0x3fffu | ((i & 1u) && row % 32u < 16u ? 0x8000u : 0u));
            else if (mode == 2u) value = i == row % 16u ? uint16_t(value & 0x7fffu) : 0u;
            else if (mode == 3u) value = uint16_t((random() & 0x807fu) | ((1u + random() % 254u) << 7u));
            else if (mode == 4u) value = uint16_t((random() & 0x807fu) | (127u << 7u));
            else if (mode == 5u) value = uint16_t((random() & 127u) | 0xbf80u);
            else if (mode == 6u && row % 32u < 16u) value = 0x8000u;
            values[guard + row * 16u + i] = value;
            if (value & 0x7fffu) {
                const int e = int((value >> 7u) & 255u); nonzero = true;
                valid = valid && e != 0 && e != 255; maximum = std::max(maximum, e);
            }
        }
        const int unit = !valid ? -1 : !nonzero ? 127 : maximum > int(Headroom+1u) ? maximum-int(Headroom) : 1;
        expected_rows[row]=core::prepare(values.data()+guard+row*16u); eligible_rows += unsigned(unit >= 0);
        if (core::unit(expected_rows[row]) != unit) throw std::runtime_error("native probe unit differs from independent maximum");
        for (unsigned i = 0u; i < 16u; ++i) {
            const uint16_t value = values[guard + row * 16u + i];
            const int magnitude = unit < 0 || !(value & 0x7fffu) ? 0 :
                int(std::ldexp(double(128u + (value & 127u)), int((value >> 7u) & 255u) - unit));
            const int expected = value & 32768u ? -magnitude : magnitude;
            integers[row * 16u + i] = expected;
            const uint16_t raw=uint16_t(expected);const unsigned word=i/4u,shift=(i&3u)*8u;
            original_rows[row].high[word]=int(uint32_t(original_rows[row].high[word])|uint32_t(raw>>8u)<<shift);
            original_rows[row].low[word]=int(uint32_t(original_rows[row].low[word])|uint32_t(raw&255u)<<shift);
            if (magnitude > int(255u<<Headroom) || half(expected_rows[row].half[i]) != expected ||
                ((uint32_t(expected_rows[row].low[word])>>shift)&255u)!=(uint32_t(expected)&255u))
                throw std::runtime_error("native probe exact FP16 core encoding failed");
        }
    }
    std::vector<unsigned char> encoded(expected_rows.size() * sizeof(core::Row) + 2u * guard, 0xa5u);
    std::memcpy(encoded.data() + guard, expected_rows.data(), expected_rows.size() * sizeof(core::Row));
    Device input(values.size() * 2u), prepared(encoded.size()), output((size_t(tiles) * 256u + 2u * guard) * sizeof(Cell));
    std::vector<OriginalRow> control_bytes(original_rows.size()+2u*guard);
    std::memset(control_bytes.data(),0xa5,control_bytes.size()*sizeof(OriginalRow));
    std::copy(original_rows.begin(),original_rows.end(),control_bytes.begin()+guard);
    Device controls(control_bytes.size()*sizeof(OriginalRow));check(hipMemset(controls.pointer,0xa5,control_bytes.size()*sizeof(OriginalRow)));
    upload(input, values); check(hipMemset(prepared.pointer, 0xa5, encoded.size()));
    check(hipMemset(output.pointer, 0xa5, (size_t(tiles) * 256u + 2u * guard) * sizeof(Cell))); finish();
    const auto begin = std::chrono::steady_clock::now();
    hipLaunchKernelGGL((prepare<Headroom>), dim3((expected_rows.size() + 255u) / 256u), dim3(256u), 0u, nullptr,
        input.as<uint16_t>() + guard, reinterpret_cast<core::Row*>(prepared.as<unsigned char>() + guard), controls.as<OriginalRow>()+guard, unsigned(expected_rows.size()));
    check(hipGetLastError()); finish(); const double preparation_ms = elapsed(begin);
    unchanged(prepared, encoded);unchanged(controls,control_bytes);
    hipLaunchKernelGGL((compare<Headroom>), dim3(tiles), dim3(32u), 0u, nullptr,
        reinterpret_cast<core::Row*>(prepared.as<unsigned char>() + guard), controls.as<OriginalRow>()+guard, output.as<Cell>() + guard);
    check(hipGetLastError()); finish();
    const auto actual = download<Cell>(output, size_t(tiles) * 256u + 2u * guard);
    size_t residue_bad=0u,recovery_bad=0u,bound_bad=0u,rejected=0u,integer_bad=0u; double max_error = 0.0;
    for (size_t cell = 0u; cell < actual.size(); ++cell) {
        if (cell < guard || cell >= size_t(tiles) * 256u + guard) {
            const auto* bytes = reinterpret_cast<const unsigned char*>(&actual[cell]);
            if (std::any_of(bytes, bytes + sizeof(Cell), [](unsigned char b) { return b != 0xa5u; }))
                throw std::runtime_error("native modular core output guard changed");
            continue;
        }
        const unsigned index = unsigned(cell - guard), tile = index / 256u, row = index % 256u / 16u, column = index % 16u;
        int64_t expected = 0;
        for (unsigned k = 0u; k < 16u; ++k) expected += int64_t(integers[tile * 512u + row * 16u + k]) * integers[tile * 512u + 256u + column * 16u + k];
        const auto& value = actual[cell];
        const double error = std::abs(double(value.approximate) - double(expected)); max_error = std::max(max_error, error);
        bound_bad += unsigned(!(error < 128.0));
        residue_bad += unsigned((uint32_t(value.residue) & 255u) != (uint32_t(expected) & 255u));
        integer_bad+=unsigned(value.integer!=expected);rejected+=unsigned(!value.accepted);
        if (value.accepted && value.recovered != expected) {
            if (recovery_bad < 4u) std::printf("{\"kind\":\"compact_byte_core_counterexample\",\"headroom\":%u,\"cell\":%u,\"expected\":%lld,\"actual\":%lld,\"approximate\":%.12g,\"residue\":%d,\"accepted\":%u}\n", Headroom,index, static_cast<long long>(expected), static_cast<long long>(value.recovered), double(value.approximate), value.residue, value.accepted);
            ++recovery_bad;
        }
    }
    unchanged(input, values); unchanged(prepared, encoded);unchanged(controls,control_bytes);

    std::printf("{\"kind\":\"compact_byte_core_native\",\"headroom\":%u,\"tiles\":8192,\"dot_positions\":2097152,\"original_bf16_bit_patterns\":65536,\"eligible_rows\":%zu,\"exact_core_encodings_checked\":4194304,\"integer_control_mismatches\":%zu,\"residue_mismatches\":%zu,\"accepted_recovery_mismatches\":%zu,\"recovery_rejections\":%zu,\"conditional_bound_violations\":%zu,\"maximum_native_absolute_error\":%.9g,\"recovery_half_spacing\":128,\"initial_preparation_ms\":%.6f,\"prepared_row_bytes\":76,\"redzones_pass\":true,\"immutable_inputs\":true,\"hardware_error_bound_proven\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",Headroom,eligible_rows,integer_bad,residue_bad,recovery_bad,rejected,bound_bad,max_error,preparation_ms);
    std::fflush(stdout);
    if(integer_bad || residue_bad)throw std::runtime_error("integer residue differs from independent oracle");
    if(recovery_bad || rejected || bound_bad)throw std::runtime_error("conditional native recovery failed in finite probe");

}
}

}
int main() try {hipDeviceProp_t device{};check(hipGetDeviceProperties(&device,0));if(std::strncmp(device.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");byte_probe::run<5u>();return 0;} catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
